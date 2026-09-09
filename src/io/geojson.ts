import { makeAnnotation } from "../annotate/store";
import {
  nextClassColor,
  ROI_CLASS_ID,
  ROI_COLOR,
  type Annotation,
  type AnnotationClass,
  type Geometry,
} from "../annotate/types";
import type { SlideMeta } from "../slide/types";

/**
 * GeoJSON import/export, shaped for QuPath round-tripping.
 *
 * Coordinates are level-0 slide pixels in the full vendor frame, which is what
 * QuPath, OpenSlide and every Python WSI tool already use — so a file written
 * here opens in QuPath with no transform, and vice versa.
 */

/** QuPath packs colour as a signed int32 of 0xFFRRGGBB. */
export function packRGB([r, g, b]: [number, number, number]): number {
  return ((255 << 24) | (r << 16) | (g << 8) | b) | 0;
}

export function unpackRGB(v: number): [number, number, number] {
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

interface QuPathClassification {
  name: string;
  colorRGB?: number;
}

interface FeatureProperties {
  objectType?: string;
  classification?: QuPathClassification | null;
  isLocked?: boolean;
  name?: string;
  measurements?: Record<string, number> | { name: string; value: number }[];
  slidecraft?: {
    id?: string;
    source?: "human" | "model";
    modelId?: string;
    confidence?: number;
    roiId?: string | null;
    isRoi?: boolean;
    createdAt?: number;
    updatedAt?: number;
  };
}

interface Feature {
  type: "Feature";
  id?: string;
  geometry: Geometry;
  properties: FeatureProperties;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Feature[];
  /** Foreign members; ignored by QuPath, used by us to restore context. */
  slidecraft?: {
    version: 1;
    slide?: { name: string; width: number; height: number; mppX: number | null };
    classes?: AnnotationClass[];
    exportedAt: string;
  };
}

// ------------------------------------------------------------------ export ---

export function toFeatureCollection(
  annotations: Annotation[],
  classes: AnnotationClass[],
  slide?: SlideMeta,
): FeatureCollection {
  const byId = new Map(classes.map((c) => [c.id, c]));

  const features: Feature[] = annotations.map((a) => {
    const isRoi = a.classId === ROI_CLASS_ID;
    const cls = isRoi ? null : a.classId ? byId.get(a.classId) : undefined;

    const classification: QuPathClassification | null = isRoi
      ? { name: "Region*", colorRGB: packRGB(ROI_COLOR) }
      : cls
        ? { name: cls.name, colorRGB: packRGB(cls.color) }
        : null;

    return {
      type: "Feature",
      id: a.id,
      geometry: a.geometry,
      properties: {
        objectType: a.objectType,
        classification,
        isLocked: a.locked,
        ...(a.name ? { name: a.name } : {}),
        ...(a.measurements ? { measurements: a.measurements } : {}),
        slidecraft: {
          id: a.id,
          source: a.source,
          ...(a.modelId ? { modelId: a.modelId } : {}),
          ...(a.confidence !== undefined ? { confidence: a.confidence } : {}),
          ...(a.roiId ? { roiId: a.roiId } : {}),
          ...(isRoi ? { isRoi: true } : {}),
          createdAt: a.createdAt,
          updatedAt: a.updatedAt,
        },
      },
    };
  });

  return {
    type: "FeatureCollection",
    features,
    slidecraft: {
      version: 1,
      ...(slide
        ? {
            slide: {
              name: slide.name,
              width: slide.levels[0].width,
              height: slide.levels[0].height,
              mppX: slide.mppX,
            },
          }
        : {}),
      classes,
      exportedAt: new Date().toISOString(),
    },
  };
}

// ------------------------------------------------------------------ import ---

export interface ImportResult {
  annotations: Annotation[];
  classes: AnnotationClass[];
  skipped: number;
}

const SUPPORTED = new Set(["Polygon", "MultiPolygon", "Point", "LineString"]);

function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "class";
}

function normaliseMeasurements(
  m: FeatureProperties["measurements"],
): Record<string, number> | undefined {
  if (!m) return undefined;
  // QuPath ≤0.3 wrote an array of {name, value}; newer versions write an object.
  if (Array.isArray(m)) {
    const out: Record<string, number> = {};
    for (const entry of m) if (entry && typeof entry.name === "string") out[entry.name] = entry.value;
    return Object.keys(out).length ? out : undefined;
  }
  return Object.keys(m).length ? m : undefined;
}

/**
 * Tolerant reader: accepts a FeatureCollection, a bare feature array, or a
 * single Feature, and invents classes for any classification it has not seen.
 */
export function fromGeoJSON(raw: unknown, existing: AnnotationClass[]): ImportResult {
  const root = raw as Partial<FeatureCollection> & { features?: unknown };
  const list: Feature[] = Array.isArray(raw)
    ? (raw as Feature[])
    : Array.isArray(root?.features)
      ? (root.features as Feature[])
      : root && (root as unknown as Feature).type === "Feature"
        ? [raw as Feature]
        : [];

  const classes: AnnotationClass[] = [...existing];
  const byName = new Map(classes.map((c) => [c.name.toLowerCase(), c]));
  const declared = (root as FeatureCollection)?.slidecraft?.classes;
  if (declared) {
    for (const c of declared) {
      if (!classes.some((e) => e.id === c.id)) {
        classes.push(c);
        byName.set(c.name.toLowerCase(), c);
      }
    }
  }

  const annotations: Annotation[] = [];
  let skipped = 0;

  for (const f of list) {
    const geometry = f?.geometry;
    if (!geometry || !SUPPORTED.has(geometry.type)) {
      skipped += 1;
      continue;
    }
    const props: FeatureProperties = f.properties ?? {};
    const extra = props.slidecraft ?? {};
    const classification = props.classification ?? null;

    let classId: string | null = null;
    if (extra.isRoi || classification?.name === "Region*") {
      classId = ROI_CLASS_ID;
    } else if (classification?.name) {
      const key = classification.name.toLowerCase();
      let cls = byName.get(key);
      if (!cls) {
        cls = {
          id: slugify(classification.name),
          name: classification.name,
          // Prefer the colour the file carries; otherwise take the next free
          // palette slot so imported classes stay visually distinct.
          color:
            typeof classification.colorRGB === "number"
              ? unpackRGB(classification.colorRGB)
              : nextClassColor(classes),
        };
        classes.push(cls);
        byName.set(key, cls);
      }
      classId = cls.id;
    }

    annotations.push(
      makeAnnotation(geometry, {
        id: extra.id ?? (typeof f.id === "string" ? f.id : undefined),
        classId,
        objectType: props.objectType === "detection" ? "detection" : "annotation",
        source: extra.source === "model" ? "model" : "human",
        modelId: extra.modelId,
        confidence: extra.confidence,
        roiId: extra.roiId ?? null,
        locked: props.isLocked === true,
        name: props.name,
        measurements: normaliseMeasurements(props.measurements),
        createdAt: extra.createdAt,
      }),
    );
  }

  return { annotations, classes, skipped };
}

// ------------------------------------------------------------------- files ---

export function downloadGeoJSON(fc: FeatureCollection, filename: string) {
  const blob = new Blob([JSON.stringify(fc)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".geojson") ? filename : `${filename}.geojson`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke on the next tick so the download has certainly started.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const isGeoJSONFile = (name: string) =>
  /\.(geojson|json)$/i.test(name);
