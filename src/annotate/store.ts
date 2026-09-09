import { create } from "zustand";
import Flatbush from "flatbush";
import {
  bboxOf,
  nextClassColor,
  type Annotation,
  type AnnotationClass,
  type Geometry,
} from "./types";

/**
 * Undo/redo is a stack of *patches*, not snapshots. A brush stroke that edits
 * one 50k-vertex polygon stores that one object, not a clone of the document,
 * so memory stays proportional to the edit rather than the slide.
 */
export interface Patch {
  label: string;
  added?: Annotation[];
  removed?: Annotation[];
  updated?: { before: Annotation; after: Annotation }[];
}

export function invertPatch(p: Patch): Patch {
  return {
    label: p.label,
    added: p.removed,
    removed: p.added,
    updated: p.updated?.map((u) => ({ before: u.after, after: u.before })),
  };
}

export type ToolId =
  | "pan"
  | "select"
  | "rectangle"
  | "polygon"
  | "freehand"
  | "brush"
  | "eraser"
  | "point"
  | "roi"
  | "segment";

/** Where a context menu or class picker was summoned, in client pixels. */
export interface MenuAnchor {
  x: number;
  y: number;
  /** Objects the action applies to; empty means "the canvas itself". */
  targetIds: string[];
}

export interface AnnotationState {
  /**
   * Mutated in place and paired with `version` for change notification.
   * Copy-on-write would be O(n) per edit, which a brush stroke cannot afford
   * at 10^5 objects; the renderer reads this imperatively anyway.
   */
  items: Map<string, Annotation>;
  version: number;

  classes: AnnotationClass[];
  activeClassId: string | null;
  selection: Set<string>;

  tool: ToolId;
  /** Open context menu, or null. Set by the tool controller, rendered by React. */
  contextMenu: MenuAnchor | null;
  /** Open "name this" class picker, or null. */
  classPicker: MenuAnchor | null;
  /** Transient explanation for an action that declined to do anything. */
  notice: string | null;
  brushRadius: number;
  simplifyTolerance: number;
  showAnnotations: boolean;
  /**
   * Classes hidden from the overlay.
   *
   * Distinct from deleting or locking: the objects are still there, still
   * exported, still counted. Tissue regions in particular are large and sit
   * under everything else, so being able to drop just those out of the view —
   * while keeping the nuclei drawn on top of them — is the difference between
   * being able to see your work and not.
   */
  hiddenClasses: Set<string>;
  fillOpacity: number;

  undoStack: Patch[];
  redoStack: Patch[];

  /** Identifies which slide the current document belongs to. */
  slideKey: string | null;

  apply: (patch: Patch) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  setTool: (t: ToolId) => void;
  setActiveClass: (id: string | null) => void;
  setBrushRadius: (r: number) => void;
  setFillOpacity: (o: number) => void;
  toggleAnnotations: () => void;
  toggleClassVisibility: (id: string) => void;
  setClassHidden: (id: string, hide: boolean) => void;
  /** Create a class, choosing a free palette colour. Returns the new class. */
  createClass: (
    name: string,
    color?: [number, number, number],
    opts?: { activate?: boolean },
  ) => AnnotationClass;
  /** Find a class by name (case-insensitive) or create it. */
  ensureClass: (name: string) => AnnotationClass;
  updateClass: (id: string, patch: Partial<Omit<AnnotationClass, "id">>) => void;
  /** Delete a class; annotations using it become unclassified (undoable). */
  deleteClass: (id: string) => void;
  reorderClass: (id: string, delta: number) => void;
  setClasses: (classes: AnnotationClass[]) => void;
  /** Assign a class to specific objects as one undoable step. */
  assignClass: (ids: string[], classId: string | null) => void;
  /** Set the free-text name of a single object. */
  renameAnnotation: (id: string, name: string) => void;
  /**
   * Replace geometries WITHOUT touching the undo stack. For live drags: the
   * caller keeps the originals and pushes one patch when the gesture ends,
   * so a drag is a single undo step rather than one per pointer event.
   */
  previewGeometries: (updates: { id: string; geometry: Geometry }[]) => void;
  /** Replace the whole selection at once. */
  setSelection: (ids: string[]) => void;

  setNotice: (n: string | null) => void;
  openContextMenu: (anchor: MenuAnchor | null) => void;
  openClassPicker: (anchor: MenuAnchor | null) => void;

  select: (ids: string[], additive?: boolean) => void;
  clearSelection: () => void;

  resetFor: (slideKey: string | null) => void;
  loadDocument: (slideKey: string, items: Annotation[], classes?: AnnotationClass[]) => void;
}

function applyPatchTo(items: Map<string, Annotation>, p: Patch) {
  p.removed?.forEach((a) => items.delete(a.id));
  p.updated?.forEach((u) => items.set(u.after.id, u.after));
  p.added?.forEach((a) => items.set(a.id, a));
}

export const useAnnotations = create<AnnotationState>((set, get) => ({
  items: new Map(),
  version: 0,
  classes: [],
  activeClassId: null,
  selection: new Set(),
  tool: "pan",
  contextMenu: null,
  classPicker: null,
  notice: null,
  brushRadius: 60,
  simplifyTolerance: 1.5,
  showAnnotations: true,
  hiddenClasses: new Set<string>(),
  fillOpacity: 0.28,
  undoStack: [],
  redoStack: [],
  slideKey: null,

  apply: (patch) => {
    const { items, undoStack } = get();
    applyPatchTo(items, patch);
    set({
      version: get().version + 1,
      undoStack: [...undoStack.slice(-199), patch],
      redoStack: [],
        });
  },

  undo: () => {
    const { undoStack, redoStack, items, selection } = get();
    if (undoStack.length === 0) return;
    const patch = undoStack[undoStack.length - 1];
    applyPatchTo(items, invertPatch(patch));
    // Anything the undo removed must not stay selected.
    const next = new Set(selection);
    patch.added?.forEach((a) => next.delete(a.id));
    set({
      items,
      version: get().version + 1,
      undoStack: undoStack.slice(0, -1),
      redoStack: [...redoStack, patch],
      selection: next,
        });
  },

  redo: () => {
    const { undoStack, redoStack, items } = get();
    if (redoStack.length === 0) return;
    const patch = redoStack[redoStack.length - 1];
    applyPatchTo(items, patch);
    set({
      items,
      version: get().version + 1,
      undoStack: [...undoStack, patch],
      redoStack: redoStack.slice(0, -1),
        });
  },

  canUndo: () => get().undoStack.length > 0,
  canRedo: () => get().redoStack.length > 0,

  setTool: (tool) => set({ tool }),
  setActiveClass: (activeClassId) => set({ activeClassId }),
  // Clamp defensively: these are fed by sliders and by zoom-derived maths, and
  // a NaN reaching the store would poison both the UI and the brush geometry.
  setBrushRadius: (r) =>
    set({ brushRadius: Number.isFinite(r) ? Math.min(20000, Math.max(4, r)) : 60 }),
  setFillOpacity: (o) =>
    set({ fillOpacity: Number.isFinite(o) ? Math.min(1, Math.max(0, o)) : 0.28 }),
  toggleAnnotations: () => set({ showAnnotations: !get().showAnnotations }),
  toggleClassVisibility: (id) => {
    const hidden = new Set(get().hiddenClasses);
    if (hidden.has(id)) hidden.delete(id);
    else hidden.add(id);
    set({ hiddenClasses: hidden });
  },
  setClassHidden: (id, hide) => {
    const hidden = new Set(get().hiddenClasses);
    if (hide) hidden.add(id);
    else hidden.delete(id);
    set({ hiddenClasses: hidden });
  },
  /**
   * `activate` is what separates "I am creating this class to draw in it" from
   * "something needed this class to exist". A detector calling `ensureClass`
   * must not take over the brush: the next thing drawn would silently land in
   * the detector's class, which for tissue means a nucleus drawn in the tissue
   * colour, on top of the tissue region, invisible.
   */
  createClass: (name, color, opts) => {
    const { classes } = get();
    const trimmed = name.trim() || "Class";
    const base = trimmed.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "class";
    let id = base;
    for (let n = 2; classes.some((c) => c.id === id); n++) id = `${base}-${n}`;
    const cls: AnnotationClass = { id, name: trimmed, color: color ?? nextClassColor(classes) };
    set({
      classes: [...classes, cls],
      ...(opts?.activate === false ? {} : { activeClassId: cls.id }),
    });
    return cls;
  },

  ensureClass: (name) => {
    const key = name.trim().toLowerCase();
    const found = get().classes.find((c) => c.name.toLowerCase() === key);
    if (found) return found;
    return get().createClass(name, undefined, { activate: false });
  },

  updateClass: (id, patch) =>
    set({ classes: get().classes.map((c) => (c.id === id ? { ...c, ...patch } : c)) }),

  deleteClass: (id) => {
    const { classes, items, activeClassId } = get();
    // Objects carrying the class must be updated in the same undoable step,
    // otherwise undoing the delete would leave them orphaned.
    const affected = [...items.values()].filter((a) => a.classId === id);
    if (affected.length > 0) {
      get().apply({
        label: `Delete class`,
        updated: affected.map((a) => ({ before: a, after: { ...a, classId: null } })),
      });
    }
    set({
      classes: classes.filter((c) => c.id !== id),
      activeClassId: activeClassId === id ? null : activeClassId,
    });
  },

  reorderClass: (id, delta) => {
    const classes = [...get().classes];
    const i = classes.findIndex((c) => c.id === id);
    const j = i + delta;
    if (i === -1 || j < 0 || j >= classes.length) return;
    [classes[i], classes[j]] = [classes[j], classes[i]];
    set({ classes });
  },

  setClasses: (classes) => set({ classes }),

  assignClass: (ids, classId) => {
    const { items } = get();
    const updated = ids
      .map((id) => items.get(id))
      .filter((a): a is Annotation => !!a && a.classId !== classId)
      .map((a) => ({ before: a, after: { ...a, classId, updatedAt: Date.now() } }));
    if (updated.length === 0) return;
    get().apply({ label: `Set class (${updated.length})`, updated });
  },

  renameAnnotation: (id, name) => {
    const a = get().items.get(id);
    if (!a) return;
    const next = name.trim();
    get().apply({
      label: "Rename",
      updated: [{ before: a, after: { ...a, name: next || undefined, updatedAt: Date.now() } }],
    });
  },

  previewGeometries: (updates) => {
    const { items } = get();
    for (const u of updates) {
      const a = items.get(u.id);
      if (a) items.set(u.id, { ...a, geometry: u.geometry, bbox: bboxOf(u.geometry) });
    }
    set({ version: get().version + 1 });
  },

  setSelection: (ids) => set({ selection: new Set(ids) }),

  setNotice: (notice) => set({ notice }),

  openContextMenu: (contextMenu) => set({ contextMenu, ...(contextMenu ? { classPicker: null } : {}) }),
  openClassPicker: (classPicker) => set({ classPicker, ...(classPicker ? { contextMenu: null } : {}) }),

  select: (ids, additive = false) => {
    const next = additive ? new Set(get().selection) : new Set<string>();
    ids.forEach((id) => (next.has(id) ? next.delete(id) : next.add(id)));
    set({ selection: next });
  },
  clearSelection: () => set({ selection: new Set() }),

  resetFor: (slideKey) =>
    set({
      items: new Map(),
      version: get().version + 1,
      selection: new Set(),
      undoStack: [],
      redoStack: [],
      slideKey,
        }),

  loadDocument: (slideKey, list, classes) => {
    const items = new Map<string, Annotation>();
    list.forEach((a) => items.set(a.id, a));
    set({
      items,
      version: get().version + 1,
      selection: new Set(),
      undoStack: [],
      redoStack: [],
      slideKey,
          ...(classes && classes.length ? { classes } : {}),
    });
  },
}));

// ------------------------------------------------------------ spatial index ---

/**
 * Flatbush is static, so it is rebuilt whenever the document changes. Build is
 * ~10 ms at 10^5 boxes, and it turns hit-testing and viewport culling from a
 * linear scan into a handful of comparisons.
 *
 * The cache lives here rather than in the store: writing it back through
 * setState would notify every subscriber mid-render and cost a redundant frame
 * on every edit.
 */
let indexCache: { version: number; tree: Flatbush; ids: string[] } | null = null;

export function getIndex(): { tree: Flatbush; ids: string[] } | null {
  const { items, version } = useAnnotations.getState();
  if (indexCache && indexCache.version === version) return indexCache;
  if (items.size === 0) return null;

  const ids: string[] = [];
  const tree = new Flatbush(items.size);
  for (const a of items.values()) {
    tree.add(a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3]);
    ids.push(a.id);
  }
  tree.finish();
  indexCache = { version, tree, ids };
  return indexCache;
}

export function queryBox(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): Annotation[] {
  const index = getIndex();
  const { items } = useAnnotations.getState();
  if (!index) return [];
  return index.tree
    .search(minX, minY, maxX, maxY)
    .map((i) => items.get(index.ids[i]))
    .filter((a): a is Annotation => a !== undefined);
}

// ------------------------------------------------------------------ helpers ---

let counter = 0;
export function newId(): string {
  counter += 1;
  return `${Date.now().toString(36)}-${counter.toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

export function makeAnnotation(
  geometry: Geometry,
  partial: Partial<Annotation> = {},
): Annotation {
  const now = Date.now();
  // geometry and bbox are set last so a stale bbox in `partial` can never win.
  return {
    ...partial,
    id: partial.id ?? newId(),
    classId: partial.classId ?? null,
    objectType: partial.objectType ?? "annotation",
    source: partial.source ?? "human",
    locked: partial.locked ?? false,
    createdAt: partial.createdAt ?? now,
    updatedAt: now,
    geometry,
    bbox: bboxOf(geometry),
  };
}

export function withGeometry(a: Annotation, geometry: Geometry): Annotation {
  return { ...a, geometry, bbox: bboxOf(geometry), updatedAt: Date.now() };
}
