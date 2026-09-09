/**
 * DEV ONLY. Loads slides straight off disk through Vite's /@fs/ route so the
 * exact production path (resolveSlides -> openSlide -> tile source) can be
 * exercised without a human at the file picker.
 */
import type { ResolvedFile } from "../slide/types";

declare global {
  interface Window {
    __slidecraftIngest?: (files: ResolvedFile[]) => void;
    __slidecraftLoad?: (absPaths: string[], rootPrefix?: string) => Promise<string>;
    __slidecraftPump?: () => Promise<string>;
  }
}

async function fetchAsFile(absPath: string, virtualPath: string): Promise<ResolvedFile> {
  const res = await fetch(`/@fs${absPath}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${absPath}`);
  const blob = await res.blob();
  const name = virtualPath.slice(virtualPath.lastIndexOf("/") + 1);
  return { path: virtualPath, file: new File([blob], name) };
}

export function installTestHarness() {
  /**
   * Automated browsers often report `document.hidden`, and the spec suspends
   * requestAnimationFrame while hidden — which stalls OpenSeadragon's render
   * loop and makes a perfectly working viewer look broken. Call this from a
   * test driver to drive the loop from a timer instead. Never used in the app.
   */
  window.__slidecraftPump = async () => {
    const OSD = (await import("openseadragon")).default as unknown as {
      requestAnimationFrame: (cb: FrameRequestCallback) => unknown;
    };
    OSD.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 16);
    return "rAF driven by timer";
  };

  window.__slidecraftLoad = async (absPaths, rootPrefix = "") => {
    const started = performance.now();
    const files = await Promise.all(
      absPaths.map((abs) => {
        const virtual =
          rootPrefix && abs.startsWith(rootPrefix)
            ? abs.slice(rootPrefix.length).replace(/^\//, "")
            : abs.slice(abs.lastIndexOf("/") + 1);
        return fetchAsFile(abs, virtual);
      }),
    );
    const bytes = files.reduce((n, f) => n + f.file.size, 0);
    if (!window.__slidecraftIngest) throw new Error("App not mounted");
    window.__slidecraftIngest(files);
    return `fetched ${files.length} file(s), ${(bytes / 1e6).toFixed(0)} MB in ${(performance.now() - started).toFixed(0)} ms`;
  };
}
