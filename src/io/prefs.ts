/**
 * Preferences that outlive a session but belong to this browser, not a slide.
 */

const REMEMBER = "slidecraft.rememberAnnotations";
const ASSOCIATED = "slidecraft.loadAssociated";

/**
 * Whether opening a slide restores what was saved against it.
 *
 * On by default, because losing an afternoon's annotations to a page reload is
 * the worse failure. Off is for going back to a slide you have already worked
 * and wanting to see it clean — a second pass, or a demo.
 *
 * Turning it off does not delete anything. The saved document stays where it
 * is; it is simply not loaded. It is replaced only if you then draw something,
 * because that is when autosave next runs.
 */
export function getRememberAnnotations(): boolean {
  try {
    return localStorage.getItem(REMEMBER) !== "0";
  } catch {
    return true;
  }
}

export function setRememberAnnotations(on: boolean): void {
  try {
    localStorage.setItem(REMEMBER, on ? "1" : "0");
  } catch {
    /* storage blocked; the session still works, the choice just will not stick */
  }
}

/**
 * Whether files sitting beside a slide load with it.
 *
 * A batch run writes `<slide>.geojson` next to each slide and a GPU run writes
 * `<slide>.expression.bin`, so dropping the folder back in brings the work with
 * it rather than leaving it to be imported by hand, one slide at a time.
 *
 * On by default: finding the files is the whole reason they are named after the
 * slide. Off is for opening a slide deliberately clean when its neighbours on
 * disk are stale — an old segmentation you do not want coloured over the new
 * one, or a map from a model you have since replaced.
 *
 * It never overwrites: anything already saved for the slide wins, and a sidecar
 * computed on a different slide is refused rather than drawn in the wrong
 * place.
 */
export function getLoadAssociated(): boolean {
  try {
    return localStorage.getItem(ASSOCIATED) !== "0";
  } catch {
    return true;
  }
}

export function setLoadAssociated(on: boolean): void {
  try {
    localStorage.setItem(ASSOCIATED, on ? "1" : "0");
  } catch {
    /* storage blocked; the session still works, the choice just will not stick */
  }
}
