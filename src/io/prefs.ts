/**
 * Preferences that outlive a session but belong to this browser, not a slide.
 */

const REMEMBER = "slidecraft.rememberAnnotations";

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
