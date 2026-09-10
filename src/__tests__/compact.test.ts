import { describe, expect, it } from "vitest";
import { compact } from "../ui/SidebarTabs";

/**
 * Tab badges sit on a 60-pixel icon, so a patch grid's real count has to be
 * readable at a glance rather than exact — the exact figure is in the tooltip.
 */
describe("compact counts", () => {
  it("leaves small counts alone", () => {
    expect(compact(0)).toBe("0");
    expect(compact(7)).toBe("7");
    expect(compact(999)).toBe("999");
  });

  it("uses one decimal in the thousands, and drops a trailing zero", () => {
    expect(compact(1200)).toBe("1.2k");
    expect(compact(1000)).toBe("1k");
    expect(compact(9949)).toBe("9.9k");
  });

  it("drops the decimal once it would not fit", () => {
    expect(compact(10_400)).toBe("10k");
    expect(compact(23_716)).toBe("24k");
    expect(compact(999_000)).toBe("999k");
  });

  it("carries on into millions", () => {
    expect(compact(1_200_000)).toBe("1.2M");
    expect(compact(12_000_000)).toBe("12M");
  });
});
