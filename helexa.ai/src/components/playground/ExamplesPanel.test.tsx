// @vitest-environment jsdom
import "../../test/i18n";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ExamplesPanel from "./ExamplesPanel";
import { EXAMPLES } from "../../data/decisionExamples";

afterEach(cleanup);

describe("ExamplesPanel", () => {
  it("lists every example under its group, with lessons tagged by primitive", () => {
    render(<ExamplesPanel onPick={() => undefined} />);
    expect(screen.getAllByRole("button")).toHaveLength(EXAMPLES.length);
    const lessons = screen.getByRole("region", { name: "Lessons" });
    expect([...lessons.querySelectorAll(".hx-pg-type")].map((b) => b.textContent)).toEqual(["noul", "score", "choice"]);
  });

  it("loads the picked example into the editors without running it", () => {
    const onPick = vi.fn();
    render(<ExamplesPanel onPick={onPick} />);
    fireEvent.click(screen.getByRole("button", { name: /Hindi/ }));
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick.mock.calls[0][0].stateText).toMatch(/[ऀ-ॿ]/);
    fireEvent.click(screen.getByRole("button", { name: /Phishing email/ }));
    expect(onPick.mock.calls[1][0].stateMode).toBe("json");
  });
});
