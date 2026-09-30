// @vitest-environment jsdom
import "../../test/i18n";
import { useMemo, useState } from "react";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ComposePane from "./ComposePane";
import { buildRequest, type Draft } from "../../lib/decisionRequest";
import { encodeShare } from "../../lib/decisionShare";
import { DRAFT_STORAGE_KEY, usePlaygroundDraft } from "../../lib/usePlaygroundDraft";

const start: Draft = {
  stateText: "My card was charged twice.",
  stateMode: "text",
  questionsText: '{\n  "refund": {"type": "noul", "instructions": "Refund?"}\n}',
  checkpoint: "auto",
};

/** The pane wired the way the page wires it: issues rebuilt on every edit. */
function Harness({ onRun, busy = false }: { onRun: () => void; busy?: boolean }) {
  const [draft, setDraft] = useState(start);
  const built = useMemo(() => buildRequest(draft), [draft]);
  return (
    <>
      <ComposePane
        draft={draft}
        onChange={(p) => setDraft((d) => ({ ...d, ...p }))}
        issues={built.ok ? [] : built.issues}
        busy={busy}
        onRun={onRun}
        onCancel={() => undefined}
      />
      <output data-testid="checkpoint">{draft.checkpoint}</output>
    </>
  );
}

const questions = (): HTMLTextAreaElement => screen.getByLabelText("Questions") as HTMLTextAreaElement;
const runButton = (): HTMLButtonElement => screen.getByRole("button", { name: /Run/ }) as HTMLButtonElement;

afterEach(() => {
  cleanup();
  localStorage.clear();
  window.location.hash = "";
});

describe("ComposePane", () => {
  it("inserts each primitive's template after the existing questions", () => {
    render(<Harness onRun={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: /Multiple choice/ }));
    fireEvent.click(screen.getByRole("button", { name: /Rubric/ }));
    fireEvent.click(screen.getByRole("button", { name: /Yes or no/ }));
    const ids = [...questions().value.matchAll(/^ {2}"([^"]+)": \{/gm)].map((m) => m[1]);
    expect(ids).toEqual(["refund", "choice", "score", "noul"]);
    expect(runButton().disabled).toBe(false);
  });

  it("shows validation problems with their position and disables Run", () => {
    render(<Harness onRun={() => undefined} />);
    fireEvent.change(questions(), { target: { value: '{\n  "q": {"type": "noul",\n}' } });
    expect(screen.getByRole("alert").textContent).toMatch(/line 3, column 1/);
    expect(runButton().disabled).toBe(true);
    // Adding a template would clobber the broken text, so it is refused.
    expect((screen.getByRole("button", { name: /Rubric/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("names the question a problem is about", () => {
    render(<Harness onRun={() => undefined} />);
    fireEvent.change(questions(), { target: { value: '{"urgency": {"type": "score", "instructions": "x"}}' } });
    expect(screen.getByRole("alert").textContent).toContain("“urgency”: a score needs criteria");
  });

  it("runs on Ctrl+Enter and Cmd+Enter, but not while invalid", () => {
    const onRun = vi.fn();
    render(<Harness onRun={onRun} />);
    fireEvent.keyDown(questions(), { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(screen.getByLabelText("State"), { key: "Enter", metaKey: true });
    expect(onRun).toHaveBeenCalledTimes(2);
    fireEvent.keyDown(questions(), { key: "Enter" });
    expect(onRun).toHaveBeenCalledTimes(2);
    fireEvent.change(questions(), { target: { value: "{" } });
    fireEvent.keyDown(questions(), { key: "Enter", ctrlKey: true });
    expect(onRun).toHaveBeenCalledTimes(2);
  });

  it("offers Cancel instead of Run while a request is in flight", () => {
    render(<Harness onRun={() => undefined} busy />);
    expect(screen.queryByRole("button", { name: /Run/ })).toBeNull();
    expect(screen.getByRole("button", { name: /Cancel/ })).toBeTruthy();
  });

  it("maps the checkpoint picker onto the draft", () => {
    render(<Harness onRun={() => undefined} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "multilingual" } });
    expect(screen.getByTestId("checkpoint").textContent).toBe("multilingual");
  });

  it("explains what each state format means", () => {
    render(<Harness onRun={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "JSON" }));
    expect(document.body.textContent).toMatch(/read as a conversation/);
  });
});

describe("usePlaygroundDraft", () => {
  it("brings back the last draft after a reload", () => {
    const { result, unmount } = renderHook(() => usePlaygroundDraft(start));
    act(() => result.current.setDraft({ ...start, stateText: "edited" }));
    unmount();
    expect(JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY)!).stateText).toBe("edited");
    const again = renderHook(() => usePlaygroundDraft(start));
    expect(again.result.current.draft.stateText).toBe("edited");
  });

  it("prefers a share link over the stored draft", async () => {
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify({ ...start, stateText: "stored" }));
    window.location.hash = await encodeShare({ ...start, stateText: "shared", checkpoint: "multilingual" });
    const { result } = renderHook(() => usePlaygroundDraft(start));
    await waitFor(() => expect(result.current.fromShare).toBe(true));
    expect(result.current.draft.stateText).toBe("shared");
    expect(result.current.draft.checkpoint).toBe("multilingual");
  });

  it("falls back to the default when storage holds junk", () => {
    localStorage.setItem(DRAFT_STORAGE_KEY, "{not json");
    const { result } = renderHook(() => usePlaygroundDraft(start));
    expect(result.current.draft).toEqual(start);
  });
});
