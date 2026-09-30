// @vitest-environment jsdom
import "../../test/i18n";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CodePanel, ShareButton } from "./SharePanel";
import { buildRequest, type Draft } from "../../lib/decisionRequest";
import { decodeShare } from "../../lib/decisionShare";

const draft: Draft = {
  stateText: "Refund me, please.",
  stateMode: "text",
  questionsText: '{"10": {"type": "noul", "instructions": "Refund?"}, "2": {"type": "noul", "instructions": "x"}}',
  checkpoint: "multilingual",
};

let clipboard: string[] = [];
beforeEach(() => {
  clipboard = [];
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn(async (s: string) => void clipboard.push(s)) },
  });
});
afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});

describe("ShareButton", () => {
  it("copies a link that restores the same draft", async () => {
    render(<ShareButton draft={draft} />);
    fireEvent.click(screen.getByRole("button", { name: /Share/ }));
    await waitFor(() => expect(clipboard).toHaveLength(1));
    const url = new URL(clipboard[0]);
    expect(url.hash).toMatch(/^#p=1\./);
    expect(window.location.hash).toBe(url.hash);
    expect(await decodeShare(url.hash)).toMatchObject(draft);
    expect(screen.getByRole("button", { name: /Link copied/ })).toBeTruthy();
  });

  it("refuses a link too long to survive being pasted", async () => {
    // Incompressible text, so deflate can't bring it under the limit.
    const noise = Array.from({ length: 12000 }, (_, i) => String.fromCharCode(0x4e00 + ((i * 7919) % 20000))).join("");
    render(<ShareButton draft={{ ...draft, stateText: noise }} />);
    fireEvent.click(screen.getByRole("button", { name: /Share/ }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(/too long to share/));
    expect(clipboard).toHaveLength(0);
  });
});

describe("CodePanel", () => {
  function bodyOf(d: Draft): string {
    const r = buildRequest(d);
    if (!r.ok) throw new Error("invalid");
    return r.request.body;
  }

  it("shows each language for the current request, against the public endpoint", () => {
    render(
      <MemoryRouter>
        <CodePanel body={bodyOf(draft)} />
      </MemoryRouter>,
    );
    const code = (): string => document.querySelector(".hx-pg-code pre")!.textContent!;
    expect(code()).toContain("curl https://helexa.ai/v1/systemone");
    fireEvent.click(screen.getByRole("tab", { name: "Python" }));
    expect(code()).toContain("urllib.request");
    fireEvent.click(screen.getByRole("tab", { name: "laya SDK" }));
    expect(code()).toContain('model="multilingual"');
    expect(code().indexOf('"10"')).toBeLessThan(code().indexOf('"2"'));
  });

  it("follows edits to the request", () => {
    const { rerender } = render(
      <MemoryRouter>
        <CodePanel body={bodyOf(draft)} />
      </MemoryRouter>,
    );
    rerender(
      <MemoryRouter>
        <CodePanel body={bodyOf({ ...draft, stateText: "A brand new state" })} />
      </MemoryRouter>,
    );
    expect(document.querySelector(".hx-pg-code pre")!.textContent).toContain("A brand new state");
  });

  it("asks for a valid request instead of showing broken code", () => {
    render(
      <MemoryRouter>
        <CodePanel body={null} />
      </MemoryRouter>,
    );
    expect(document.querySelector(".hx-pg-code pre")).toBeNull();
    expect(screen.getByText(/Fix the problems/)).toBeTruthy();
    expect(screen.getByText(/Jev client works unchanged/)).toBeTruthy();
  });
});
