import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { pyLiteral, snippet } from "./decisionSnippets";
import { parseOrdered } from "./orderedJson";

const body =
  '{"model":"helexa/one","state":"it\'s broken","questions":{"10":{"type":"noul","instructions":"x","labels":null},"2":{"type":"choice","instructions":"y","criteria":[true,1.5]}}}';

describe("snippets", () => {
  it("curl posts the exact body, quoted for the shell", () => {
    const s = snippet("curl", body, "https://helexa.ai/");
    expect(s).toContain("curl https://helexa.ai/v1/systemone");
    const quoted = s.slice(s.indexOf("-d ") + 3);
    // Let a real shell undo the quoting: the payload must survive it intact.
    const unquoted = execFileSync("sh", ["-c", `printf %s ${quoted}`], { encoding: "utf8" });
    expect(JSON.parse(unquoted)).toEqual(JSON.parse(body));
    expect(unquoted.indexOf('"10"')).toBeLessThan(unquoted.indexOf('"2"'));
    expect(s).toContain('"Authorization: Bearer $HELEXA_API_KEY"');
  });

  it("python uses urllib and Python literals, in order", () => {
    const s = snippet("python", body, "https://helexa.ai");
    expect(s).toContain("import urllib.request");
    expect(s).toContain('"https://helexa.ai/v1/systemone"');
    expect(s).toContain("True");
    expect(s).toContain('"labels": None');
    expect(s.indexOf('"10"')).toBeLessThan(s.indexOf('"2"'));
    expect(s).not.toContain("true,");
  });

  it("the laya snippet calls the SDK's remote client with the model", () => {
    const s = snippet("laya", body, "https://helexa.ai");
    expect(s).toContain("from laya.integrations.llamaindex import _call_remote");
    expect(s).toContain('"https://helexa.ai",');
    expect(s).toContain('model="helexa/one"');
    expect(s).toContain('state = "it\'s broken"');
  });

  it("reflects edits to the request", () => {
    const edited = body.replace("helexa/one", "multilingual");
    for (const lang of ["curl", "python", "laya"] as const) {
      expect(snippet(lang, edited, "https://h")).toContain("multilingual");
      expect(snippet(lang, body, "https://h")).not.toContain("multilingual");
    }
  });

  it("writes Python literals for every JSON type", () => {
    expect(pyLiteral(parseOrdered('{"a":[null,false,1e3,"x"],"b":{}}'))).toBe(
      '{\n    "a": [\n        None,\n        False,\n        1e3,\n        "x",\n    ],\n    "b": {},\n}',
    );
  });
});
