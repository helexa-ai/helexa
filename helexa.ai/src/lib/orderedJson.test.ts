import { describe, expect, it } from "vitest";
import {
  JsonNumber,
  JsonSyntaxError,
  fromPlain,
  parseOrdered,
  stringifyOrdered,
  toPlain,
} from "./orderedJson";

describe("parseOrdered", () => {
  it("keeps integer-like keys in the order they were written", () => {
    const text = '{"b": 1, "10": 2, "2": 3}';
    // The defect this module exists for: plain objects hoist integer keys.
    expect(Object.keys(JSON.parse(text))).toEqual(["2", "10", "b"]);
    expect(stringifyOrdered(parseOrdered(text), 0)).toBe('{"b":1,"10":2,"2":3}');
  });

  it("keeps a repeated key's first position and its last value, like a Python dict", () => {
    expect(stringifyOrdered(parseOrdered('{"a": 1, "b": 2, "a": 3}'), 0)).toBe('{"a":3,"b":2}');
  });

  it("keeps number lexemes, so nothing is re-rounded", () => {
    const v = parseOrdered("[1.7601518630981445, 1e3, -0, 12345678901234567890]");
    expect(stringifyOrdered(v, 0)).toBe("[1.7601518630981445,1e3,-0,12345678901234567890]");
    expect((v as JsonNumber[])[1].value).toBe(1000);
  });

  it("decodes escapes and non-ASCII text", () => {
    expect(parseOrdered('"a\\n\\u00e9\\"x\\\\ 日本"')).toBe('a\né"x\\ 日本');
  });

  it("reports syntax errors with a line and column", () => {
    try {
      parseOrdered('{\n  "a": 1,\n  "b" 2\n}');
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(JsonSyntaxError);
      expect((e as JsonSyntaxError).line).toBe(3);
      expect((e as JsonSyntaxError).column).toBe(7);
    }
  });

  it.each([
    ["", "unexpected end of input"],
    ["{", "expected a string key"],
    ['{"a":1,}', "expected a string key"],
    ["[1 2]", "expected ',' or ']'"],
    ['"unterminated', "unterminated string"],
    ["{} extra", "unexpected text after the value"],
    ["tru", "unexpected character"],
  ])("rejects %j", (text, msg) => {
    expect(() => parseOrdered(text)).toThrow(msg);
  });

  it("refuses absurd nesting instead of overflowing the stack", () => {
    expect(() => parseOrdered("[".repeat(10_000))).toThrow("nested too deeply");
  });

  it("round-trips through plain values", () => {
    const plain = { a: [1, true, null, "x"], b: { c: 2.5 } };
    expect(toPlain(fromPlain(plain))).toEqual(plain);
    expect(stringifyOrdered(fromPlain(plain), 2)).toBe(JSON.stringify(plain, null, 2));
  });
});
