// An order-preserving JSON reader.
//
// `JSON.parse` returns plain objects, and JavaScript objects enumerate
// integer-like keys ("0", "10", "2") before every other key, in numeric
// order — so `{"b": 1, "10": 2, "2": 3}` comes back as 2, 10, b. For a
// decision request that is not cosmetic: the server tokenizes question
// ids, options and object states in the order they were written, so a
// reordering changes the token ids and can change the answer. This reader
// keeps objects as `Map`s, which remember insertion order for every key.
//
// A repeated key keeps its first position and takes the last value, as a
// Python dict (the reference server) does. Numbers keep their source text
// so nothing is re-rounded on the way through.

/** A JSON number with its original lexeme. */
export class JsonNumber {
  readonly raw: string;
  constructor(raw: string) {
    this.raw = raw;
  }
  get value(): number {
    return Number(this.raw);
  }
}

export type OJ = null | boolean | string | JsonNumber | OJ[] | OMap;
export type OMap = Map<string, OJ>;

export function isMap(v: OJ | undefined): v is OMap {
  return v instanceof Map;
}

/** A syntax error with a 1-based line and column. */
export class JsonSyntaxError extends Error {
  line: number;
  column: number;
  constructor(message: string, line: number, column: number) {
    super(message);
    this.name = "JsonSyntaxError";
    this.line = line;
    this.column = column;
  }
}

/** Nesting limit, so a hostile document cannot exhaust the stack. */
const MAX_DEPTH = 256;

export function parseOrdered(text: string): OJ {
  let i = 0;

  function fail(msg: string, at = i): never {
    let line = 1;
    let col = 1;
    for (let k = 0; k < at && k < text.length; k++) {
      if (text[k] === "\n") {
        line++;
        col = 1;
      } else col++;
    }
    throw new JsonSyntaxError(msg, line, col);
  }
  function ws(): void {
    while (i < text.length && " \t\n\r".includes(text[i])) i++;
  }
  function str(): string {
    const start = i;
    i++; // opening quote
    let out = "";
    while (i < text.length) {
      const c = text[i];
      if (c === '"') {
        i++;
        return out;
      }
      if (c === "\\") {
        const e = text[i + 1];
        const simple: Record<string, string> = {
          '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t",
        };
        if (e in simple) {
          out += simple[e];
          i += 2;
        } else if (e === "u") {
          const hex = text.slice(i + 2, i + 6);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("invalid \\u escape");
          out += String.fromCharCode(parseInt(hex, 16));
          i += 6;
        } else fail("invalid escape");
      } else if (c < " ") {
        fail("control character in string");
      } else {
        out += c;
        i++;
      }
    }
    fail("unterminated string", start);
  }
  function value(depth: number): OJ {
    if (depth > MAX_DEPTH) fail("nested too deeply");
    ws();
    const c = text[i];
    if (c === "{") {
      i++;
      const m: OMap = new Map();
      ws();
      if (text[i] === "}") {
        i++;
        return m;
      }
      for (;;) {
        ws();
        if (text[i] !== '"') fail("expected a string key");
        const k = str();
        ws();
        if (text[i] !== ":") fail("expected ':'");
        i++;
        m.set(k, value(depth + 1));
        ws();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "}") {
          i++;
          return m;
        }
        fail("expected ',' or '}'");
      }
    }
    if (c === "[") {
      i++;
      const a: OJ[] = [];
      ws();
      if (text[i] === "]") {
        i++;
        return a;
      }
      for (;;) {
        a.push(value(depth + 1));
        ws();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "]") {
          i++;
          return a;
        }
        fail("expected ',' or ']'");
      }
    }
    if (c === '"') return str();
    if (text.startsWith("true", i)) {
      i += 4;
      return true;
    }
    if (text.startsWith("false", i)) {
      i += 5;
      return false;
    }
    if (text.startsWith("null", i)) {
      i += 4;
      return null;
    }
    const num = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(i));
    if (num) {
      i += num[0].length;
      return new JsonNumber(num[0]);
    }
    if (i >= text.length) fail("unexpected end of input");
    fail(`unexpected character '${c}'`);
  }

  const v = value(0);
  ws();
  if (i < text.length) fail("unexpected text after the value");
  return v;
}

/** Serialise, preserving order, as `JSON.stringify` would lay it out. */
export function stringifyOrdered(v: OJ, indent = 2, level = 0): string {
  const pad = (n: number): string => (indent ? "\n" + " ".repeat(indent * n) : "");
  const sep = indent ? ": " : ":";
  if (v === null) return "null";
  if (typeof v === "boolean") return String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (v instanceof JsonNumber) return v.raw;
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    return `[${v.map((x) => pad(level + 1) + stringifyOrdered(x, indent, level + 1)).join(",")}${pad(level)}]`;
  }
  if (!v.size) return "{}";
  const parts = [...v].map(
    ([k, x]) => pad(level + 1) + JSON.stringify(k) + sep + stringifyOrdered(x, indent, level + 1),
  );
  return `{${parts.join(",")}${pad(level)}}`;
}

/** A plain JS value (order of integer-like keys NOT kept). */
export function toPlain(v: OJ): unknown {
  if (v instanceof JsonNumber) return v.value;
  if (Array.isArray(v)) return v.map(toPlain);
  if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [k, toPlain(x)]));
  return v;
}

/** An ordered value from a plain one (for building from code). */
export function fromPlain(v: unknown): OJ {
  if (v === null || v === undefined) return null;
  if (typeof v === "boolean" || typeof v === "string") return v;
  if (typeof v === "number") return new JsonNumber(JSON.stringify(v));
  if (Array.isArray(v)) return v.map(fromPlain);
  return new Map(Object.entries(v as Record<string, unknown>).map(([k, x]) => [k, fromPlain(x)]));
}
