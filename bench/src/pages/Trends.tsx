import { useEffect, useMemo, useState } from "react";
import { Alert, Col, Form, Row, Spinner } from "react-bootstrap";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDimensions, getRegimes, getSeries } from "../api";
import type { Dimensions, MeasurementRegime, SeriesPoint } from "../types";
import { BASELINE_SOURCE, baselineFor } from "../baseline";

function Picker({
  label,
  value,
  set,
  options,
}: {
  label: string;
  value: string;
  set: (v: string) => void;
  options: string[];
}) {
  return (
    <Form.Group as={Col}>
      <Form.Label>{label}</Form.Label>
      <Form.Select value={value} onChange={(e) => set(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </Form.Select>
    </Form.Group>
  );
}

type SeriesDef = {
  key: string;
  name: string;
  stroke: string;
  dashed?: boolean;
};

/** One titled chart over the shared build timeline.
 *
 * Every panel draws the same x-axis and the same regime divider, so a
 * reader can line a change up across metrics — which is the whole point
 * of having more than two of them. */
/** A vertical rule at a build where the metric changed meaning. */
type Rule = { at: string; label: string; detail?: string };

function MetricChart({
  title,
  hint,
  data,
  lines,
  rules,
  unit,
}: {
  title: string;
  hint?: string;
  data: Record<string, unknown>[];
  lines: SeriesDef[];
  rules?: Rule[];
  unit?: string;
}) {
  const hasAny = data.some((d) => lines.some((l) => d[l.key] != null));
  if (!hasAny) return null;
  return (
    <>
      <h5 className="mt-4">
        {title}
        {unit ? <span className="text-muted fw-normal"> ({unit})</span> : null}
      </h5>
      {hint && <p className="text-muted small mb-2">{hint}</p>}
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" />
          <YAxis />
          <Tooltip />
          <Legend />
          {(rules ?? [])
            // Only draw a rule whose build is actually on this x-axis;
            // recharts would otherwise place it at the origin and imply
            // the boundary sits before all the data.
            .filter((r) => data.some((d) => d.label === r.at))
            .map((r) => (
              <ReferenceLine
                key={`${r.at}-${r.label}`}
                x={r.at}
                stroke="#bbb"
                strokeDasharray="3 3"
                label={{
                  value: r.label,
                  position: "top",
                  fill: "#999",
                  fontSize: 11,
                }}
              />
            ))}
          {lines.map((l) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.name}
              stroke={l.stroke}
              strokeDasharray={l.dashed ? "5 5" : undefined}
              connectNulls
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

export default function Trends() {
  const [dims, setDims] = useState<Dimensions | null>(null);
  const [model, setModel] = useState("");
  const [scenario, setScenario] = useState("");
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [regimes, setRegimes] = useState<MeasurementRegime[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getDimensions()
      .then((d) => {
        setDims(d);
        if (d.models[0]) setModel(d.models[0]);
        if (d.scenarios[0]) setScenario(d.scenarios[0]);
      })
      .catch((e) => setErr(String(e)));
    // A missing/older API just means no rules — not a page failure.
    getRegimes()
      .then(setRegimes)
      .catch(() => setRegimes([]));
  }, []);

  useEffect(() => {
    if (model && scenario) {
      getSeries(model, scenario)
        .then(setSeries)
        .catch((e) => setErr(String(e)));
    }
  }, [model, scenario]);

  // Prepend the pre-helexa-bench baseline (dashed, separate keys) so it
  // anchors the timeline without being merged into the live line. Different
  // measurement regime — see baseline.ts / doc/benchmarks.md.
  const base = useMemo(() => baselineFor(model, scenario), [model, scenario]);
  const data = useMemo(
    () => [
      ...base.map((p) => ({
        label: p.git_sha,
        baseTtft: p.ttft_s,
        baseDecode: p.decode_tps,
        baseTotal: p.total_s,
      })),
      ...series.map((p) => ({
        label: p.git_sha,
        ttft: p.ttft_s_median,
        decode: p.decode_tps_median,
        total: p.total_s_median,
        ttftP95: p.ttft_p95_s_median,
        queueWait: p.queue_wait_ms_median,
        rejected: p.rejected_median,
        prefillTps: p.prefill_tps_median,
        reasoning: p.reasoning_tokens_median,
        cached: p.cached_tokens_median,
        completion: p.completion_tokens_median,
        tpot: p.tpot_p95_ms_median,
      })),
    ],
    [series, base],
  );

  // Divider marking the boundary between the two regimes (drawn at the
  // first live build, with baseline points to its left).
  const firstLive = series[0]?.git_sha;
  const showDivider = base.length > 0 && series.length > 0;

  // The build where the measuring identity changed, derived from the
  // data rather than declared — a constant would go stale the moment
  // the principal is reconfigured (#288).
  const identityShift = useMemo(() => {
    for (let i = 1; i < series.length; i++) {
      const prev = series[i - 1].principal ?? "anonymous";
      const cur = series[i].principal ?? "anonymous";
      if (prev !== cur) {
        return {
          at: series[i].git_sha,
          label: `measured as ${cur === "anonymous" ? "anonymous" : "identified"}`,
          detail:
            "The identity bench measured under changed here. Anonymous " +
            "samples are subject to the #262 yield policy and understate " +
            "capacity, so points either side are not comparable (#288).",
        } as Rule;
      }
    }
    return null;
  }, [series]);

  /** Declared boundaries touching `metric`, plus the regime-independent
   *  ones that apply to every panel. */
  const rulesFor = (metric: string): Rule[] => {
    const out: Rule[] = [];
    if (showDivider && firstLive) {
      out.push({ at: firstLive, label: "bench.py → helexa-bench" });
    }
    for (const r of regimes) {
      if (r.affects.includes(metric)) {
        out.push({ at: r.first_sha, label: r.label, detail: r.detail });
      }
    }
    if (identityShift) out.push(identityShift);
    return out;
  };

  // Anonymous and identified samples are not comparable once #262 is in
  // the build: an anonymous caller is capped below max_in_flight and
  // parked at the class gate, so it characterises the yield policy
  // rather than serving capacity. Say so rather than letting someone
  // read a step change as an engine regression (#288).
  const identities = useMemo(
    () => new Set(series.map((p) => p.principal ?? "anonymous")),
    [series],
  );
  const mixedIdentity = identities.size > 1;
  const anyAnonymous = identities.has("anonymous");

  if (err) return <Alert variant="danger">{err}</Alert>;
  if (!dims) return <Spinner animation="border" />;

  return (
    <>
      <h3 className="mb-3">Trends over builds</h3>
      <Row className="g-3 mb-4">
        <Picker
          label="Model"
          value={model}
          set={setModel}
          options={dims.models}
        />
        <Picker
          label="Scenario"
          value={scenario}
          set={setScenario}
          options={dims.scenarios}
        />
      </Row>

      {dims.model_gpus[model] && (
        <p className="text-muted mb-3">
          Measured on <strong>{dims.model_gpus[model]}</strong>.
        </p>
      )}

      {mixedIdentity && (
        <Alert variant="warning" className="py-2">
          <strong>Mixed measurement identity.</strong> Some builds were
          sampled anonymously and some under a principal. Since{" "}
          <code>#262</code> an anonymous caller is capped below{" "}
          <code>max_in_flight</code> and yields to identified traffic, so
          those points measure the admission policy rather than serving
          capacity. A step change across that boundary is the instrument,
          not the engine — see <code>#288</code>.
        </Alert>
      )}
      {!mixedIdentity && anyAnonymous && (
        <Alert variant="secondary" className="py-2 small">
          Sampled anonymously. Since <code>#262</code> anonymous callers are
          served from leftover capacity, so these numbers understate what an
          authenticated caller gets (<code>#288</code>).
        </Alert>
      )}

      {data.length === 0 ? (
        <Alert variant="info">No data for this selection yet.</Alert>
      ) : (
        <>
          {base.length > 0 && (
            <p className="text-muted small mb-3">
              Dashed = pre-helexa-bench baseline ({BASELINE_SOURCE}); solid =
              helexa-bench (direct to neuron). Different measurement regimes —
              see <code>doc/benchmarks.md</code>.
            </p>
          )}

          <MetricChart
            title="decode tok/s"
            unit="higher is better"
            data={data}
            rules={rulesFor("decode")}
            lines={[
              { key: "decode", name: "decode tok/s", stroke: "#0d6efd" },
              ...(base.length > 0
                ? [
                    {
                      key: "baseDecode",
                      name: "baseline (bench.py · gateway)",
                      stroke: "#888",
                      dashed: true,
                    },
                  ]
                : []),
            ]}
          />

          <MetricChart
            title="prefill tok/s"
            unit="higher is better"
            hint="The other half of serving speed, derived from prefill_tokens / prefill_ms. A prefix-cache hit shortens prefill_ms while the token count stays whole, so a high rate here is itself the cache-hit signal."
            data={data}
            rules={rulesFor("prefillTps")}
            lines={[
              { key: "prefillTps", name: "prefill tok/s", stroke: "#20c997" },
            ]}
          />

          <MetricChart
            title="TTFT"
            unit="seconds, lower is better"
            hint="Median and p95 together on purpose. Under concurrency the median is dominated by whichever streams were admitted immediately; the p95 is the one that moves when a caller is made to wait. Charting only the median is how a 0.53 s → 11.77 s tail went unnoticed for a week."
            data={data}
            rules={rulesFor("ttft")}
            lines={[
              { key: "ttft", name: "TTFT median (s)", stroke: "#dc3545" },
              { key: "ttftP95", name: "TTFT p95 (s)", stroke: "#fd7e14" },
              ...(base.length > 0
                ? [
                    {
                      key: "baseTtft",
                      name: "baseline (bench.py · gateway)",
                      stroke: "#888",
                      dashed: true,
                    },
                  ]
                : []),
            ]}
          />

          <MetricChart
            title="inter-token gap p95"
            unit="ms, lower is better"
            hint="Stream smoothness. decode tok/s is a mean over the whole window, so a stream that stalls and then catches up is indistinguishable from one that never stalled — this is the number a user feels."
            data={data}
            rules={rulesFor("tpot")}
            lines={[
              { key: "tpot", name: "inter-token p95 (ms)", stroke: "#6f42c1" },
            ]}
          />

          <MetricChart
            title="admission"
            unit="queue wait ms · requests shed"
            hint="Separates “the server is slow” from “you were queued behind someone”. Queue wait is TTFT minus server-measured prefill; rejected counts honest backpressure rather than silent failures."
            data={data}
            rules={rulesFor("queueWait")}
            lines={[
              { key: "queueWait", name: "queue wait (ms)", stroke: "#d63384" },
              { key: "rejected", name: "rejected (count)", stroke: "#adb5bd" },
            ]}
          />

          <MetricChart
            title="tokens per sample"
            unit="counts"
            hint="Cost, not speed. Reasoning tokens are the dominant driver on a reasoning model and move independently of every rate above — a template or sampling change can double what the model thinks before answering while the speed charts stay flat. Cached tokens are why prefill timing varies between otherwise identical samples."
            data={data}
            rules={rulesFor("completion")}
            lines={[
              { key: "completion", name: "completion tokens", stroke: "#0dcaf0" },
              { key: "reasoning", name: "reasoning tokens", stroke: "#ffc107" },
              { key: "cached", name: "cached prompt tokens", stroke: "#198754" },
            ]}
          />

          {(() => {
            // Explain every rule actually drawn for this selection. The
            // chart label is a name; without the reason, a reader still
            // has to go and rediscover what it meant — which is the cost
            // this whole feature exists to remove.
            const shown = [
              ...regimes
                .filter((r) => series.some((p) => p.git_sha === r.first_sha))
                .map((r) => ({ at: r.first_sha, label: r.label, detail: r.detail })),
              ...(identityShift ? [identityShift] : []),
            ];
            if (shown.length === 0) return null;
            return (
              <div className="mt-4 pt-3 border-top">
                <p className="text-muted small mb-2">
                  <strong>Dashed rules mark measurement-regime changes</strong> —
                  builds where a number changed meaning rather than value. A step
                  across one is the instrument, not the engine.
                </p>
                <dl className="row small text-muted mb-0">
                  {shown.map((r) => (
                    <div key={`${r.at}-${r.label}`} className="mb-2">
                      <dt className="fw-semibold">
                        <code>{r.at}</code> — {r.label}
                      </dt>
                      <dd className="mb-0">{r.detail}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            );
          })()}
        </>
      )}
    </>
  );
}
