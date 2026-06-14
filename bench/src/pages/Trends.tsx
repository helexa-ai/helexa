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
import { getDimensions, getSeries } from "../api";
import type { Dimensions, SeriesPoint } from "../types";
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

export default function Trends() {
  const [dims, setDims] = useState<Dimensions | null>(null);
  const [model, setModel] = useState("");
  const [scenario, setScenario] = useState("");
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getDimensions()
      .then((d) => {
        setDims(d);
        if (d.models[0]) setModel(d.models[0]);
        if (d.scenarios[0]) setScenario(d.scenarios[0]);
      })
      .catch((e) => setErr(String(e)));
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
  const base = useMemo(
    () => baselineFor(model, scenario),
    [model, scenario],
  );
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
      })),
    ],
    [series, base],
  );

  // Divider marking the boundary between the two regimes (drawn at the
  // first live build, with baseline points to its left).
  const firstLive = series[0]?.git_sha;
  const showDivider = base.length > 0 && series.length > 0;

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
          <h5 className="mt-3">decode tok/s (higher is better)</h5>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Legend />
              {showDivider && firstLive && (
                <ReferenceLine
                  x={firstLive}
                  stroke="#bbb"
                  strokeDasharray="3 3"
                  label={{
                    value: "bench.py → helexa-bench",
                    position: "top",
                    fill: "#999",
                    fontSize: 11,
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey="decode"
                name="decode tok/s"
                stroke="#0d6efd"
                connectNulls
              />
              {base.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="baseDecode"
                  name="baseline (bench.py · gateway)"
                  stroke="#888"
                  strokeDasharray="5 5"
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>

          <h5 className="mt-4">TTFT seconds (lower is better)</h5>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Legend />
              {showDivider && firstLive && (
                <ReferenceLine
                  x={firstLive}
                  stroke="#bbb"
                  strokeDasharray="3 3"
                  label={{
                    value: "bench.py → helexa-bench",
                    position: "top",
                    fill: "#999",
                    fontSize: 11,
                  }}
                />
              )}
              <Line
                type="monotone"
                dataKey="ttft"
                name="TTFT (s)"
                stroke="#dc3545"
                connectNulls
              />
              {base.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="baseTtft"
                  name="baseline (bench.py · gateway)"
                  stroke="#888"
                  strokeDasharray="5 5"
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </>
  );
}
