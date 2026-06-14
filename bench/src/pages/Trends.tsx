import { useEffect, useMemo, useState } from "react";
import { Alert, Col, Form, Row, Spinner } from "react-bootstrap";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDimensions, getSeries } from "../api";
import type { Dimensions, SeriesPoint } from "../types";

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
  const [host, setHost] = useState("");
  const [model, setModel] = useState("");
  const [scenario, setScenario] = useState("");
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getDimensions()
      .then((d) => {
        setDims(d);
        if (d.hosts[0]) setHost(d.hosts[0]);
        if (d.models[0]) setModel(d.models[0]);
        if (d.scenarios[0]) setScenario(d.scenarios[0]);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (host && model && scenario) {
      getSeries(host, model, scenario)
        .then(setSeries)
        .catch((e) => setErr(String(e)));
    }
  }, [host, model, scenario]);

  const data = useMemo(
    () =>
      series.map((p) => ({
        label: p.git_sha,
        ttft: p.ttft_s_median,
        decode: p.decode_tps_median,
        total: p.total_s_median,
      })),
    [series],
  );

  if (err) return <Alert variant="danger">{err}</Alert>;
  if (!dims) return <Spinner animation="border" />;

  return (
    <>
      <h3 className="mb-3">Trends over builds</h3>
      <Row className="g-3 mb-4">
        <Picker label="Host" value={host} set={setHost} options={dims.hosts} />
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

      {data.length === 0 ? (
        <Alert variant="info">No data for this selection yet.</Alert>
      ) : (
        <>
          <h5 className="mt-3">decode tok/s (higher is better)</h5>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="decode"
                name="decode tok/s"
                stroke="#0d6efd"
                connectNulls
              />
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
              <Line
                type="monotone"
                dataKey="ttft"
                name="TTFT (s)"
                stroke="#dc3545"
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </>
  );
}
