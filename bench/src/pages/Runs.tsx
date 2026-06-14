import { useEffect, useState } from "react";
import { Alert, Badge, Col, Form, Row, Spinner, Table } from "react-bootstrap";
import { getDimensions, getRuns } from "../api";
import type { Dimensions, RunRow } from "../types";

const f = (n: number | null, p = 2) => (n == null ? "—" : n.toFixed(p));

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
        <option value="">(all)</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </Form.Select>
    </Form.Group>
  );
}

export default function Runs() {
  const [dims, setDims] = useState<Dimensions | null>(null);
  const [host, setHost] = useState("");
  const [model, setModel] = useState("");
  const [scenario, setScenario] = useState("");
  const [rows, setRows] = useState<RunRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getDimensions()
      .then(setDims)
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    setLoading(true);
    getRuns({
      host: host || undefined,
      model: model || undefined,
      scenario: scenario || undefined,
      limit: 200,
    })
      .then(setRows)
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [host, model, scenario]);

  if (err) return <Alert variant="danger">{err}</Alert>;

  return (
    <>
      <h3 className="mb-3">Runs</h3>
      {dims && (
        <Row className="g-3 mb-3">
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
      )}
      {loading ? (
        <Spinner animation="border" />
      ) : (
        <Table striped bordered hover responsive size="sm">
          <thead>
            <tr>
              <th>ts</th>
              <th>host</th>
              <th>model</th>
              <th>scenario</th>
              <th>build</th>
              <th className="text-end">TTFT</th>
              <th className="text-end">tok/s</th>
              <th className="text-end">total</th>
              <th>ok</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.ts}</td>
                <td>{r.host}</td>
                <td>{r.model_id}</td>
                <td>{r.scenario_id}</td>
                <td>
                  <code>{r.git_sha}</code>
                </td>
                <td className="text-end">{f(r.ttft_s, 3)}</td>
                <td className="text-end">{f(r.decode_tps, 1)}</td>
                <td className="text-end">{f(r.total_s, 3)}</td>
                <td>
                  {r.ok ? (
                    <Badge bg="success">ok</Badge>
                  ) : (
                    <Badge bg="danger" title={r.error ?? ""}>
                      fail
                    </Badge>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </>
  );
}
