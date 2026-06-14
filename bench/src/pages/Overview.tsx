import { useEffect, useState } from "react";
import { Alert, Spinner, Table } from "react-bootstrap";
import { getSummary } from "../api";
import type { ReportRow } from "../types";

const f = (n: number | null, p = 2) => (n == null ? "—" : n.toFixed(p));

export default function Overview() {
  const [rows, setRows] = useState<ReportRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSummary()
      .then(setRows)
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner animation="border" />;
  if (err) return <Alert variant="danger">{err}</Alert>;

  return (
    <>
      <h3 className="mb-3">Latest results per cell</h3>
      <p className="text-muted">
        Median of each cell's samples on the most recent build seen for that
        (host, model, scenario).
      </p>
      <Table striped bordered hover responsive size="sm">
        <thead>
          <tr>
            <th>GPU</th>
            <th>model</th>
            <th className="text-end">prompt tok</th>
            <th className="text-end">TTFT (s)</th>
            <th className="text-end">decode tok/s</th>
            <th className="text-end">total (s)</th>
            <th>build</th>
            <th className="text-end">n</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.gpu ?? r.target_name}</td>
              <td>{r.model_id}</td>
              <td className="text-end">
                {r.prompt_tokens ?? `~${r.prompt_size_approx}`}
              </td>
              <td className="text-end">{f(r.ttft_s_median, 3)}</td>
              <td className="text-end">{f(r.decode_tps_median, 1)}</td>
              <td className="text-end">{f(r.total_s_median, 3)}</td>
              <td>
                <code>{r.git_sha}</code>
              </td>
              <td className="text-end">{r.samples}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </>
  );
}
