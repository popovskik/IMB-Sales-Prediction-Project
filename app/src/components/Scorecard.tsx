import type { ReactNode } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { LeaderboardRow, ModelDiagnostics, Predictions } from "../types";
import { InfoTip } from "./InfoTip";

const n = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

/** A table header cell with an explanatory bubble for the statistical term.
 *  Opens downward (into the rows) because the table sits in an overflow-x wrapper
 *  that would clip an upward bubble. */
function Th({ children, tip, align = "center" }: { children: ReactNode; tip: ReactNode; align?: "center" | "end" }) {
  return (
    <th>
      <span style={{ display: "inline-flex", alignItems: "center" }}>
        {children}
        <InfoTip placement="bottom" align={align}>{tip}</InfoTip>
      </span>
    </th>
  );
}

export function Scorecard({ leaderboard, models, diagnostics }: {
  leaderboard: LeaderboardRow[];
  models: Predictions["models"];
  diagnostics?: ModelDiagnostics;
}) {
  const reg = leaderboard.filter((r) => r.task === "regression");
  const clf = leaderboard.filter((r) => r.task === "classification");

  // Deployed (best) models: regression by CV R², classification by ROC-AUC.
  // Exclude all report-only comparison rows — only the core ladder is deployable.
  const REPORT_ONLY = new Set(["XGBoost (+lags)", "XGBoost (+holidays)", "SARIMA"]);
  const bestReg = reg
    .filter((r) => !REPORT_ONLY.has(r.model))
    .reduce((a, b) => ((b.cv_r2 ?? -9) > (a.cv_r2 ?? -9) ? b : a), reg[0]);
  const bestClf = clf.reduce((a, b) => ((b.test.roc_auc ?? 0) > (a.test.roc_auc ?? 0) ? b : a), clf[0]);

  return (
    <section className="card">
      <div className="eyebrow">Model scorecard</div>
      <h2 style={{ fontSize: 20, margin: "4px 0 6px" }}>How the models compare</h2>
      <p style={{ color: "var(--ink-2)", fontSize: 13, marginTop: 0 }}>
        Cross-validation (TimeSeriesSplit) is the fairer metric; the Nov–Dec hold-out sits in
        calendar positions unseen in training. Deployed models are picked by CV R² (revenue)
        and ROC-AUC (demand). Baselines are kept for honesty.
      </p>

      <h3 style={{ fontSize: 14, margin: "16px 0 6px" }}>Regression — daily revenue</h3>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <Th tip={<><strong>Cross-validated R².</strong> How much of the day-to-day revenue swing the model explains, averaged over rolling time splits. 1.0 = perfect, 0 = no better than always guessing the average. This is the fairer score.</>}>CV R²</Th>
              <Th tip={<><strong>Test R²</strong> on the held-out Nov–Dec window. Low/negative here because those months are calendar positions the model never trained on — see the note above.</>}>Test R²</Th>
              <Th tip={<><strong>Mean Absolute Error.</strong> On average, the prediction is off by about this many dollars (lower is better).</>}>Test MAE</Th>
              <Th tip={<><strong>Root Mean Squared Error.</strong> Like MAE but punishes big misses more (lower is better), in dollars.</>}>Test RMSE</Th>
              <Th tip={<><strong>Train R².</strong> Fit on the data the model learned from. Much higher than CV/Test = the model is <strong>overfitting</strong> (memorising, not generalising).</>} align="end">Train R²</Th>
            </tr>
          </thead>
          <tbody>
            {reg.map((r) => (
              <tr key={r.model} className={r.model === bestReg.model ? "best" : ""}>
                <td>{r.model}{r.model === bestReg.model ? " · deployed" : ""}</td>
                <td>{n(r.cv_r2)}</td><td>{n(r.test.r2)}</td>
                <td>{n(r.test.mae, 0)}</td><td>{n(r.test.rmse, 0)}</td><td>{n(r.train_r2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={{ fontSize: 14, margin: "20px 0 6px" }}>Classification — high-demand day</h3>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <Th tip={<><strong>Cross-validated F1.</strong> A balance of catching busy days (recall) and not crying wolf (precision), averaged over time splits. Higher is better.</>}>CV F1</Th>
              <Th tip={<><strong>Accuracy.</strong> Share of days the busy/not-busy call was correct on the held-out window. Can mislead when days aren't 50/50.</>}>Test Acc</Th>
              <Th tip={<><strong>F1 score</strong> on the held-out window. Note the "always say busy" baseline scores a deceptively high F1 — which is why we judge by ROC-AUC instead.</>}>Test F1</Th>
              <Th tip={<><strong>ROC-AUC.</strong> The honest headline metric: probability the model ranks a real busy day above a quiet one. 0.50 = coin-flip, 1.0 = perfect. The deployed model reaches ~0.74.</>} align="end">ROC-AUC</Th>
            </tr>
          </thead>
          <tbody>
            {clf.map((r) => (
              <tr key={r.model} className={r.model === bestClf.model ? "best" : ""}>
                <td>{r.model}{r.model === bestClf.model ? " · deployed" : ""}</td>
                <td>{n(r.cv_f1)}</td><td>{n(r.test.accuracy)}</td>
                <td>{n(r.test.f1)}</td><td>{n(r.test.roc_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {diagnostics && (
        <DiagnosticsRow diagnostics={diagnostics} />
      )}

      <p style={{ color: "var(--ink-3)", fontSize: 12, marginTop: 14, marginBottom: 0 }}>
        Trained with scikit-learn {models.sklearn_version ?? "?"}. Features:{" "}
        {models.regression.join(", ")}.
      </p>
    </section>
  );
}

const ORANGE = "#ff4800";
const BLUE = "#2360c4";

function DiagnosticsRow({ diagnostics }: { diagnostics: ModelDiagnostics }) {
  const { regression, classification } = diagnostics;

  // Build time-series data: actual / XGBoost / SARIMA for Nov-Dec
  const pva = regression.predicted_vs_actual;
  const pvaData = pva
    ? pva.dates.map((d, i) => ({
        date: d.slice(5),  // "MM-DD"
        actual: Math.round(pva.actual[i]),
        xgb: Math.round(pva.predicted[i]),
        sarima: pva.sarima_predicted ? Math.round(pva.sarima_predicted[i]) : undefined,
      }))
    : null;

  // Build ROC curve points
  const roc = classification.roc_curve;
  const rocData = roc ? roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr[i] })) : null;

  // Confusion matrix helpers
  const cm = classification.confusion_matrix;

  return (
    <div style={{ marginTop: 24 }}>
      <h3 style={{ fontSize: 14, margin: "0 0 12px" }}>Model visualisations</h3>
      <div className="diagnostics-panels">

        {/* Nov-Dec forecast comparison: Actual vs XGBoost vs SARIMA */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, color: "var(--ink-2)", marginBottom: 6, display: "flex", alignItems: "center" }}>
            Nov–Dec revenue forecast
            <InfoTip placement="bottom">
              <>Daily revenue on the <strong>Nov–Dec hold-out</strong>. Grey = actual,
              orange = XGBoost (calendar features only), blue dashed = SARIMA (time-series model).
              Both models track the weekly rhythm but miss the absolute level — consistent
              with near-zero test R².</>
            </InfoTip>
          </div>
          {pvaData ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={pvaData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
                <XAxis dataKey="date" fontSize={10} interval={13} />
                <YAxis fontSize={11} width={46} tickFormatter={(v: number) => `$${Math.round(v / 1000)}k`} />
                <Tooltip formatter={(v: number, name: string) => [`$${v.toLocaleString()}`, name]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="actual" name="Actual" stroke="#555" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="xgb" name="XGBoost" stroke={ORANGE} strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                <Line type="monotone" dataKey="sarima" name="SARIMA" stroke={BLUE} strokeWidth={1.5} strokeDasharray="2 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <span style={{ color: "var(--ink-3)", fontSize: 13 }}>No data.</span>}
          <p style={{ fontSize: 12, color: "var(--ink-3)", margin: "4px 0 0" }}>
            {regression.best_model ?? ""} vs SARIMA — 61-day hold-out
          </p>
        </div>

        {/* ROC curve */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, color: "var(--ink-2)", marginBottom: 6, display: "flex", alignItems: "center" }}>
            ROC curve — classifier
            <InfoTip placement="bottom">
              <>The <strong>ROC curve</strong> shows the trade-off between catching real
              busy days (true-positive rate) and false alarms (false-positive rate) as the
              decision threshold changes. The grey diagonal is a random coin-flip (AUC = 0.50).
              A curve closer to the top-left corner means a better classifier.</>
            </InfoTip>
          </div>
          {rocData ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={rocData} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} fontSize={11}
                       label={{ value: "False positive rate", position: "insideBottom", offset: -12, fontSize: 11 }} />
                <YAxis dataKey="tpr" type="number" domain={[0, 1]} fontSize={11}
                       label={{ value: "True positive rate", angle: -90, position: "insideLeft", offset: 12, fontSize: 11 }} />
                <Tooltip formatter={(v: number, name: string) =>
                  [v.toFixed(3), name === "tpr" ? "TPR" : "FPR"]} />
                {/* diagonal baseline */}
                <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                      dataKey="tpr" dot={false} stroke="#ccc" strokeDasharray="4 4" strokeWidth={1} />
                <Line dataKey="tpr" dot={false} stroke={BLUE} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : <span style={{ color: "var(--ink-3)", fontSize: 13 }}>No data.</span>}
          <p style={{ fontSize: 12, color: "var(--ink-3)", margin: "4px 0 0" }}>
            {classification.best_model ?? ""} — AUC {classification.roc_auc?.toFixed(3) ?? "—"}
          </p>
        </div>

        {/* Confusion matrix */}
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, color: "var(--ink-2)", marginBottom: 6, display: "flex", alignItems: "center" }}>
            Confusion matrix
            <InfoTip placement="bottom" align="end">
              <>The <strong>confusion matrix</strong> counts predictions vs reality on the
              Nov–Dec hold-out. Rows = actual class, columns = predicted class.
              Cells on the diagonal are correct calls; off-diagonal cells are mistakes.
              False negatives (missed busy days) cost more operationally than false positives.</>
            </InfoTip>
          </div>
          {cm ? <ConfusionMatrix cm={cm} /> : <span style={{ color: "var(--ink-3)", fontSize: 13 }}>No data.</span>}
          <p style={{ fontSize: 12, color: "var(--ink-3)", margin: "4px 0 0" }}>
            {classification.best_model ?? ""} — Nov–Dec hold-out
          </p>
        </div>

      </div>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const labels = ["Normal", "High demand"];
  const max = Math.max(...cm.flat(), 1);
  return (
    <table style={{ borderCollapse: "separate", borderSpacing: 4, margin: "0 auto" }}>
      <thead>
        <tr>
          <th style={{ fontSize: 10, color: "var(--ink-3)", padding: "0 6px 4px", textAlign: "right" }}>
            Actual ↓ / Pred →
          </th>
          {labels.map((l) => (
            <th key={l} style={{ fontSize: 11, color: "var(--ink-2)", padding: "0 4px 4px", textAlign: "center" }}>{l}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {cm.map((row, i) => (
          <tr key={i}>
            <td style={{ fontSize: 11, color: "var(--ink-2)", padding: "0 8px 0 0", textAlign: "right" }}>{labels[i]}</td>
            {row.map((v, j) => {
              const isCorrect = i === j;
              const intensity = v / max;
              return (
                <td key={j} style={{
                  width: 64, height: 52, textAlign: "center", verticalAlign: "middle",
                  borderRadius: 6, fontSize: 18, fontWeight: 700,
                  color: intensity > 0.5 ? "#fff" : "var(--ink)",
                  background: isCorrect
                    ? `rgba(35, 96, 196, ${0.15 + intensity * 0.8})`
                    : `rgba(255, 72, 0, ${0.1 + intensity * 0.7})`,
                }}>
                  {v}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
