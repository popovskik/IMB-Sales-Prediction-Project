import type { ReactNode } from "react";
import type { LeaderboardRow, Predictions } from "../types";
import { InfoTip } from "./InfoTip";

const n = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

/** A table header cell with an explanatory bubble for the statistical term. */
function Th({ children, tip, align = "center" }: { children: ReactNode; tip: ReactNode; align?: "center" | "end" }) {
  return (
    <th>
      <span style={{ display: "inline-flex", alignItems: "center" }}>
        {children}
        <InfoTip placement="top" align={align}>{tip}</InfoTip>
      </span>
    </th>
  );
}

export function Scorecard({ leaderboard, models }: { leaderboard: LeaderboardRow[]; models: Predictions["models"] }) {
  const reg = leaderboard.filter((r) => r.task === "regression");
  const clf = leaderboard.filter((r) => r.task === "classification");

  // Deployed (best) models: regression by CV R², classification by ROC-AUC.
  const bestReg = reg.filter((r) => r.model !== "XGBoost (+lags)")
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

      <p style={{ color: "var(--ink-3)", fontSize: 12, marginTop: 14, marginBottom: 0 }}>
        Trained with scikit-learn {models.sklearn_version ?? "?"}. Features:{" "}
        {models.regression.join(", ")}.
      </p>
    </section>
  );
}
