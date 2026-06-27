import type { LeaderboardRow, Predictions } from "../types";

const n = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

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
            <tr><th>Model</th><th>CV R²</th><th>Test R²</th><th>Test MAE</th><th>Test RMSE</th><th>Train R²</th></tr>
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
            <tr><th>Model</th><th>CV F1</th><th>Test Acc</th><th>Test F1</th><th>ROC-AUC</th></tr>
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
