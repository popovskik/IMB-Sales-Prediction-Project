/** "How this works" + an honest model-card / limitations box, so the project's
 *  strongest asset — its honesty — is visible to anyone who only opens the live app. */
export function About() {
  return (
    <section className="card">
      <div className="eyebrow">How this works</div>
      <h2 style={{ fontSize: 20, margin: "4px 0 12px" }}>About this dashboard</h2>

      <p style={{ color: "var(--ink-2)", marginTop: 0 }}>
        Two models were trained <strong>offline</strong> on one year (2015) of a pizza
        restaurant's orders and saved to disk. This dashboard calls a <strong>live API</strong>{" "}
        that loads those models and predicts from whatever date you pick — no training happens
        when you click <em>Forecast</em>. The charts and the scorecard read pre-computed results.
        "Confidence" is how sure the busy-day model is that a day will be high-demand (50% is a
        coin-flip; closer to 100% is more certain).
      </p>

      <div
        style={{
          marginTop: 8, padding: "12px 16px", borderRadius: "var(--r-md)",
          background: "var(--amber-soft)", border: "1px solid #e0c074",
        }}
      >
        <div className="eyebrow" style={{ color: "var(--amber)" }}>Honest limitations</div>
        <p style={{ margin: "6px 0 0", color: "var(--ink)" }}>
          <strong>Revenue forecasts are approximate.</strong> With only one year of data and
          large day-to-day swings, the dollar figure is a planning ballpark — not a financial
          commitment. The <strong>busy-day classifier</strong> (ROC-AUC ≈ 0.74 from the calendar
          alone) is a useful staffing aid, but a guide, not a guarantee. More years of data —
          and signals like promotions, weather, and local events — would sharpen both.
        </p>
      </div>
    </section>
  );
}
