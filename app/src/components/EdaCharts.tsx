import type { ReactNode } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { Charts } from "../types";
import { InfoTip } from "./InfoTip";

const ORANGE = "#ff4800";

function Empty({ label }: { label: string }) {
  return <div style={{ color: "var(--ink-3)", padding: 24, textAlign: "center" }}>No data for {label}.</div>;
}

export function EdaCharts({ charts }: { charts: Charts }) {
  return (
    <section className="card">
      <div className="eyebrow">Exploratory analysis</div>
      <h2 style={{ fontSize: 20, margin: "4px 0 18px" }}>Demand patterns across 2015</h2>

      <div className="eda-grid">
        <Panel title="Mean daily revenue by month"
               tip={<>The average day's revenue in each month. Use it to spot <strong>seasonal trends</strong> — busier or quieter times of year.</>}>
          {charts.revenue_by_month?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={charts.revenue_by_month}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
                <XAxis dataKey="label" fontSize={12} />
                <YAxis fontSize={12} width={48} />
                <Tooltip formatter={(v: number) => `$${Math.round(v).toLocaleString()}`} />
                <Line type="monotone" dataKey="mean_revenue" stroke={ORANGE} strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : <Empty label="monthly revenue" />}
        </Panel>

        <Panel title="Mean daily revenue by day of week"
               tip={<>The average revenue for each weekday across the year. The clearest pattern in the data — <strong>weekends and Fridays earn the most.</strong></>}>
          {charts.revenue_by_dow?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={charts.revenue_by_dow}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
                <XAxis dataKey="label" fontSize={12} />
                <YAxis fontSize={12} width={48} />
                <Tooltip formatter={(v: number) => `$${Math.round(v).toLocaleString()}`} />
                <Bar dataKey="mean_revenue" fill={ORANGE} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty label="day-of-week revenue" />}
        </Panel>

        <Panel title="Distribution of daily revenue"
               tip={<>How many days fell into each revenue range. A tall middle means most days are typical; the bars far left are the rare <strong>closed days</strong>.</>}>
          {charts.revenue_histogram?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={charts.revenue_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line-soft)" />
                <XAxis dataKey="bin_left" fontSize={11} tickFormatter={(v: number) => `$${Math.round(v / 1000)}k`} />
                <YAxis fontSize={12} width={32} />
                <Tooltip formatter={(v: number) => `${v} days`} />
                <Bar dataKey="count" fill={ORANGE} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty label="revenue distribution" />}
        </Panel>

        <Panel title="Revenue heatmap — day of week × month"
               tip={<>Each cell is the average revenue for that weekday in that month. <strong>Darker = higher revenue.</strong> Read across a row to see a weekday over the year, or down a column to compare weekdays within a month.</>}>
          {charts.heatmap?.values?.length ? <Heatmap heat={charts.heatmap} /> : <Empty label="heatmap" />}
        </Panel>
      </div>
    </section>
  );
}

function Panel({ title, tip, children }: { title: string; tip: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div style={{ fontWeight: 600, fontSize: 13, color: "var(--ink-2)", marginBottom: 8, display: "flex", alignItems: "center" }}>
        {title}
        <InfoTip placement="bottom">{tip}</InfoTip>
      </div>
      {children}
    </div>
  );
}

function Heatmap({ heat }: { heat: Charts["heatmap"] }) {
  const flat = heat.values.flat().filter((v): v is number => v != null);
  const max = Math.max(...flat, 1);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "separate", borderSpacing: 2, minWidth: 360 }}>
        <thead>
          <tr>
            <th></th>
            {heat.cols.map((c) => <th key={c} style={{ fontSize: 10, color: "var(--ink-3)", padding: 2 }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {heat.rows.map((row, i) => (
            <tr key={row}>
              <td style={{ fontSize: 11, color: "var(--ink-2)", paddingRight: 6 }}>{row}</td>
              {heat.values[i].map((v, j) => {
                const a = v == null ? 0 : v / max;
                return (
                  <td key={j} title={v == null ? "n/a" : `$${Math.round(v)}`}
                      style={{ width: 26, height: 22, borderRadius: 4, padding: 0,
                               background: `rgba(255,72,0,${0.12 + a * 0.85})` }} />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {/* legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, fontSize: 11, color: "var(--ink-3)" }}>
        <span>Lower</span>
        <span style={{ width: 90, height: 10, borderRadius: 5,
                       background: "linear-gradient(90deg, rgba(255,72,0,0.12), rgba(255,72,0,0.97))" }} />
        <span>Higher</span>
        <span style={{ marginLeft: 4 }}>· avg daily revenue (${Math.round(max).toLocaleString()} max)</span>
      </div>
    </div>
  );
}
