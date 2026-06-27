// Shape of app/public/predictions.json (produced by analysis/export_predictions.py).

export interface DailyRecord {
  date: string;
  actual_revenue: number;
  predicted_revenue: number;
  high_demand_actual: boolean;
  high_demand_pred: boolean;
  high_demand_prob: number | null;
}

export interface LeaderboardRow {
  task: "regression" | "classification";
  model: string;
  test: {
    r2?: number; mae?: number; rmse?: number;
    accuracy?: number; precision?: number; recall?: number; f1?: number;
    roc_auc?: number | null;
  };
  train_r2?: number;
  train_accuracy?: number;
  cv_r2?: number;
  cv_f1?: number;
}

export interface Charts {
  summary: {
    n_days: number; closed_days: number; total_revenue: number;
    mean_daily_revenue: number; median_daily_revenue: number;
    mean_daily_orders: number; high_demand_share: number;
  };
  revenue_by_dow: { label: string; mean_revenue: number; mean_orders: number }[];
  revenue_by_month: { label: string; total_revenue: number; mean_revenue: number }[];
  weekend_contrast: { day_type: string; mean_revenue: number; mean_orders: number }[];
  revenue_histogram: { bin_left: number; bin_right: number; count: number }[];
  heatmap: { rows: string[]; cols: string[]; values: (number | null)[][] };
}

export interface Predictions {
  year: number;
  models: { regression: string[]; classification: string[]; sklearn_version: string | null };
  leaderboard: LeaderboardRow[];
  charts: Charts;
  daily: DailyRecord[];
}

export interface PredictResponse {
  date: string;
  predicted_revenue: number;
  high_demand: { label: "High" | "Normal"; probability: number | null };
  out_of_training_range: boolean;
}
