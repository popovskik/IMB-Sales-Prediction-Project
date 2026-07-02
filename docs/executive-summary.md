# Executive Summary — Restaurant Demand Analytics

**The question.** Can a pizza restaurant predict, from the calendar alone, *how busy a day
will be* and *how much it will earn* — so it can plan staffing, inventory, and purchasing
ahead of demand? We studied one full year (2015) of order data: about 21,000 orders.

**What the data shows.**

- **The week has a strong, dependable rhythm.** Weekends and Fridays are consistently the
  busiest and highest-earning days; quiet midweek days are equally predictable. There is
  also a mild seasonal lift in summer and around year-end.
- **"Will today be busy?" can be answered in advance — reasonably well.** Using nothing but
  the date, the model identifies high-demand days clearly better than guessing. It is a solid
  planning aid for setting shift levels, though not a guarantee for any single day.
- **"Exactly how much will we earn?" is much harder.** Day-to-day revenue swings are large,
  and a single year of history is not enough to pin down the dollar figure from the calendar
  alone. We can give a sensible ballpark, not a precise forecast.

**Why the revenue forecast is limited (stated honestly).** With only one year of data, the
months we tested the model on (November–December) are calendar positions it had never seen
while learning. That, plus genuine day-to-day randomness, caps how accurate a
calendar-only revenue forecast can be. We cross-checked with a dedicated time-series
forecasting model (SARIMA) — it does no better, which confirms the ceiling is the data,
not our choice of model. This is a data limitation, not a modelling mistake — and it
points directly at the fix.

**Recommendations.**

1. **Staff and stock to the weekly pattern** — weight Friday through Sunday heavily. This is
   the single most reliable signal in the data.
2. **Use the busy-day forecast for shift planning**, where "better than guessing" already
   creates real value, rather than for precise financial commitments.
3. **Invest in more data.** More years of history, plus promotions, weather, and local
   events, would turn the rough revenue estimate into a dependable forecast. This is the
   highest-value next step.

**Bottom line.** The calendar alone is enough to plan *staffing* with confidence and to
forecast *revenue* approximately. The path to precise revenue forecasting runs through more
data, not a fancier model.
