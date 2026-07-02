# AI-Workflow Reflection

## Tools and MCP servers used

- **Claude Code** (Anthropic) as the primary agent — drove the whole pipeline: data
  cleaning, EDA, feature engineering, model training, the FastAPI service, the React
  dashboard, and this documentation.
- **Compound-engineering skill workflow** — `brainstorm` → `plan` → `doc-review` → `work`.
  The brainstorm and plan documents (in the course `Project Work` folder) framed scope before
  any code; a multi-persona document review hardened the plan and caught two real gaps before
  implementation (a cross-directory import that would have broken the Railway deploy, and an
  undefined producer/schema for `predictions.json`).
- **Mermaid MCP server** — attempted for the architecture diagram (D0). It failed on this
  Windows machine (the renderer shells out to `npx`, which errors with `EPERM`), so the
  diagram was rendered deterministically with **matplotlib** instead. The Mermaid *source* is
  still kept in `architecture.md`.
- **Git / GitHub** — the GitHub CLI (`gh`) was not installed, so the repository was wired and
  pushed with plain `git` over HTTPS (credentials cached by the Windows credential manager).
- **Quarto** — for the analysis report (D1) and these slides (D4).

## How I verified the AI's output

I treated generated code as a draft to be proven, not trusted:

- **49 automated tests** across the analysis pipeline and the API, including leakage guards
  (lag features use only past data; scalers fit on the training split only), a
  reproducibility check (seed 42), and an **offline-vs-online feature-parity test** that
  compares the API's vendored feature code against the analysis source.
- **Manual sanity checks on the data**: confirmed line revenue against hand-computed
  `quantity × price`, found and fixed a real encoding bug (`pizza_types.csv` is Windows-1252,
  not UTF-8), and investigated the seven zero-order days (a genuine data-quality finding).
- **Scrutiny of the results, not just acceptance**: when XGBoost initially looked like it
  "lost" to the baseline, I dug in rather than hiding it — and found the real cause (the
  chronological hold-out tests on calendar positions unseen in a single year of training).
  That led to reporting cross-validation as the fairer primary metric, which is the honest
  and correct call.

## Synthetic data augmentation (disclosed)

The Maven dataset has no customer, table, or server identifiers. With the professor's
approval, I generated them synthetically to enrich the EDA and dashboard. This data is
clearly labelled as synthetic and is **isolated from model training** — enforced by tests
asserting that no synthetic field appears in the model feature sets and that the training
modules never import the augmentation code. The two models are trained only on real,
date-derived features.

## Rough cost and effort

A single focused session: brainstorm and plan (~1 hour of dialogue), then implementation
across eleven units. The agent did the bulk of the typing; my effort went into direction,
verifying outputs, the honest-evaluation judgment calls, and the deploy steps. The largest
time costs were dependency installs (the scientific-Python and Node toolchains) and the
XGBoost hyper-parameter search. I can explain every part of the pipeline and every modelling
decision in it.
