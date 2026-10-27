# Portfolio finalisation plan

The user supplied the scope, financial rules, visual direction and execution approval.
Retain the Streamlit/Plotly interface and existing PR rather than rewrite the app.

1. Preserve the local worktree and IDE/environment; fetch and use PR #1. **Done.**
   Backup: `/private/tmp/oundle-backup-20261027/local-worktree.tgz` (local only).
2. Audit every source file and supplier row. Add regression tests before model fixes.
   Centralise inventory/reorder handling and cash available after tax and loan repayment.
   Preserve the 20% illustrative tax, upfront per-unit costs and one-time shipping assumption.
3. Correct the shifted supplier columns, retaining the original CSV and documenting changes.
   Reject malformed values; keep missing fields unknown. Exclude incomplete comparisons.
4. Polish existing charts and narrative, validate demand inputs, and simplify headline KPIs.
5. Update assumptions, README, deployment guidance and genuine running-app screenshots.
6. Run full tests, compile checks, browser smoke tests and a fresh dependency installation.
7. Review the final branch independently, fix material findings, push, verify CI, merge PR #1,
   and fast-forward the local main checkout. No force pushes or history rewriting.

Review focus: zero/negative contribution; overselling; reordered inventory; missing shipping;
negative-margin extrema; correlated demand vs independent scenarios; post-tax cash-limited
payouts; chart bounds and zero-value pies; schema drift and price-tier assumptions.

## Verification and review record (27 October 2026)

- Local regression and Streamlit interaction suite: 41 tests passed on Python 3.12.
- Compilation and `pip check` passed; no generated environments or IDE files tracked.
- Chromium rendered all inspected charts without browser JavaScript or app exceptions;
  desktop screenshots saved in `assets/screenshots/`, mobile layout also inspected.
- All 13 suppliers retained; original CSV is byte-identical to the pre-audit source.
- 2,000-sample Monte Carlo calculation measured about 0.04 seconds locally (not a benchmark).
- Independent read-only reviewer checked finance, charts, data and tests. Custom MOQ
  enforcement was identified and fixed with a regression test. Final self-review also
  fixed binary rounding at break-even and direct risk-call reorder inheritance.
- Fresh-environment verification and remote integration are performed after the
  implementation commit; GitHub Actions and PR #1 provide the durable remote record.
- Deployment needs an authenticated Streamlit Community Cloud account; no public
  deployment is claimed. Configuration and exact account-side steps are in README.
