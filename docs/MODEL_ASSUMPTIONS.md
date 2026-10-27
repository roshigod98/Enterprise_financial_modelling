# Model assumptions and financial definitions

This is an educational *scenario model*, not audited accounts or tax advice. The financial calculations are intended to make commercial choices inspectable.

## Core inputs

- `sell_price_input`: user-entered customer selling price per deck. When `sell_price_is_gross=True`, the model divides by `1 + vat_rate` to produce an illustrative net price.
- `unit_cost_gbp`: supplier purchase cost per deck at the selected quantity.
- `extra_variable_cost_per_unit_gbp`: additional per-deck cost.
- `units_sold` (`U`): number of decks sold.
- `order_size_units` (`Q`): decks bought, including any modelled reorders.
- `shipping_cost_gbp`, `setup_cost_gbp` and `extra_fixed_costs_gbp`: fixed costs.
- `grant_gbp`: **legacy code name**; used for Tycoon *goodwill loan* amount. The funding is repayable financing and does not enter profit.

## Calculations

Let `P` be net price, `C` be supplier-plus-variable cost per deck, and `F` be total fixed costs.

| Measure | Formula | Interpretation |
| --- | --- | --- |
| Net revenue | `P × U` | Turnover, excluding modelled VAT |
| COGS | `C × U` | Cost attributed to decks sold |
| Trading profit before tax | `(P - C) × U - F` | Excludes loan funding |
| Illustrative tax | `0.20 × max(PBT, 0)` | Simplified *assumption*, not actual tax liability |
| Profit after illustrative tax | `PBT - tax` | The app's PAT |
| Contribution per deck | `P - C` | Incremental profit before fixed costs |
| Contribution margin | `(P - C) / P` | Undefined when `P=0` |
| Upfront cash requirement | `Q × C + F` | Cash to pay for all ordered inventory and initial expenses |
| Trading break-even | `ceil(F/(P-C))`, if `P>C` | First count covering fixed costs on a per-sale P&L basis |
| Cash recovery units | `ceil((Q×C + F)/P)`, if `P>0` | Units whose *net receipts* recover upfront cash |
| Inventory remaining | `max(Q-U, 0)` | Bought decks not yet sold |
| PAT per unit | `PAT/U` | Undefined if no sales |
| Effective net margin | `PAT/(P×U)` | Undefined if revenue is zero |

**Why the distinction matters:** Trading break-even can show zero when fixed costs are zero, although substantial cash was used to buy inventory. Conversely, the full-batch inventory purchase is already counted in upfront cash and must not be charged *again* through a contribution-margin denominator in the cash-recovery calculation.

Cash recovery is feasible within a single batch only when `cash_recovery_units <= Q`. This model does not forecast dates when sales cash will reach the school's account.

## Repayable Tycoon funding

The official programme calls startup support a **goodwill loan**, and sets a **£1,000 per-school ceiling for KS3–4** in the relevant competition category. The loan is repayable to the Peter Jones Foundation; teams may allocate profits after repayment. The initial funding is **not revenue**. [Official FAQ](https://tycoon.com/faqs/) · [Rules](https://tycoon.com/terms-conditions/)

The dashboard's simplified *loan repayment feasibility* test asks whether net trading receipts can recover all upfront outlay. This is conservative and **does not replace a cash ledger** of initial loan disbursement, unspent loan cash, repayment timetable, VAT settlements, and school-bill collections.

For illustrative purposes only, after a positive post-tax trading result and sufficient cash recovery, the app can allocate 30% to charity and split the remainder 45% to the CEO and 11% each to five team members. These percentages are user-selectable assumptions, **not verified distributions** or Foundation rules.

## VAT, tax and accounting caveats

The code uses a 20% VAT price adjustment and flat 20% profit-tax illustration. Whether the business should actually deduct output VAT, reclaim input VAT or pay corporation tax depends on its legal and school finance arrangements. Real cash, tax and accounting profit can therefore differ substantially from the displayed scenario. Before public financial claims, verify treatment with the school's finance staff.

Unknown supplier shipping, VAT status and MOQ values are **not £0**. Some blank cells are temporarily exposed as editable zero-cost *placeholders* in the dashboard; these scenarios should be treated as **incomplete quotes**, not verified supplier economics.

The simplified P&L charges fixed shipping once, and COGS for units sold. Multiple production batches may require repeated shipping or different supplier unit prices: both remain model limitations.

## Range charts and Monte Carlo

- Manual price/volume heatmaps plot hypothetical combinations of prices and units; they do **not** imply those combinations are equally likely.
- The price band evaluates four input extremes of volume and unit cost at each price; it is a mathematical envelope, **not a statistical confidence interval**.
- Optional demand mode interpolates between user-chosen `(price, units)` anchor points and clips to minimum and maximum quantities. It is not an empirically estimated demand curve.
- `model/risk.py` samples uniform selling prices and unit costs, and uniform integer quantities (or quantities from the assumed demand curve), using a fixed reproducible seed. Its returned values are **fractions of deliberately generated scenarios**, not calibrated predictions of a real business's success chances.
- Modelling assumptions such as stock constraints and reorders must be held consistent when reading the outputs.

## Known limitations / next research steps

Source timestamps and supplier quotes should be independently verified; demand can be fitted to actual sales or a designed willingness-to-pay experiment; school-bill settlement dates, inventory write-downs and remaining goodwill-loan cash should be modelled explicitly; and any after-tax payout should be checked against actual school finance and charity policies.
