# Model assumptions and financial definitions

This educational scenario model is not audited accounts or a determination of
UK tax liability. Reported enterprise results are separate from model outputs.
All displayed amounts are GBP. School-bill sales have **no card-processing fees**.

## Inputs and accounting basis

`P` is the modelled net selling price, `C` is supplier cost plus extra variable
cost per deck, `U` is units sold, `Q` is total purchased inventory, and `F` is
shipping + setup + extra fixed expenditure. Additional per-deck costs are assumed
to be committed for the full purchased quantity, preserving the project's
original cash assumption. Unsold inventory remains at cost; no write-down or
salvage income is assumed.

Customer-price mode uses `P = entered price / (1 + 0.20)`; net-price mode uses
the entered price directly. **VAT registration, input VAT recovery and the school's
actual financial treatment are unverified.** Purchase costs stay on their quoted
GBP basis, with no automatic input VAT adjustment. Set the price mode deliberately;
do not interpret this convention as confirmation of taxes payable.

Inputs must be finite and nonnegative, with integer quantities. Sales cannot
exceed purchased inventory without explicit reorders. Zero sales and losses are
valid scenarios, with undefined percentage margins shown as a dash.

## Trading performance and cash recovery

| Measure | Formula |
| --- | --- |
| Net revenue | `P × U` |
| COGS | `C × U` |
| Trading profit before tax (PBT) | `P × U − C × U − F` |
| Illustrative profit tax | `20% × max(PBT, 0)` |
| Illustrative profit after tax (PAT) | `PBT − illustrative tax` |
| Contribution per deck | `P − C` |
| Upfront cash requirement | `Q × C + F` |
| Trading break-even | `ceil(F / (P − C))` when contribution is positive |
| Cash recovery units | `ceil((Q × C + F) / P)` when net price is positive |
| Inventory remaining | `Q − U` |
| ROI on upfront expenditure | `PAT / upfront × 100` |

At zero fixed costs, trading is at zero profit at zero sales, even if subsequent
sales lose money. With positive fixed costs and nonpositive contribution, trading
break-even is impossible. Zero upfront expenditure is recovered at zero sales;
otherwise zero-price sales cannot recover it. A numerical threshold beyond `Q`
is explicitly labelled unreachable within purchased inventory. Cash recovery is
**before illustrative tax and financing**, and the denominator is net receipts,
not contribution: inventory has already been purchased upfront.

## Goodwill loan, external capital and distributions

The £1,000 goodwill-loan ceiling is a **project funding assumption**. The legacy
API name `grant_gbp` means repayable loan principal `L`, never revenue or trading
profit. Official programme background: [Tycoon FAQ](https://tycoon.com/faqs/) and
[terms](https://tycoon.com/terms-conditions/). Confirm current rules independently.

The single-period ledger assumes that external capital fills any initial gap
`E = max(upfront − L, 0)`. Its displayed gap must actually be funded for the
scenario to proceed. With net receipts collected and illustrative tax reserved:

- Cash before loan repayment = `L + E + revenue − upfront − tax`.
- Full repayment is feasible if that cash is at least `L`.
- Cash after full repayment = cash before repayment minus `L`; a negative value
  represents a shortfall, not an actual repayment made.
- Cash available after also recovering external capital = `revenue − upfront − tax`.
- Allocation pool = `min(max(PAT, 0), max(cash available after capital recovery, 0))`.

This prevents accounting profit held in unsold stock from being distributed as
cash. Loan principal is repaid from the ledger; it is **not deducted from PAT**.
Unspent loan proceeds remain cash and are never distributed as earned profit.
Default charity is **30% of the available allocation pool**, followed by **45% to
the CEO and 11% each to five members** of the remaining team pool. With sufficient
cash this is the original PAT allocation; insufficient cash reduces the pool.
These are illustrative allocations, not verified payments or competition rules.
At 100% charity all available allocation goes to charity, with no team payout.

## Inventory and reorders

Without reorders, charts and simulations cap assumed sales at the original batch.
With reorders, whole batches of that size are bought: `Q = max(batch, ceil(U/batch) × batch)`.
The model conservatively reserves cash for **all** those batches upfront. It does
not assume earlier sales finance later purchases. Supplier unit price stays fixed,
and shipping, setup and other fixed costs are charged **once per scenario**.
Repeated shipping, price-tier changes, delivery timing and intra-period funding
are not modelled. A cash-recovery marker refers to the currently modelled purchased
quantity; the cash curve shows steps when reorder quantities change.

## Scenarios, demand and Monte Carlo

Manual scenario extrema evaluate all price/volume/cost corners, including cases
where higher sales increase a loss. Bands evaluate all volume/cost corners at each
price; these are envelopes, not confidence intervals. Heatmaps represent independent
hypothetical price/volume combinations. Their zero contour is trading break-even.

Demand uses two user-selected anchors and linear interpolation/extrapolation,
rounded to integer decks and clipped to demand limits and available inventory.
Identical prices require identical quantities. Rising demand anchors are allowed
as an explicit user assumption; the model does not enforce an economic law.
Demand-linked scenario bounds sample **201 prices**, retaining price/volume
correlation. They are evaluated bounds, not guaranteed continuous extrema.

Monte Carlo samples uniform prices and costs and uniform integer sales, or the
assumed demand curve with optional bounded multiplicative noise. The default uses
**2,000 samples and seed 42**. Inventory, tax and repayment use the same finance
functions as the rest of the app. Scenario shares are **not calibrated commercial
probabilities or verified outcomes**. Re-running unchanged inputs is reproducible;
changing assumptions changes the experiment. The widest relative input range in
the narrative is a description of inputs, not a measured financial sensitivity.

## Data and limitations

See [supplier data audit](DATA_NOTES.md) for CSV alignment repairs and the retained
original. Missing shipping/costs exclude supplier rankings; unknown MOQ and VAT
remain visible. Fallback price tiers are assumptions, not confirmed quotes at the
selected order size. Historical prices, supplier identity, import charges, demand,
collection dates, VAT settlements, competition rules and actual profit
appropriations require confirmation before a real decision.
