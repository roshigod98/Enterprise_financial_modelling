# Supplier data audit

These are historical research notes, not current supplier offers. Source dates,
original quotations, import charges and legal VAT treatment have not been verified.
No new prices have been invented. `data/quotes_original.csv` preserves the exact
pre-audit CSV; the application reads only `data/quotes_clean.csv`.

## Column corrections

Ten rows contained an extra blank before the purchase-price columns. Their
500-deck unit price appeared under `TotalCost500GBP`, their batch total under
`ShippingCostGBP`, and subsequent fields were displaced. The consistent
`unit price × 500 = batch total` pattern identifies the alignment error.
One empty cell at column 7 was removed and a trailing blank added in each row.

| Supplier | 500-deck unit cost GBP | 500-deck total GBP | Shipping GBP |
| --- | ---: | ---: | ---: |
| Ningbo Yinzhou Chaoyang | 1.01 | 505 | Unknown |
| Hangzhou Wonder Tree | 0.94 | 470 | 160 |
| Wuhan Jarmoo (row 1) | 0.52 | 260 | 127 |
| Yangzhou Jumbay | 0.67 | 335 | Unknown |
| Shenzhen Xmt | 1.05 | 525 | Unknown |
| Fullbright | 1.08 | 540 | Unknown |
| CS Poker | 0.69 | 345 | Unknown |
| Yiwu Pengle | 0.59 | 295 | 134 |
| Guilin Kangrui | 0.85 | 425 | 160 |
| Game Foundary | 1.04 | 520 | Unknown |

The same shift restores Yangzhou's 250-deck quote of £0.98 and Ningbo's
62x87 mm / 54-card specification. Wuhan row 1 also contained separately
misaligned descriptive text: its explicit `63x88` and `54` values were restored
to size and cards-per-deck, and `500pcs pokers` retained as a note rather than
interpreted as 500 cards per deck.

Alibaba, Printed4You and Wuhan row 2 had correctly aligned cost columns and
were retained. All 13 suppliers remain present. Spelling and uncertain supplier
identity such as `Alibaba 1?` are preserved from the source.

## Incomplete quotes

Blank numeric cells remain `None`. Invalid, negative or nonfinite prices are
rejected. The loader checks that a stated 500-deck unit cost agrees with its
batch total within £1 (rounding tolerance). Unknown shipping excludes a supplier
from financial rankings; a selected supplier requires an explicit manual shipping
assumption. Known minimum-order constraints are enforced; unknown minima and
fallback price tiers are labelled. Costs use quoted GBP without automatic input
VAT reclaim, currency conversion or guessed duties. A positive quote is not proof
that every landed cost is covered.
