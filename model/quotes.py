from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Quote:
    supplier: str
    region: Optional[str]
    moq: Optional[int]

    unit_cost_250: Optional[float]
    unit_cost_500: Optional[float]
    total_cost_500: Optional[float]

    shipping_cost: Optional[float]
    vat_included: Optional[bool]

    # aesthetics/specs
    card_stock: Optional[str]
    finish: Optional[str]
    tuck_box: Optional[str]
    card_size_mm: Optional[str]
    cards_per_deck: Optional[int]
    tuck_box_finish: Optional[str]
    shrink_wrap: Optional[str]
    lead_time: Optional[str]
    payment_terms: Optional[str]
    notes: Optional[str]


def _none_if_blank(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    return None if s == "" or s.lower() in {"nan", "none"} else s


def load_quotes_csv(path: str) -> list[Quote]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        # Resolve relative paths against project root (parent of /model).
        p = Path(__file__).resolve().parents[1] / p
    df = pd.read_csv(p)
    quotes: list[Quote] = []

    for _, r in df.iterrows():
        def f(col): return _none_if_blank(r.get(col))

        def num(col):
            v = f(col)
            if v is None:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def num_int(col):
            v = num(col)
            return int(v) if v is not None else None

        vat_flag_raw = f("VATIncludedFlag")
        vat_included = None
        if vat_flag_raw is not None:
            if vat_flag_raw.lower() == "yes":
                vat_included = True
            elif vat_flag_raw.lower() == "no":
                vat_included = False

        quotes.append(Quote(
            supplier=str(f("Supplier") or "").strip(),
            region=f("Region"),
            moq=num_int("MOQ"),
            unit_cost_250=num("UnitCost250GBP"),
            unit_cost_500=num("UnitCost500GBP"),
            total_cost_500=num("TotalCost500GBP"),
            shipping_cost=num("ShippingCostGBP"),
            vat_included=vat_included,
            card_stock=f("CardStock"),
            finish=f("Finish"),
            tuck_box=f("CustomTuckBox"),
            card_size_mm=f("CardSizeMM"),
            cards_per_deck=num_int("CardsPerDeck"),
            tuck_box_finish=f("TuckBoxFinish"),
            shrink_wrap=f("ShrinkWrap"),
            lead_time=f("LeadTimeText"),
            payment_terms=f("PaymentTerms"),
            notes=f("Notes"),
        ))

    # drop empty supplier rows
    return [q for q in quotes if q.supplier]


def pick_unit_cost(q: Quote, order_size: int) -> tuple[Optional[float], str]:
    """
    Returns (unit_cost, note). note is empty when perfect match.
    """
    if order_size == 250 and q.unit_cost_250 is not None:
        return q.unit_cost_250, ""
    if order_size == 500 and q.unit_cost_500 is not None:
        return q.unit_cost_500, ""
    if order_size == 500 and q.total_cost_500 is not None:
        return q.total_cost_500 / 500.0, "inferred unit cost from total_cost_500"
    # fallback: use any known
    if q.unit_cost_500 is not None:
        return q.unit_cost_500, "fallback to unit_cost_500"
    if q.unit_cost_250 is not None:
        return q.unit_cost_250, "fallback to unit_cost_250"
    return None, "missing unit cost"
