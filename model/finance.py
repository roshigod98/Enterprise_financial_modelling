from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Optional


@dataclass(frozen=True)
class ScenarioInputs:
    vat_rate: float = 0.20
    grant_gbp: float = 1000.0

    sell_price_input: float = 0.0      # user enters
    sell_price_is_gross: bool = True   # inc VAT?

    units_sold: int = 0

    unit_cost_gbp: float = 0.0         # per unit
    shipping_cost_gbp: float = 0.0     # fixed (batch-level)
    setup_cost_gbp: float = 0.0        # fixed


@dataclass(frozen=True)
class ScenarioOutputs:
    sell_price_net: float
    revenue_gbp: float
    cogs_gbp: float
    fixed_costs_gbp: float
    profit_gbp: float
    gross_margin_pct: Optional[float]
    breakeven_units: Optional[int]
    roi_pct: Optional[float]
    contribution_margin_gbp: float
    margin_of_safety_units: Optional[int]


def net_price_from_input(price: float, is_gross: bool, vat_rate: float) -> float:
    if vat_rate < 0:
        raise ValueError("vat_rate must be >= 0")
    return price / (1.0 + vat_rate) if is_gross else price


def compute_outputs(x: ScenarioInputs) -> ScenarioOutputs:
    sell_net = net_price_from_input(x.sell_price_input, x.sell_price_is_gross, x.vat_rate)

    revenue = sell_net * x.units_sold
    cogs = x.unit_cost_gbp * x.units_sold
    fixed_costs = x.shipping_cost_gbp + x.setup_cost_gbp
    profit = revenue + x.grant_gbp - (cogs + fixed_costs)

    gross_margin = None
    if revenue > 0:
        gross_margin = (revenue - cogs) / revenue * 100.0

    contribution = sell_net - x.unit_cost_gbp
    breakeven = None
    margin_of_safety = None
    
    if contribution > 0:
        needed = max(0.0, fixed_costs - x.grant_gbp)
        breakeven = ceil(needed / contribution)
        if x.units_sold >= breakeven:
            margin_of_safety = x.units_sold - breakeven

    roi = None
    total_investment = cogs + fixed_costs
    if total_investment > 0:
        roi = (profit / total_investment) * 100.0

    return ScenarioOutputs(
        sell_price_net=sell_net,
        revenue_gbp=revenue,
        cogs_gbp=cogs,
        fixed_costs_gbp=fixed_costs,
        profit_gbp=profit,
        gross_margin_pct=gross_margin,
        breakeven_units=breakeven,
        roi_pct=roi,
        contribution_margin_gbp=contribution,
        margin_of_safety_units=margin_of_safety,
    )


def distribute_profit(profit_gbp: float, charity_pct: float, ceo_pct: float, other_pct_each: float, n_others: int = 5):
    if profit_gbp <= 0:
        return {"charity": 0.0, "ceo": 0.0, "others": [0.0]*n_others}

    if not (0 <= charity_pct <= 1):
        raise ValueError("charity_pct must be 0..1")

    # Validate split sums to 1
    if abs((ceo_pct + other_pct_each*n_others) - 1.0) > 1e-6:
        raise ValueError("team split must sum to 1.0 (CEO + 5*others)")

    charity = profit_gbp * charity_pct
    remaining = profit_gbp - charity
    ceo = remaining * ceo_pct
    others = [remaining * other_pct_each for _ in range(n_others)]
    return {"charity": charity, "ceo": ceo, "others": others}
