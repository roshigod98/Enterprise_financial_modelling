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
    order_size_units: int = 0          # used for cash metrics

    unit_cost_gbp: float = 0.0         # per unit
    shipping_cost_gbp: float = 0.0     # fixed (batch-level)
    setup_cost_gbp: float = 0.0        # fixed

    extra_fixed_costs_gbp: float = 0.0
    extra_variable_cost_per_unit_gbp: float = 0.0


@dataclass(frozen=True)
class ScenarioOutputs:
    sell_price_net: float
    revenue_gbp: float

    variable_cost_per_unit_gbp: float
    cogs_gbp: float
    fixed_costs_gbp: float

    profit_before_tax_gbp: float
    tax_gbp: float
    profit_after_tax_gbp: float

    gross_margin_pct: Optional[float]
    contribution_margin_gbp: float
    contribution_margin_pct: Optional[float]

    breakeven_units: Optional[int]          # ignores grant
    cash_breakeven_units: Optional[int]     # ignores grant, uses upfront cash
    upfront_cash_gbp: float                 # order_size * var_cost + fixed

    roi_on_upfront_pct: Optional[float]
    profit_per_unit_after_tax: Optional[float]
    effective_net_margin_pct: Optional[float]
    inventory_left_units: Optional[int]


def net_price_from_input(price: float, is_gross: bool, vat_rate: float) -> float:
    if vat_rate < 0:
        raise ValueError("vat_rate must be >= 0")
    return price / (1.0 + vat_rate) if is_gross else price


def compute_outputs(x: ScenarioInputs) -> ScenarioOutputs:
    sell_net = net_price_from_input(x.sell_price_input, x.sell_price_is_gross, x.vat_rate)

    variable_per_unit = x.unit_cost_gbp + x.extra_variable_cost_per_unit_gbp

    revenue = sell_net * x.units_sold
    cogs = variable_per_unit * x.units_sold

    fixed_costs = x.shipping_cost_gbp + x.setup_cost_gbp + x.extra_fixed_costs_gbp

    # Operating profit (before tax). Grant included here as income, but not used in break-even.
    pbt = revenue + x.grant_gbp - (cogs + fixed_costs)

    tax = 0.20 * pbt if pbt > 0 else 0.0
    pat = pbt - tax

    gross_margin = None
    if revenue > 0:
        gross_margin = (revenue - cogs) / revenue * 100.0

    contribution = sell_net - variable_per_unit
    contrib_pct = None
    if sell_net > 0:
        contrib_pct = (contribution / sell_net) * 100.0

    # Break-even ignores grant
    breakeven = None
    if contribution > 0:
        breakeven = ceil(fixed_costs / contribution) if fixed_costs > 0 else 0

    order_units = max(0, int(x.order_size_units))
    upfront_cash = (variable_per_unit * order_units) + fixed_costs

    cash_be = None
    if contribution > 0 and upfront_cash > 0:
        cash_be = ceil(upfront_cash / contribution)

    roi_upfront = None
    if upfront_cash > 0:
        roi_upfront = (pat / upfront_cash) * 100.0

    profit_per_unit = None
    if x.units_sold > 0:
        profit_per_unit = pat / x.units_sold

    effective_margin = None
    if revenue > 0:
        effective_margin = (pat / revenue) * 100.0

    inventory_left = None
    if order_units > 0:
        inventory_left = max(0, order_units - x.units_sold)

    return ScenarioOutputs(
        sell_price_net=sell_net,
        revenue_gbp=revenue,
        variable_cost_per_unit_gbp=variable_per_unit,
        cogs_gbp=cogs,
        fixed_costs_gbp=fixed_costs,
        profit_before_tax_gbp=pbt,
        tax_gbp=tax,
        profit_after_tax_gbp=pat,
        gross_margin_pct=gross_margin,
        contribution_margin_gbp=contribution,
        contribution_margin_pct=contrib_pct,
        breakeven_units=breakeven,
        cash_breakeven_units=cash_be,
        upfront_cash_gbp=upfront_cash,
        roi_on_upfront_pct=roi_upfront,
        profit_per_unit_after_tax=profit_per_unit,
        effective_net_margin_pct=effective_margin,
        inventory_left_units=inventory_left,
    )


def distribute_profit_after_tax(
    pat_gbp: float,
    charity_pct: float,
    ceo_pct: float,
    other_pct_each: float,
    n_others: int = 5,
):
    if pat_gbp <= 0:
        return {"charity": 0.0, "ceo": 0.0, "others": [0.0] * n_others, "distributable": 0.0}

    if not (0 <= charity_pct <= 1):
        raise ValueError("charity_pct must be 0..1")

    if abs((ceo_pct + other_pct_each * n_others) - 1.0) > 1e-6:
        raise ValueError("team split must sum to 1.0")

    charity = pat_gbp * charity_pct
    distributable = pat_gbp - charity
    ceo = distributable * ceo_pct
    others = [distributable * other_pct_each for _ in range(n_others)]
    return {"charity": charity, "ceo": ceo, "others": others, "distributable": distributable}
