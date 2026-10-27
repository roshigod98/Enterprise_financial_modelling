"""Trading results and a simplified fully funded, single-period cash ledger.

All amounts are GBP on the selected modelling basis. ``grant_gbp`` is retained
for API compatibility but means repayable goodwill-loan principal, never income.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, isclose
from numbers import Integral


@dataclass(frozen=True)
class ScenarioInputs:
    vat_rate: float = 0.20
    grant_gbp: float = 1000.0
    sell_price_input: float = 0.0
    sell_price_is_gross: bool = True
    units_sold: int = 0
    order_size_units: int = 0  # initial batch; reorders use the same batch size
    unit_cost_gbp: float = 0.0
    shipping_cost_gbp: float = 0.0  # charged once, including in reorder scenarios
    setup_cost_gbp: float = 0.0
    extra_fixed_costs_gbp: float = 0.0
    extra_variable_cost_per_unit_gbp: float = 0.0
    allow_reorder: bool = False


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
    gross_margin_pct: float | None
    contribution_margin_gbp: float
    contribution_margin_pct: float | None
    breakeven_units: int | None
    cash_breakeven_units: int | None
    upfront_cash_gbp: float
    roi_on_upfront_pct: float | None
    profit_per_unit_after_tax: float | None
    effective_net_margin_pct: float | None
    inventory_left_units: int
    ordered_units: int
    trading_breakeven_feasible: bool
    cash_recovery_feasible: bool
    funding_gap_gbp: float
    cash_before_loan_repayment_gbp: float
    loan_repayment_feasible: bool
    cash_after_loan_repayment_gbp: float
    distribution_capacity_gbp: float


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def effective_order_units(units: int, base_order: int, allow_reorder: bool) -> int:
    """Purchase whole batches; shipping remains a one-time scenario assumption."""
    if base_order <= 0:
        return 0
    return max(base_order, ((units + base_order - 1) // base_order) * base_order) if allow_reorder else base_order


def scenario_for_units(base: ScenarioInputs, units: int) -> ScenarioInputs:
    """Apply inventory limits consistently in sensitivity charts and simulations."""
    from dataclasses import replace
    sold = max(0, units) if base.allow_reorder else min(max(0, units), base.order_size_units)
    return replace(base, units_sold=sold)


def net_price_from_input(price: float, is_gross: bool, vat_rate: float) -> float:
    _nonnegative('price', price)
    _nonnegative('vat_rate', vat_rate)
    return price / (1.0 + vat_rate) if is_gross else price


def _ceil_units(value: float) -> int:
    """Avoid a spurious extra unit for a ratio such as 3.0000000000000004."""
    nearest = round(value)
    return nearest if isclose(value, nearest, rel_tol=0, abs_tol=1e-10) else ceil(value)


def compute_outputs(x: ScenarioInputs) -> ScenarioOutputs:
    for name in ('grant_gbp', 'unit_cost_gbp', 'shipping_cost_gbp', 'setup_cost_gbp',
                 'extra_fixed_costs_gbp', 'extra_variable_cost_per_unit_gbp'):
        _nonnegative(name, getattr(x, name))
    for name in ('units_sold', 'order_size_units'):
        value = getattr(x, name)
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if x.units_sold > x.order_size_units and (not x.allow_reorder or x.order_size_units == 0):
        raise ValueError('Sales exceed purchased inventory; enable reorders with a positive batch size')

    price = net_price_from_input(x.sell_price_input, x.sell_price_is_gross, x.vat_rate)
    cost = x.unit_cost_gbp + x.extra_variable_cost_per_unit_gbp
    fixed = x.shipping_cost_gbp + x.setup_cost_gbp + x.extra_fixed_costs_gbp
    ordered = effective_order_units(x.units_sold, x.order_size_units, x.allow_reorder)
    revenue = price * x.units_sold
    cogs = cost * x.units_sold
    contribution = price - cost
    pbt = revenue - cogs - fixed
    tax = .20 * max(pbt, 0.0)  # illustrative assumption, not confirmed liability
    pat = pbt - tax
    upfront = ordered * cost + fixed

    # At zero sales and zero fixed costs, trading is already at zero profit,
    # even if subsequent sales would have a negative contribution.
    breakeven = 0 if fixed == 0 else (_ceil_units(fixed / contribution) if contribution > 0 else None)
    cash_be = 0 if upfront == 0 else (_ceil_units(upfront / price) if price > 0 else None)

    # Assume external capital fills any funding gap. Unspent loan cash remains
    # cash, never profit. Reserve tax before assessing full loan repayment.
    gap = max(upfront - x.grant_gbp, 0.0)
    cash_before = x.grant_gbp + gap + revenue - upfront - tax
    after_repayment = cash_before - x.grant_gbp
    # Also recover the external capital before offering any charity/team payout.
    capacity = min(max(pat, 0.0), max(after_repayment - gap, 0.0))
    return ScenarioOutputs(
        sell_price_net=price, revenue_gbp=revenue, variable_cost_per_unit_gbp=cost,
        cogs_gbp=cogs, fixed_costs_gbp=fixed, profit_before_tax_gbp=pbt,
        tax_gbp=tax, profit_after_tax_gbp=pat,
        gross_margin_pct=(revenue-cogs)/revenue*100 if revenue else None,
        contribution_margin_gbp=contribution,
        contribution_margin_pct=contribution/price*100 if price else None,
        breakeven_units=breakeven, cash_breakeven_units=cash_be, upfront_cash_gbp=upfront,
        roi_on_upfront_pct=pat/upfront*100 if upfront else None,
        profit_per_unit_after_tax=pat/x.units_sold if x.units_sold else None,
        effective_net_margin_pct=pat/revenue*100 if revenue else None,
        inventory_left_units=ordered-x.units_sold, ordered_units=ordered,
        trading_breakeven_feasible=breakeven is not None and breakeven <= ordered,
        cash_recovery_feasible=cash_be is not None and cash_be <= ordered,
        funding_gap_gbp=gap, cash_before_loan_repayment_gbp=cash_before,
        loan_repayment_feasible=after_repayment >= -1e-9,
        cash_after_loan_repayment_gbp=after_repayment,
        distribution_capacity_gbp=capacity,
    )


def distribute_profit_after_tax(
    pat_gbp: float, charity_pct: float, ceo_pct: float, other_pct_each: float,
    n_others: int = 5,
) -> dict[str, float | list[float]]:
    """Allocate an available profit pool; callers supply cash-limited capacity."""
    if not isfinite(pat_gbp):
        raise ValueError('profit must be finite')
    if not isinstance(n_others, Integral) or n_others < 0:
        raise ValueError('n_others must be a nonnegative integer')
    for name, share in [('charity',charity_pct), ('ceo',ceo_pct), ('other',other_pct_each)]:
        if not isfinite(share) or not 0 <= share <= 1:
            raise ValueError(f'{name} share must be 0..1')
    if abs(ceo_pct + other_pct_each*n_others - 1) > 1e-6:
        raise ValueError('team split must sum to 1.0')
    pool = max(pat_gbp, 0.0)
    charity = pool * charity_pct
    distributable = pool - charity
    return {'charity':charity, 'ceo':distributable*ceo_pct,
            'others':[distributable*other_pct_each for _ in range(n_others)],
            'distributable':distributable}
