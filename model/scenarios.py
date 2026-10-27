from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from model.demand import DemandCurve
from model.finance import ScenarioInputs, compute_outputs, scenario_for_units, ScenarioOutputs


def linspace_inclusive(a: float, b: float, n: int) -> np.ndarray:
    if n < 2:
        return np.array([a], dtype=float)
    return np.linspace(a, b, n)


def sweep_profit_over_price(
    base: ScenarioInputs,
    price_min: float,
    price_max: float,
    n: int = 80,
    units_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    prices = linspace_inclusive(price_min, price_max, n)
    profits = np.zeros_like(prices, dtype=float)
    for i, p in enumerate(prices):
        x = replace(base, sell_price_input=float(p))
        if units_override is not None:
            x = scenario_for_units(x, int(units_override))
        profits[i] = compute_outputs(x).profit_after_tax_gbp
    return prices, profits


def profit_band_over_price(
    base: ScenarioInputs,
    price_min: float,
    price_max: float,
    units_min: int,
    units_max: int,
    unit_cost_min: float,
    unit_cost_max: float,
    n_price: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each price, compute min/max profit across a rectangle of uncertainty:
      units in [units_min, units_max]
      unit_cost in [unit_cost_min, unit_cost_max]
    """
    prices = linspace_inclusive(price_min, price_max, n_price)

    lo = np.zeros_like(prices, dtype=float)
    hi = np.zeros_like(prices, dtype=float)

    u_lo, u_hi = sorted([int(units_min), int(units_max)])
    c_lo, c_hi = sorted([float(unit_cost_min), float(unit_cost_max)])

    for i, p in enumerate(prices):
        # All corners matter: more sales can actually *reduce* profit if the
        # unit contribution is negative. Min/max input values are not probabilities.
        outcomes = [
            compute_outputs(
                scenario_for_units(replace(base, sell_price_input=float(p), unit_cost_gbp=c), u)
            ).profit_after_tax_gbp
            for u in (u_lo, u_hi)
            for c in (c_lo, c_hi)
        ]
        lo[i], hi[i] = min(outcomes), max(outcomes)

    return prices, lo, hi


def profit_heatmap_price_units(
    base: ScenarioInputs,
    price_min: float,
    price_max: float,
    units_min: int,
    units_max: int,
    n_price: int = 50,
    n_units: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = linspace_inclusive(price_min, price_max, n_price)

    if not base.allow_reorder:
        units_min, units_max = min(units_min, base.order_size_units), min(units_max, base.order_size_units)
    raw_units = np.linspace(units_min, units_max, n_units)
    units = np.unique(np.round(raw_units).astype(int))
    units = units[(units >= min(units_min, units_max)) & (units <= max(units_min, units_max))]
    if len(units) < 2:
        units = np.array([units_min, units_max], dtype=int)
        units = np.unique(units)

    z = np.zeros((len(units), len(prices)), dtype=float)
    for ui, u in enumerate(units):
        for pi, p in enumerate(prices):
            x = scenario_for_units(replace(base, sell_price_input=float(p)), int(u))
            z[ui, pi] = compute_outputs(x).profit_after_tax_gbp
    return prices, units, z


def profit_curve_with_demand(
    base: ScenarioInputs,
    price_min: float,
    price_max: float,
    demand: DemandCurve,
    n_price: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = linspace_inclusive(price_min, price_max, n_price)
    units = np.zeros_like(prices, dtype=int)
    profits = np.zeros_like(prices, dtype=float)

    for i, p in enumerate(prices):
        u = demand.predict_units(float(p))
        x = scenario_for_units(replace(base, sell_price_input=float(p)), int(u))
        units[i] = x.units_sold
        out = compute_outputs(x)
        profits[i] = out.profit_after_tax_gbp

    return prices, units, profits


def scenario_extrema(
    base: ScenarioInputs, price_min: float, price_max: float, units_min: int,
    units_max: int, unit_cost_min: float, unit_cost_max: float,
    demand: DemandCurve | None = None,
) -> tuple[ScenarioOutputs, ScenarioOutputs]:
    """Manual rectangle extrema, or sampled price-linked demand outcomes.

    Demand mode uses 201 price points, not independent price/volume corners.
    Its range is deliberately labelled sampled in the interface.
    """
    prices = (price_min, price_max) if demand is None else linspace_inclusive(price_min, price_max, 201)
    outcomes = []
    for price in prices:
        volumes = (units_min, units_max) if demand is None else (demand.predict_units(float(price)),)
        for units in volumes:
            for cost in (unit_cost_min, unit_cost_max):
                x = scenario_for_units(replace(base, sell_price_input=float(price), unit_cost_gbp=cost), units)
                outcomes.append(compute_outputs(x))
    return min(outcomes, key=lambda o:o.profit_after_tax_gbp), max(outcomes, key=lambda o:o.profit_after_tax_gbp)
