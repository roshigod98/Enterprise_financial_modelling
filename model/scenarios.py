from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from model.demand import DemandCurve
from model.finance import ScenarioInputs, compute_outputs


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
            x = replace(x, units_sold=int(units_override))
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
        worst = replace(base, sell_price_input=float(p), units_sold=u_lo, unit_cost_gbp=c_hi)
        best = replace(base, sell_price_input=float(p), units_sold=u_hi, unit_cost_gbp=c_lo)
        lo[i] = compute_outputs(worst).profit_after_tax_gbp
        hi[i] = compute_outputs(best).profit_after_tax_gbp

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

    raw_units = np.linspace(units_min, units_max, n_units)
    units = np.unique(np.round(raw_units).astype(int))
    units = units[(units >= min(units_min, units_max)) & (units <= max(units_min, units_max))]
    if len(units) < 2:
        units = np.array([units_min, units_max], dtype=int)
        units = np.unique(units)

    z = np.zeros((len(units), len(prices)), dtype=float)
    for ui, u in enumerate(units):
        for pi, p in enumerate(prices):
            x = replace(base, sell_price_input=float(p), units_sold=int(u))
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
        units[i] = u
        x = replace(base, sell_price_input=float(p), units_sold=int(u))
        out = compute_outputs(x)
        profits[i] = out.profit_after_tax_gbp

    return prices, units, profits
