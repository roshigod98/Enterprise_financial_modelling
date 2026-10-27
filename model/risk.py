from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from model.finance import ScenarioInputs, compute_outputs


def estimate_likelihoods(
    base: ScenarioInputs,
    price_min: float,
    price_max: float,
    unit_cost_min: float,
    unit_cost_max: float,
    units_min: int,
    units_max: int,
    grant_cap_gbp: float = 1000.0,
    n: int = 2000,
    seed: int = 42,
    demand_curve: Optional[object] = None,
    demand_noise_pct: float = 0.0,
    base_order_size_units: Optional[int] = None,
    allow_reorder: bool = False,
) -> dict[str, float]:
    """
    Monte Carlo likelihood estimates based on uniformly sampling within ranges.

    - If demand_curve is provided, units are derived from price via demand_curve.predict_units(price).
      Optional multiplicative noise can be applied via demand_noise_pct (e.g. 0.10 = ±10%).
    - Trading break-even: pretax trading profit >= 0 (loan excluded).
    - Repayment feasible: net sales receipts cover the initial cash outlay.
    - Funding cap: upfront outlay <= loan cap.
    Uniform what-if sampling yields *scenario shares*, not calibrated probabilities.
    """
    rng = np.random.default_rng(seed)

    p_samples = rng.uniform(price_min, price_max, size=n)
    c_samples = rng.uniform(unit_cost_min, unit_cost_max, size=n)

    if demand_curve is None:
        u_samples = rng.integers(
            low=min(units_min, units_max),
            high=max(units_min, units_max) + 1,
            size=n,
        )
    else:
        u_list = []
        for p in p_samples:
            u = int(demand_curve.predict_units(float(p)))
            u_list.append(u)
        u_samples = np.array(u_list, dtype=int)

        if demand_noise_pct > 0:
            noise = rng.uniform(1 - demand_noise_pct, 1 + demand_noise_pct, size=n)
            u_samples = np.round(u_samples * noise).astype(int)

    batch_size = int(base_order_size_units or base.order_size_units)
    if batch_size < 1:
        raise ValueError("order size must be at least one")
    u_samples = np.maximum(u_samples, 0)
    if not allow_reorder:
        u_samples = np.minimum(u_samples, batch_size)

    pbt = np.zeros(n, dtype=float)
    receipts = np.zeros(n, dtype=float)
    upfront = np.zeros(n, dtype=float)

    for i in range(n):
        units = int(u_samples[i])
        ordered = max(batch_size, ((units + batch_size - 1) // batch_size) * batch_size) if allow_reorder else batch_size
        xi = replace(
            base,
            sell_price_input=float(p_samples[i]),
            unit_cost_gbp=float(c_samples[i]),
            units_sold=units,
            order_size_units=ordered,
        )
        out = compute_outputs(xi)
        pbt[i] = out.profit_before_tax_gbp
        receipts[i] = out.revenue_gbp
        upfront[i] = out.upfront_cash_gbp

    p_break_even = float(np.mean(pbt >= -1e-9))
    # Financing can be repaid once trading receipts have recovered the outlay.
    p_payback_grant = float(np.mean(receipts >= upfront - 1e-9))
    p_upfront_within_grant = float(np.mean(upfront <= grant_cap_gbp))

    return {
        "p_break_even": p_break_even,
        "p_payback_grant": p_payback_grant,
        "p_upfront_within_grant": p_upfront_within_grant,
    }
