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
) -> dict[str, float]:
    """
    Monte Carlo likelihood estimates based on uniformly sampling within ranges.

    - If demand_curve is provided, units are derived from price via demand_curve.predict_units(price).
      Optional multiplicative noise can be applied via demand_noise_pct (e.g. 0.10 = ±10%).
    - Break-even is defined as profit_after_tax >= 0.
    - Grant payback is defined as profit_after_tax >= grant_cap_gbp (before charity/team payout).
    - Upfront within grant is defined as upfront_cash_gbp <= grant_cap_gbp.
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

    max_units = max(0, int(getattr(base, "order_size_units", 0)), int(max(units_min, units_max)))
    u_samples = np.clip(u_samples, 0, max_units)

    pat = np.zeros(n, dtype=float)
    upfront = np.zeros(n, dtype=float)

    for i in range(n):
        xi = replace(
            base,
            sell_price_input=float(p_samples[i]),
            unit_cost_gbp=float(c_samples[i]),
            units_sold=int(u_samples[i]),
        )
        out = compute_outputs(xi)
        pat[i] = out.profit_after_tax_gbp
        upfront[i] = out.upfront_cash_gbp

    p_break_even = float(np.mean(pat >= 0.0))
    p_payback_grant = float(np.mean(pat >= grant_cap_gbp))
    p_upfront_within_grant = float(np.mean(upfront <= grant_cap_gbp))

    return {
        "p_break_even": p_break_even,
        "p_payback_grant": p_payback_grant,
        "p_upfront_within_grant": p_upfront_within_grant,
    }
