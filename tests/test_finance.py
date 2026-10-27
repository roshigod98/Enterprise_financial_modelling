from pathlib import Path

import pytest

from model.finance import ScenarioInputs, compute_outputs, distribute_profit_after_tax
from model.quotes import load_quotes_csv, pick_unit_cost
from model.risk import estimate_likelihoods
from model.scenarios import profit_band_over_price


def scenario(**overrides):
    inputs = dict(
        sell_price_input=5.0,
        sell_price_is_gross=False,
        unit_cost_gbp=2.0,
        order_size_units=100,
        units_sold=50,
        grant_gbp=1000.0,
    )
    inputs.update(overrides)
    return ScenarioInputs(**inputs)


def test_goodwill_loan_is_financing_not_profit():
    out = compute_outputs(scenario(units_sold=0))
    assert out.profit_before_tax_gbp == pytest.approx(0)
    assert out.profit_after_tax_gbp == pytest.approx(0)
    assert out.upfront_cash_gbp == pytest.approx(200)


def test_break_evens_are_different_and_do_not_double_count_inventory():
    out = compute_outputs(scenario(shipping_cost_gbp=90.0))
    assert out.breakeven_units == 30  # 90 / (5 - 2)
    assert out.upfront_cash_gbp == pytest.approx(290)
    assert out.cash_breakeven_units == 58  # 290 / 5, NOT 290 / 3
    assert out.inventory_left_units == 50


def test_illustrative_tax_and_effective_margin():
    out = compute_outputs(scenario(units_sold=100, sell_price_input=10.0))
    assert out.profit_before_tax_gbp == pytest.approx(800)
    assert out.tax_gbp == pytest.approx(160)
    assert out.profit_after_tax_gbp == pytest.approx(640)
    assert out.effective_net_margin_pct == pytest.approx(64)


def test_price_vat_conversion():
    out = compute_outputs(scenario(sell_price_input=12.0, sell_price_is_gross=True))
    assert out.sell_price_net == pytest.approx(10.0)


def test_profit_splitting_sums_to_total():
    out = distribute_profit_after_tax(1000.0, 0.30, 0.45, 0.11)
    assert out["charity"] == pytest.approx(300)
    assert out["ceo"] == pytest.approx(315)
    assert out["others"] == pytest.approx([77] * 5)
    assert out["charity"] + out["ceo"] + sum(out["others"]) == pytest.approx(1000)


def test_no_distributions_after_loss():
    out = distribute_profit_after_tax(-50, 0.30, 0.45, 0.11)
    assert out["charity"] == 0 and out["ceo"] == 0
    assert out["distributable"] == 0


def test_supplier_dataset_loads():
    csv = Path(__file__).resolve().parents[1] / "data" / "quotes_clean.csv"
    quotes = load_quotes_csv(str(csv))
    assert len(quotes) >= 10
    cost, _ = pick_unit_cost(quotes[0], 500)
    assert cost is not None and cost > 0


def test_range_band_handles_negative_unit_contribution():
    base = scenario(unit_cost_gbp=9, units_sold=5)
    prices, lo, hi = profit_band_over_price(base, 4, 5, 0, 20, 8, 10, n_price=5)
    assert len(prices) == 5
    assert all(a <= b for a, b in zip(lo, hi))


def test_risk_scenario_shares_use_sales_receipts_for_repayment():
    base = scenario()
    success = estimate_likelihoods(base, 5, 5, 2, 2, 50, 50, n=30, seed=42)
    assert success["p_break_even"] == pytest.approx(1)
    assert success["p_payback_grant"] == pytest.approx(1)  # 250 receipts >= 200 outlay
    assert success["p_upfront_within_grant"] == pytest.approx(1)
    failure = estimate_likelihoods(base, 5, 5, 2, 2, 0, 0, n=30, seed=42)
    assert failure["p_payback_grant"] == pytest.approx(0)
