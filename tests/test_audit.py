
import pytest

from model.demand import DemandCurve
from model.finance import ScenarioInputs, compute_outputs, distribute_profit_after_tax, net_price_from_input
from model.quotes import load_quotes_csv
from model.risk import estimate_likelihoods
from model.scenarios import profit_curve_with_demand, profit_heatmap_price_units


def base(**kw):
    values = dict(sell_price_input=5, sell_price_is_gross=False, unit_cost_gbp=2,
                  order_size_units=100, units_sold=50, grant_gbp=1000)
    values.update(kw)
    return ScenarioInputs(**values)


@pytest.mark.parametrize('field,value', [('units_sold', -1), ('units_sold', 1.5),
    ('unit_cost_gbp', -1), ('shipping_cost_gbp', float('nan')), ('grant_gbp', float('inf'))])
def test_invalid_finance_inputs(field, value):
    with pytest.raises(ValueError):
        compute_outputs(base(**{field: value}))


def test_overselling_requires_explicit_reorder():
    with pytest.raises(ValueError, match='inventory'):
        compute_outputs(base(units_sold=101))


def test_reorder_buys_whole_batches_and_cash_is_not_profit():
    out = compute_outputs(base(units_sold=101, allow_reorder=True, shipping_cost_gbp=10))
    assert out.ordered_units == 200
    assert out.inventory_left_units == 99
    assert out.upfront_cash_gbp == 410
    assert out.profit_before_tax_gbp == 293


def test_distribution_is_capped_by_post_tax_cash_after_recovering_capital():
    # Accounting PAT 120; receipts 250 - inventory 200 - tax 30 leaves only 20.
    out = compute_outputs(base())
    assert out.distribution_capacity_gbp == 20
    assert out.loan_repayment_feasible
    tight = compute_outputs(base(units_sold=42))
    assert tight.revenue_gbp >= tight.upfront_cash_gbp
    assert not tight.loan_repayment_feasible  # illustrative tax uses remaining cash
    assert tight.distribution_capacity_gbp == 0


@pytest.mark.parametrize('price,cost,fixed,expected', [(2,2,0,0), (1,2,0,0), (1,2,1,None), (2,2,1,None)])
def test_break_even_zero_negative_contribution(price,cost,fixed,expected):
    out = compute_outputs(base(sell_price_input=price, unit_cost_gbp=cost, shipping_cost_gbp=fixed))
    assert out.breakeven_units == expected
    if price < cost:
        assert not out.cash_recovery_feasible


def test_zero_outlay_recovers_at_zero_sales():
    out = compute_outputs(base(unit_cost_gbp=0, sell_price_input=0, units_sold=0))
    assert out.cash_breakeven_units == 0
    assert out.inventory_left_units == 100


def test_demand_curve_validation_and_interpolation():
    d = DemandCurve(p1=5, u1=100, p2=10, u2=0, max_units=100)
    assert d.predict_units(7.5) == 50
    assert d.predict_units(0) == 100
    assert d.predict_units(20) == 0
    with pytest.raises(ValueError):
        DemandCurve(p1=5,p2=5,u1=10,u2=20)
    with pytest.raises(ValueError):
        DemandCurve(min_units=200,max_units=100)


def test_demand_and_heatmap_cannot_sell_unbought_stock():
    d = DemandCurve(u1=300,u2=200)
    _, units, _ = profit_curve_with_demand(base(), 5, 10, d)
    assert units.max() <= 100
    _, units, z = profit_heatmap_price_units(base(), 5, 5, 150, 150)
    assert units.tolist() == [100]
    assert z.shape[0] == 1


def test_risk_is_reproducible_and_uses_after_tax_cash():
    kwargs = dict(price_min=4,price_max=6,unit_cost_min=1,unit_cost_max=3,units_min=0,units_max=150,n=100,seed=7)
    assert estimate_likelihoods(base(), **kwargs) == estimate_likelihoods(base(), **kwargs)
    result = estimate_likelihoods(base(),5,5,2,2,42,42,n=10)
    assert result['p_cash_recovery'] == 1
    assert result['p_payback_grant'] == 0
    with pytest.raises(ValueError):
        estimate_likelihoods(base(),5,5,2,2,0,0,n=0)


def test_supplier_alignment_preserves_real_costs():
    quotes = {q.supplier: q for q in load_quotes_csv('data/quotes_clean.csv')}
    q = quotes['Ningbo Yinzhou Chaoyang Packing Colorful Printing Factory']
    assert q.unit_cost_500 == 1.01
    assert q.total_cost_500 == 505
    assert q.shipping_cost is None
    assert q.cards_per_deck == 54
    q = quotes['Hangzhou Wonder Tree Technology Co., Ltd.']
    assert q.shipping_cost == 160
    assert q.vat_included is None


@pytest.mark.parametrize('value', ['bad', '-1', 'inf'])
def test_malformed_supplier_number_is_rejected(tmp_path, value):
    p = tmp_path/'q.csv'
    p.write_text(f'Supplier,UnitCost500GBP\nTest,{value}\n')
    with pytest.raises(ValueError):
        load_quotes_csv(str(p))


def test_missing_supplier_fields_remain_unknown(tmp_path):
    p = tmp_path/'q.csv'
    p.write_text('Supplier,UnitCost500GBP\nTest,1\n')
    q = load_quotes_csv(str(p))[0]
    assert q.shipping_cost is None and q.moq is None and q.vat_included is None


@pytest.mark.parametrize('ceo,other', [(-.1,.22), (1.5,-.1)])
def test_negative_distribution_shares_are_rejected(ceo,other):
    with pytest.raises(ValueError):
        distribute_profit_after_tax(100,.3,ceo,other)


def test_vat_and_loss_tax_edges():
    assert net_price_from_input(12,True,.2) == 10
    assert net_price_from_input(12,False,.2) == 12
    assert compute_outputs(base(sell_price_input=0)).tax_gbp == 0
    with pytest.raises(ValueError):
        net_price_from_input(-1,True,.2)


def test_manual_extrema_include_negative_margin_volume_corners():
    from model.scenarios import scenario_extrema
    lo, hi = scenario_extrema(base(), 1, 2, 0, 100, 3, 4)
    assert lo.profit_after_tax_gbp == -300
    assert hi.profit_after_tax_gbp == 0


def test_break_even_does_not_add_a_deck_for_binary_rounding():
    out = compute_outputs(base(sell_price_input=.3, unit_cost_gbp=.2, shipping_cost_gbp=.3))
    assert out.breakeven_units == 3


def test_risk_reorders_match_deterministic_finance():
    b = base(units_sold=101, allow_reorder=True, unit_cost_gbp=4)
    out = compute_outputs(b)
    result = estimate_likelihoods(b,5,5,4,4,101,101,n=10)
    assert result['p_payback_grant'] == float(out.loan_repayment_feasible)
    assert result['p_cash_recovery'] == float(out.revenue_gbp >= out.upfront_cash_gbp)


def test_loan_amount_cannot_change_earned_profit_or_distribution_capacity():
    with_loan = compute_outputs(base(grant_gbp=1000))
    without_loan = compute_outputs(base(grant_gbp=0))
    assert with_loan.profit_after_tax_gbp == without_loan.profit_after_tax_gbp
    assert with_loan.distribution_capacity_gbp == without_loan.distribution_capacity_gbp
    assert without_loan.funding_gap_gbp == 200
    assert with_loan.funding_gap_gbp == 0


def test_cash_limited_allocation_conserves_pool_and_handles_all_charity():
    out = compute_outputs(base())
    dist = distribute_profit_after_tax(out.distribution_capacity_gbp,.3,.45,.11)
    assert dist['charity'] + dist['ceo'] + sum(dist['others']) == pytest.approx(out.distribution_capacity_gbp)
    dist = distribute_profit_after_tax(out.distribution_capacity_gbp,1,.45,.11)
    assert dist['charity'] == out.distribution_capacity_gbp
    assert dist['distributable'] == 0
