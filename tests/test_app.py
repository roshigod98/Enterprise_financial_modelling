"""Execute real Streamlit reruns, including charts in every tab."""
from streamlit.testing.v1 import AppTest


def test_dashboard_default_and_zero_sales_launch():
    app = AppTest.from_file('app/main.py', default_timeout=30).run()
    assert not app.exception
    sales = next(w for w in app.number_input if w.label == 'Projected Sales (Units)')
    sales.set_value(0).run()
    assert not app.exception


def test_ranges_demand_reorders_and_incomplete_quote():
    app = AppTest.from_file('app/main.py', default_timeout=30).run()
    next(w for w in app.radio if w.label == 'Selling price mode').set_value('Range').run()
    next(w for w in app.radio if w.label == 'Units sold mode').set_value('Range').run()
    assert not app.exception
    next(w for w in app.checkbox if w.label == 'Allow reorders if demand exceeds order size').check().run()
    next(w for w in app.number_input if w.label == 'Max units sold').set_value(1200).run()
    assert not app.exception
    next(w for w in app.radio if w.label == 'Units input mode').set_value('Demand curve').run()
    assert not app.exception
    next(w for w in app.number_input if w.label == 'Price point 2 (£)').set_value(7).run()
    assert not app.exception
    assert any('anchor prices' in e.value for e in app.error)
    next(w for w in app.selectbox if w.label == 'Supplier').set_value('Ningbo Yinzhou Chaoyang Packing Colorful Printing Factory').run()
    assert not app.exception
    assert any('Shipping is unknown' in w.value for w in app.warning)


def test_custom_supplier_minimum_order_is_enforced():
    app = AppTest.from_file('app/main.py', default_timeout=30).run()
    next(w for w in app.selectbox if w.label == 'Supplier').set_value('Custom (manual entry)').run()
    next(w for w in app.number_input if w.label == 'Order size (units)').set_value(1).run()
    assert not app.exception
    assert any('minimum' in e.value for e in app.error)
