import sys
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path when running via `python app/main.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.demand import DemandCurve
from model.finance import ScenarioInputs, compute_outputs, distribute_profit_after_tax
from model.risk import estimate_likelihoods
from model.quotes import Quote, load_quotes_csv, pick_unit_cost
from model.scenarios import (
    profit_band_over_price,
    profit_curve_with_demand,
    profit_heatmap_price_units,
    sweep_profit_over_price,
)


def fmt_money(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def narrative_summary(
    q,
    x: ScenarioInputs,
    out,
    out_worst,
    out_best,
    dist,
    sell_price_min: float,
    sell_price_max: float,
    units_min: int,
    units_max: int,
    unit_cost_min: float,
    unit_cost_max: float,
    charity_pct: float,
):
    bullets: list[str] = []

    bullets.append(
        f"Using **{q.supplier}** with an order of **{x.order_size_units} decks**, unit cost is about "
        f"**{fmt_money(x.unit_cost_gbp, 2)}** (range {fmt_money(unit_cost_min, 2)}–{fmt_money(unit_cost_max, 2)})."
    )

    price_label = "customer price (inc VAT)" if x.sell_price_is_gross else "net price (ex VAT)"
    bullets.append(
        f"Selling price is **{fmt_money(x.sell_price_input, 2)}** ({price_label}); net price is "
        f"**{fmt_money(out.sell_price_net, 2)}**. Variable cost per unit is "
        f"**{fmt_money(out.variable_cost_per_unit_gbp, 2)}**, so contribution is "
        f"**{fmt_money(out.contribution_margin_gbp, 2)}** per deck."
    )

    trading_be = "not reachable" if out.breakeven_units is None else f"about {out.breakeven_units} units"
    cash_be = "not reachable" if out.cash_breakeven_units is None else f"about {out.cash_breakeven_units} units"
    bullets.append(
        f"Upfront cash required is **{fmt_money(out.upfront_cash_gbp)}**; trading break-even is "
        f"**{trading_be}** and cash break-even is **{cash_be}** (ignoring the grant)."
    )

    bullets.append(
        f"Profit after tax is **{fmt_money(out.profit_after_tax_gbp)}**; "
        f"charity **{fmt_money(dist['charity'])}**, CEO **{fmt_money(dist['ceo'])}**, "
        f"each team member **{fmt_money(dist['others'][0])}**."
    )

    if units_min == units_max and sell_price_min == sell_price_max:
        bullets.append(
            f"Single-point scenario at **{x.units_sold} units**: worst/base/best are identical."
        )
    else:
        bullets.append(
            f"Range outcomes (PAT) span **{fmt_money(out_worst.profit_after_tax_gbp)}** to "
            f"**{fmt_money(out_best.profit_after_tax_gbp)}** across your ranges."
        )

    price_mid = (sell_price_min + sell_price_max) / 2.0
    units_mid = (units_min + units_max) / 2.0
    cost_mid = (unit_cost_min + unit_cost_max) / 2.0

    price_span = sell_price_max - sell_price_min
    units_span = units_max - units_min
    cost_span = unit_cost_max - unit_cost_min

    floor_note = ""
    if x.units_sold > 0:
        floor_net = (out.fixed_costs_gbp / x.units_sold) + out.variable_cost_per_unit_gbp
        if x.sell_price_is_gross:
            floor_display = floor_net * (1.0 + x.vat_rate)
            floor_note = f"Pricing floor at current volume is **{fmt_money(floor_display, 2)}** (gross)."
        else:
            floor_note = f"Pricing floor at current volume is **{fmt_money(floor_net, 2)}** (net)."

    if price_span == 0 and units_span == 0 and cost_span == 0:
        driver_note = "All ranges are fixed, so sensitivity is driven by the single-point assumptions."
    else:
        rel_price = price_span / max(price_mid, 1e-6)
        rel_units = units_span / max(units_mid, 1.0)
        rel_cost = cost_span / max(cost_mid, 1e-6)
        driver = max(
            [(rel_price, "selling price"), (rel_units, "units sold"), (rel_cost, "unit cost")],
            key=lambda x: x[0],
        )[1]
        driver_note = f"Biggest sensitivity driver (by input range) is **{driver}**."

    bullets.append(f"{driver_note} {floor_note}".strip())

    return bullets


def effective_order_units(units: int, base_order: int, allow_reorder: bool) -> int:
    if base_order <= 0:
        return 0
    if not allow_reorder:
        return base_order
    if units <= base_order:
        return base_order
    batches = ceil(units / base_order)
    return base_order * batches


GRANT_CAP = 1000.0
# --- Page Config ---
st.set_page_config(page_title="Enterprise Financial Modeller", layout="wide")

# --- Header row (left title, right badge) ---
left, right = st.columns([7, 3])
with left:
    st.title("Oundle Branded Playing Cards — Financial Modeller")
with right:
    st.markdown(
        """
        <div style="text-align:right; font-size:12px; line-height:1.25; padding-top:6px;">
          <b>Made by Roshan Narayan</b><br/>
          Oundle School • 2026<br/>
          Peter Jones Enterprise Tycoon
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Load Data & Sidebar Inputs ---
quotes = load_quotes_csv("data/quotes_clean.csv")
supplier_names = [q.supplier for q in quotes]

with st.sidebar:
    st.title("Inputs")

    with st.expander("Supplier & Production", expanded=True):
        supplier_options = ["Custom (manual entry)"] + supplier_names
        default_index = 1 if supplier_names else 0
        supplier = st.selectbox("Supplier", supplier_options, index=default_index)

        if supplier == "Custom (manual entry)":
            st.caption("Enter your own supplier assumptions (not saved to CSV yet).")
            custom_name = st.text_input("Custom supplier name", value="Custom Supplier")
            custom_moq = st.number_input("MOQ", min_value=1, value=500, step=50)
            order_size = st.number_input(
                "Order size (units)", min_value=1, value=int(custom_moq), step=50
            )
            order_size_int = int(order_size)
            custom_unit_cost = st.number_input(
                "Unit cost (£) (for this order size)", min_value=0.0, value=0.80, step=0.01
            )
            custom_shipping = st.number_input(
                "Shipping / Fixed Costs (£)", min_value=0.0, value=0.0, step=1.0
            )
            custom_lead = st.text_input("Lead time", value="")
            custom_notes = st.text_area("Notes", value="")

            q = Quote(
                supplier=custom_name.strip() or "Custom Supplier",
                region=None,
                moq=int(custom_moq),
                unit_cost_250=None,
                unit_cost_500=None,
                total_cost_500=None,
                shipping_cost=float(custom_shipping),
                vat_included=None,
                card_stock=None,
                finish=None,
                tuck_box=None,
                card_size_mm=None,
                cards_per_deck=None,
                tuck_box_finish=None,
                shrink_wrap=None,
                lead_time=custom_lead.strip() or None,
                payment_terms=None,
                notes=custom_notes.strip() or None,
            )

            unit_cost = float(custom_unit_cost)
            unit_cost_note = "custom"
            shipping = float(custom_shipping)
        else:
            q = next(x for x in quotes if x.supplier == supplier)

            order_size = st.number_input("Order size (units)", min_value=1, value=500, step=50)
            order_size_int = int(order_size)

            unit_cost, unit_cost_note = pick_unit_cost(q, order_size_int)
            if unit_cost is None:
                st.error("Missing unit cost data for this row.")
                st.stop()

            shipping = q.shipping_cost or 0.0
            shipping = st.number_input(
                "Shipping / Fixed Costs (£)", min_value=0.0, value=float(shipping), step=1.0
            )

        st.caption(f"Base Unit Cost: £{unit_cost:.2f} ({unit_cost_note or 'exact'})")

        override_cost_flag = st.checkbox("Override Unit Cost")
        if override_cost_flag:
            unit_cost_mode = st.radio("Unit cost mode", ["Fixed", "Range"], horizontal=True)
            if unit_cost_mode == "Fixed":
                unit_cost = st.number_input(
                    "Purchase unit cost (£)",
                    min_value=0.0,
                    value=float(unit_cost),
                    step=0.01,
                )
                unit_cost_min = float(unit_cost)
                unit_cost_max = float(unit_cost)
            else:
                c_uc_1, c_uc_2 = st.columns(2)
                with c_uc_1:
                    unit_cost_min = st.number_input(
                        "Min unit cost (£)",
                        min_value=0.0,
                        value=float(unit_cost),
                        step=0.01,
                    )
                with c_uc_2:
                    unit_cost_max = st.number_input(
                        "Max unit cost (£)",
                        min_value=0.0,
                        value=float(unit_cost) + 0.25,
                        step=0.01,
                    )
                unit_cost = (float(unit_cost_min) + float(unit_cost_max)) / 2.0
        else:
            unit_cost_min = float(unit_cost)
            unit_cost_max = float(unit_cost)


    with st.expander("Sales Assumptions", expanded=True):
        demand_curve = None
        price_mode = st.radio("Selling price mode", ["Fixed", "Range"], horizontal=True)
        sell_is_gross = st.selectbox("Price Type", ["Customer (inc VAT)", "Net (ex VAT)"]) == "Customer (inc VAT)"

        if price_mode == "Fixed":
            sell_price = st.number_input("Selling Price per Deck (£)", min_value=0.0, value=8.0, step=0.5)
            sell_price_min = float(sell_price)
            sell_price_max = float(sell_price)
        else:
            c_price_1, c_price_2 = st.columns(2)
            with c_price_1:
                sell_price_min = st.number_input("Min selling price (£)", min_value=0.0, value=6.0, step=0.5)
            with c_price_2:
                sell_price_max = st.number_input("Max selling price (£)", min_value=0.0, value=10.0, step=0.5)
            sell_price = (float(sell_price_min) + float(sell_price_max)) / 2.0

        units_input_mode = st.radio(
            "Units input mode", ["Manual (Fixed/Range)", "Demand curve"], horizontal=True
        )
        allow_reorder = st.checkbox("Allow reorders if demand exceeds order size", value=False)

        if units_input_mode == "Manual (Fixed/Range)":
            units_mode = st.radio("Units sold mode", ["Fixed", "Range"], horizontal=True)
            default_units_sold = min(125, order_size_int)
            units_limit = order_size_int if not allow_reorder else max(order_size_int * 5, order_size_int)

            if units_mode == "Fixed":
                units_sold = st.number_input(
                    "Projected Sales (Units)",
                    min_value=0,
                    max_value=units_limit,
                    value=default_units_sold,
                    step=1,
                )
                units_min = int(units_sold)
                units_max = int(units_sold)
            else:
                default_units_min = min(125, units_limit)
                default_units_max = min(250, units_limit)
                if default_units_max < default_units_min:
                    default_units_max = default_units_min
                c_units_1, c_units_2 = st.columns(2)
                with c_units_1:
                    units_min = st.number_input(
                        "Min units sold",
                        min_value=0,
                        max_value=units_limit,
                        value=default_units_min,
                        step=10,
                    )
                with c_units_2:
                    units_max = st.number_input(
                        "Max units sold",
                        min_value=0,
                        max_value=units_limit,
                        value=default_units_max,
                        step=10,
                    )
                units_sold = int((int(units_min) + int(units_max)) / 2)
        else:
            st.caption("Define two anchor points and we linearly interpolate units vs price.")
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                demand_p1 = st.number_input("Price point 1 (£)", min_value=0.0, value=7.0, step=0.5)
                demand_u1 = st.number_input("Units at price 1", min_value=0, value=300, step=10)
            with c_d2:
                demand_p2 = st.number_input("Price point 2 (£)", min_value=0.0, value=10.0, step=0.5)
                demand_u2 = st.number_input("Units at price 2", min_value=0, value=200, step=10)

            clamp_max_default = order_size_int if not allow_reorder else max(order_size_int * 3, order_size_int)
            demand_min_units = st.number_input("Clamp min units", min_value=0, value=0, step=10)
            demand_max_units = st.number_input(
                "Clamp max units", min_value=1, value=int(clamp_max_default), step=50
            )

            demand_curve = DemandCurve(
                p1=float(demand_p1),
                u1=int(demand_u1),
                p2=float(demand_p2),
                u2=int(demand_u2),
                min_units=int(demand_min_units),
                max_units=int(demand_max_units),
            )

            units_sold = int(demand_curve.predict_units(float(sell_price)))
            u_at_min = demand_curve.predict_units(float(sell_price_min))
            u_at_max = demand_curve.predict_units(float(sell_price_max))
            units_min, units_max = sorted([int(u_at_min), int(u_at_max)])

            demand_prices = np.linspace(min(sell_price_min, sell_price_max), max(sell_price_min, sell_price_max), 30)
            demand_units = [demand_curve.predict_units(float(p)) for p in demand_prices]
            fig_demand = go.Figure()
            fig_demand.add_trace(
                go.Scatter(x=demand_prices, y=demand_units, mode="lines", name="Demand curve")
            )
            fig_demand.update_layout(height=200, xaxis_title="Price (£)", yaxis_title="Units")
            st.plotly_chart(fig_demand, use_container_width=True)

    with st.expander("Finance & Distribution", expanded=False):
        grant = st.number_input("Grant Received (£)", min_value=0.0, value=1000.0, step=50.0)
        vat_rate = 0.20

        with st.expander("Extra costs (optional)", expanded=False):
            extra_fixed = st.number_input("Extra fixed costs (£)", min_value=0.0, value=0.0, step=10.0)
            extra_var_unit = st.number_input(
                "Extra variable cost per deck (£)", min_value=0.0, value=0.0, step=0.01
            )

        st.subheader("Distributions")
        charity_pct = st.slider("Charity Share (%)", 0, 100, 30) / 100.0
        ceo_pct = st.slider("CEO Share (of remaining) (%)", 0, 100, 45) / 100.0
        other_each_pct = (1.0 - ceo_pct) / 5.0
        st.caption(f"Team Members (x5): {other_each_pct*100:.1f}% each")

# Normalize min/max to avoid inverted ranges.
sell_price_min, sell_price_max = sorted([float(sell_price_min), float(sell_price_max)])
unit_cost_min, unit_cost_max = sorted([float(unit_cost_min), float(unit_cost_max)])

sell_price = (sell_price_min + sell_price_max) / 2.0
unit_cost = (unit_cost_min + unit_cost_max) / 2.0

if units_input_mode == "Demand curve" and demand_curve is not None:
    units_sold = int(demand_curve.predict_units(float(sell_price)))
    u_at_min = demand_curve.predict_units(float(sell_price_min))
    u_at_max = demand_curve.predict_units(float(sell_price_max))
    units_min, units_max = sorted([int(u_at_min), int(u_at_max)])
else:
    units_min, units_max = sorted([int(units_min), int(units_max)])
    units_sold = int((units_min + units_max) / 2)

if not allow_reorder:
    units_sold = min(units_sold, order_size_int)
    units_min = min(units_min, order_size_int)
    units_max = min(units_max, order_size_int)

order_size_units_base = effective_order_units(units_sold, order_size_int, allow_reorder)
order_size_units_worst = effective_order_units(units_min, order_size_int, allow_reorder)
order_size_units_best = effective_order_units(units_max, order_size_int, allow_reorder)

# --- Computations ---
x = ScenarioInputs(
    vat_rate=vat_rate,
    grant_gbp=grant,
    sell_price_input=float(sell_price),
    sell_price_is_gross=sell_is_gross,
    units_sold=int(units_sold),
    order_size_units=order_size_units_base,
    unit_cost_gbp=float(unit_cost),
    shipping_cost_gbp=float(shipping),
    setup_cost_gbp=0.0,
    extra_fixed_costs_gbp=float(extra_fixed),
    extra_variable_cost_per_unit_gbp=float(extra_var_unit),
)

out = compute_outputs(x)
dist = distribute_profit_after_tax(out.profit_after_tax_gbp, charity_pct, ceo_pct, other_each_pct)

worst = ScenarioInputs(
    vat_rate=vat_rate,
    grant_gbp=grant,
    sell_price_input=float(sell_price_min),
    sell_price_is_gross=sell_is_gross,
    units_sold=int(units_min),
    order_size_units=order_size_units_worst,
    unit_cost_gbp=float(unit_cost_max),
    shipping_cost_gbp=float(shipping),
    setup_cost_gbp=0.0,
    extra_fixed_costs_gbp=float(extra_fixed),
    extra_variable_cost_per_unit_gbp=float(extra_var_unit),
)
best = ScenarioInputs(
    vat_rate=vat_rate,
    grant_gbp=grant,
    sell_price_input=float(sell_price_max),
    sell_price_is_gross=sell_is_gross,
    units_sold=int(units_max),
    order_size_units=order_size_units_best,
    unit_cost_gbp=float(unit_cost_min),
    shipping_cost_gbp=float(shipping),
    setup_cost_gbp=0.0,
    extra_fixed_costs_gbp=float(extra_fixed),
    extra_variable_cost_per_unit_gbp=float(extra_var_unit),
)

out_worst = compute_outputs(worst)
out_best = compute_outputs(best)

# --- Tabs ---
tab_overview, tab_range, tab_supplier, tab_distributions, tab_info = st.tabs(
    ["Overview", "Explore Ranges", "Supplier Compare", "Distributions", "Info"]
)

# --- TAB 1: OVERVIEW ---
with tab_overview:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Profit after tax", fmt_money(out.profit_after_tax_gbp))
    k2.metric(
        "Profit / unit (PAT)",
        "-" if out.profit_per_unit_after_tax is None else fmt_money(out.profit_per_unit_after_tax, 2),
    )
    k3.metric("Upfront cash needed", fmt_money(out.upfront_cash_gbp))
    k4.metric(
        "Cash break-even units",
        "-" if out.cash_breakeven_units is None else str(out.cash_breakeven_units),
    )
    k5.metric("Effective net margin", fmt_pct(out.effective_net_margin_pct))
    k6.metric(
        "Inventory left",
        "-" if out.inventory_left_units is None else str(out.inventory_left_units),
    )

    st.subheader("Grant cap check")
    upfront_cash = out.upfront_cash_gbp
    ratio = upfront_cash / GRANT_CAP if GRANT_CAP > 0 else 0.0
    st.progress(min(1.0, max(0.0, ratio)))
    if upfront_cash <= GRANT_CAP:
        st.success(
            f"Upfront cash required is {fmt_money(upfront_cash)} — within the £{int(GRANT_CAP)} grant cap."
        )
    else:
        gap = upfront_cash - GRANT_CAP
        st.warning(
            f"Upfront cash required is {fmt_money(upfront_cash)} — exceeds the £{int(GRANT_CAP)} grant cap "
            f"by {fmt_money(gap)}."
        )

    st.subheader("Likelihood (based on your ranges)")
    demand_mode = units_input_mode == "Demand curve"
    demand_noise = 0.0
    if demand_mode:
        demand_noise = st.slider("Demand uncertainty (±%)", 0, 30, 0) / 100.0

    lik = estimate_likelihoods(
        base=x,
        price_min=sell_price_min,
        price_max=sell_price_max,
        unit_cost_min=unit_cost_min,
        unit_cost_max=unit_cost_max,
        units_min=units_min,
        units_max=units_max,
        grant_cap_gbp=GRANT_CAP,
        n=2000,
        seed=42,
        demand_curve=demand_curve if demand_mode else None,
        demand_noise_pct=demand_noise,
    )

    l1, l2, l3 = st.columns(3)
    l1.metric("Chance of break-even (PAT ≥ £0)", f"{lik['p_break_even']*100:.0f}%")
    l2.metric("Chance of paying back grant (PAT ≥ £1000)", f"{lik['p_payback_grant']*100:.0f}%")
    l3.metric("Chance upfront cash within £1000", f"{lik['p_upfront_within_grant']*100:.0f}%")
    st.caption(
        "These are simple Monte Carlo estimates based on uniformly sampling within your min/max assumptions "
        "(not a guarantee)."
    )

    st.markdown("### Narrative Summary (for presenting)")
    bullets = narrative_summary(
        q,
        x,
        out,
        out_worst,
        out_best,
        dist,
        sell_price_min,
        sell_price_max,
        units_min,
        units_max,
        unit_cost_min,
        unit_cost_max,
        charity_pct,
    )
    st.success("\n".join([f"• {b}" for b in bullets]))

    st.subheader("Waterfall: Where the money goes")
    measures = ["relative", "relative", "relative", "relative", "relative", "total"]
    xw = ["Revenue", "COGS", "Fixed costs", "Grant", "Tax", "Profit after tax"]
    yw = [
        out.revenue_gbp,
        -out.cogs_gbp,
        -out.fixed_costs_gbp,
        x.grant_gbp,
        -out.tax_gbp,
        out.profit_after_tax_gbp,
    ]
    fig_w = go.Figure(
        go.Waterfall(
            name="",
            measure=measures,
            x=xw,
            y=yw,
            connector={"line": {"dash": "dot"}},
        )
    )
    fig_w.update_layout(height=380, yaxis_title="£")
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Unit economics (per deck)")
    unit_df = pd.DataFrame(
        {
            "Component": ["Net selling price", "Variable cost", "Contribution"],
            "Value": [
                out.sell_price_net,
                out.variable_cost_per_unit_gbp,
                out.contribution_margin_gbp,
            ],
        }
    )
    fig_unit = px.bar(unit_df, x="Component", y="Value", text_auto=".2f", title="")
    fig_unit.update_layout(height=320, yaxis_title="£ per deck", xaxis_title="")
    st.plotly_chart(fig_unit, use_container_width=True)

    st.subheader("Upfront cash composition")
    batch_cost = out.variable_cost_per_unit_gbp * x.order_size_units
    fixed_costs = out.fixed_costs_gbp
    cash_df = pd.DataFrame(
        {"Component": ["Batch production cost", "Fixed costs"], "Value": [batch_cost, fixed_costs]}
    )
    fig_cash = px.pie(cash_df, names="Component", values="Value", hole=0.35)
    fig_cash.update_layout(height=320)
    st.plotly_chart(fig_cash, use_container_width=True)

    with st.expander("Range snapshot"):
        p1, p2, p3 = st.columns(3)
        p1.metric("Worst PAT (£)", f"{out_worst.profit_after_tax_gbp:,.0f}")
        p2.metric("Base PAT (£)", f"{out.profit_after_tax_gbp:,.0f}")
        p3.metric("Best PAT (£)", f"{out_best.profit_after_tax_gbp:,.0f}")

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Trading break-even units",
            "-" if out.breakeven_units is None else str(out.breakeven_units),
        )
        r2.metric(
            "Profit / unit (mid)",
            "-" if out.profit_per_unit_after_tax is None else f"{out.profit_per_unit_after_tax:,.2f}",
        )
        r3.metric(
            "Profit range",
            f"{out_worst.profit_after_tax_gbp:,.0f} to {out_best.profit_after_tax_gbp:,.0f}",
        )

    with st.expander("Worst / Base / Best summary"):
        summary = pd.DataFrame(
            [
                {
                    "Scenario": "Worst",
                    "Revenue (net)": out_worst.revenue_gbp,
                    "PAT": out_worst.profit_after_tax_gbp,
                    "Upfront cash": out_worst.upfront_cash_gbp,
                    "Cash BE units": out_worst.cash_breakeven_units,
                    "Inventory left": out_worst.inventory_left_units,
                },
                {
                    "Scenario": "Base",
                    "Revenue (net)": out.revenue_gbp,
                    "PAT": out.profit_after_tax_gbp,
                    "Upfront cash": out.upfront_cash_gbp,
                    "Cash BE units": out.cash_breakeven_units,
                    "Inventory left": out.inventory_left_units,
                },
                {
                    "Scenario": "Best",
                    "Revenue (net)": out_best.revenue_gbp,
                    "PAT": out_best.profit_after_tax_gbp,
                    "Upfront cash": out_best.upfront_cash_gbp,
                    "Cash BE units": out_best.cash_breakeven_units,
                    "Inventory left": out_best.inventory_left_units,
                },
            ]
        )
        st.dataframe(summary, use_container_width=True)

    with st.expander("Detailed metrics"):
        st.markdown(
            f"""
        **Revenue (net):** {fmt_money(out.revenue_gbp, 2)}  
        **COGS:** {fmt_money(out.cogs_gbp, 2)}  
        **Fixed costs:** {fmt_money(out.fixed_costs_gbp, 2)}  
        **Grant:** {fmt_money(x.grant_gbp, 2)}  
        **Profit before tax:** {fmt_money(out.profit_before_tax_gbp, 2)}  
        **Tax (20%):** {fmt_money(out.tax_gbp, 2)}  
        **Profit after tax:** {fmt_money(out.profit_after_tax_gbp, 2)}  

        ---  
        **Net price:** {fmt_money(out.sell_price_net, 2)}  
        **Variable cost / unit:** {fmt_money(out.variable_cost_per_unit_gbp, 2)}  
        **Contribution / unit:** {fmt_money(out.contribution_margin_gbp, 2)}  
        **Contribution margin:** {fmt_pct(out.contribution_margin_pct)}
        """
        )

    st.divider()

    col_chart, col_details = st.columns([2, 1])

    with col_chart:
        st.subheader("Trading Profit vs Units (ignoring grant)")

        max_u = max(order_size_int, units_max, (out.cash_breakeven_units or 0) * 2, 50)
        u_grid = np.unique(np.round(np.linspace(0, max_u, 140)).astype(int))
        u_grid = u_grid[u_grid >= 0]

        profit_curve = []
        for u in u_grid:
            order_units_here = effective_order_units(int(u), order_size_int, allow_reorder)
            xi = ScenarioInputs(
                vat_rate=vat_rate,
                grant_gbp=0.0,
                sell_price_input=float(sell_price),
                sell_price_is_gross=sell_is_gross,
                units_sold=int(u),
                order_size_units=int(order_units_here),
                unit_cost_gbp=float(unit_cost),
                shipping_cost_gbp=float(shipping),
                setup_cost_gbp=0.0,
                extra_fixed_costs_gbp=float(extra_fixed),
                extra_variable_cost_per_unit_gbp=float(extra_var_unit),
            )
            oi = compute_outputs(xi)
            profit_curve.append(oi.profit_after_tax_gbp)

        fig_be = go.Figure()
        fig_be.add_trace(
            go.Scatter(x=u_grid, y=profit_curve, mode="lines", name="Trading profit after tax (grant=0)")
        )
        fig_be.add_hline(y=0, line=dict(color="#444", width=1))

        if out.breakeven_units is not None:
            fig_be.add_vline(x=out.breakeven_units, line=dict(color="#1f77b4", dash="dash"))
            fig_be.add_annotation(
                x=out.breakeven_units,
                y=0,
                text="Trading BE",
                showarrow=True,
                arrowhead=2,
                yshift=18,
            )

        if out.cash_breakeven_units is not None:
            fig_be.add_vline(x=out.cash_breakeven_units, line=dict(color="#d62728", dash="dot"))
            fig_be.add_annotation(
                x=out.cash_breakeven_units,
                y=0,
                text="Cash BE",
                showarrow=True,
                arrowhead=2,
                yshift=-28,
            )

        fig_be.update_layout(
            height=440,
            xaxis_title="Units sold",
            yaxis_title="Trading profit after tax (£)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_be, use_container_width=True)
        st.caption("This chart sets grant=£0 so break-even reflects trading performance only.")

    with col_details:
        st.subheader("Assumptions (midpoint)")
        price_label = "Price (inc VAT)" if sell_is_gross else "Price (ex VAT)"
        st.markdown(
            f"""
        **{price_label}:** {fmt_money(x.sell_price_input, 2)}  
        **Net price:** {fmt_money(out.sell_price_net, 2)}  
        **Units sold:** {x.units_sold}  
        **Order size:** {x.order_size_units}  
        **Unit cost:** {fmt_money(x.unit_cost_gbp, 2)}  
        **Extra variable / unit:** {fmt_money(x.extra_variable_cost_per_unit_gbp, 2)}  
        **Extra fixed costs:** {fmt_money(x.extra_fixed_costs_gbp, 2)}  
        **Grant:** {fmt_money(x.grant_gbp, 2)}  
        """
        )

        with st.expander("Selected supplier details"):
            st.write(
                {
                    "Supplier": q.supplier,
                    "MOQ": q.moq,
                    "Card stock": q.card_stock,
                    "Finish": q.finish,
                    "Tuck box": q.tuck_box,
                    "Lead time": q.lead_time,
                    "Payment terms": q.payment_terms,
                    "Notes": q.notes,
                }
            )

# --- TAB 2: EXPLORE RANGES ---
with tab_range:
    if units_input_mode == "Demand curve" and demand_curve is not None:
        st.subheader("Likely profit vs price (demand curve)")
        prices, units_pred, profits = profit_curve_with_demand(
            base=x,
            price_min=float(sell_price_min),
            price_max=float(sell_price_max),
            demand=demand_curve,
            n_price=120,
        )

        fig_profit = go.Figure()
        if unit_cost_min != unit_cost_max:
            base_low = replace(x, unit_cost_gbp=float(unit_cost_min))
            base_high = replace(x, unit_cost_gbp=float(unit_cost_max))
            _, _, profits_low = profit_curve_with_demand(
                base=base_low,
                price_min=float(sell_price_min),
                price_max=float(sell_price_max),
                demand=demand_curve,
                n_price=120,
            )
            _, _, profits_high = profit_curve_with_demand(
                base=base_high,
                price_min=float(sell_price_min),
                price_max=float(sell_price_max),
                demand=demand_curve,
                n_price=120,
            )
            lo = np.minimum(profits_low, profits_high)
            hi = np.maximum(profits_low, profits_high)
            fig_profit.add_trace(
                go.Scatter(x=prices, y=hi, mode="lines", line=dict(width=0), showlegend=False)
            )
            fig_profit.add_trace(
                go.Scatter(
                    x=prices,
                    y=lo,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(30, 144, 255, 0.20)",
                    line=dict(width=0),
                    name="Profit range (cost uncertainty)",
                )
            )

        fig_profit.add_trace(
            go.Scatter(
                x=prices,
                y=profits,
                mode="lines",
                name="Likely profit (PAT)",
                line=dict(width=3, color="#1f77b4"),
            )
        )
        fig_profit.add_hline(y=0, line=dict(color="#444", width=1))
        fig_profit.update_layout(
            height=420, xaxis_title="Selling price (£)", yaxis_title="Profit after tax (£)"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

        st.subheader("Predicted units vs price")
        fig_units = go.Figure()
        fig_units.add_trace(
            go.Scatter(x=prices, y=units_pred, mode="lines", name="Predicted units")
        )
        fig_units.update_layout(height=320, xaxis_title="Selling price (£)", yaxis_title="Units")
        st.plotly_chart(fig_units, use_container_width=True)
    else:
        st.subheader("Profit across selling price range")

        price_collapsed = sell_price_min == sell_price_max
        units_collapsed = units_min == units_max

        if price_collapsed:
            st.warning(
                "Selling price range is a single value — switch to Range mode to see the band and landscape."
            )
            max_u = max(order_size_int, units_max, 50)
            units_curve = np.unique(np.round(np.linspace(0, max_u, 80)).astype(int))
            units_curve = units_curve[units_curve >= 0]
            profits = []
            for u in units_curve:
                xi = replace(x, sell_price_input=float(sell_price_min), units_sold=int(u))
                profits.append(compute_outputs(xi).profit_after_tax_gbp)
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(x=units_curve, y=profits, mode="lines", name="Profit (PAT)")
            )
            fig_line.add_hline(y=0, line=dict(color="#444", width=1))
            fig_line.update_layout(
                height=420, xaxis_title="Units sold", yaxis_title="Profit after tax (£)"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            if units_collapsed:
                st.warning("Units range is a single value — the band collapses to a line.")

            prices, p_lo, p_hi = profit_band_over_price(
                base=x,
                price_min=float(sell_price_min),
                price_max=float(sell_price_max),
                units_min=int(units_min),
                units_max=int(units_max),
                unit_cost_min=float(unit_cost_min),
                unit_cost_max=float(unit_cost_max),
                n_price=140,
            )

            fig_band = go.Figure()
            fig_band.add_trace(
                go.Scatter(
                    x=prices,
                    y=p_hi,
                    mode="lines",
                    name="Best case (high units, low cost)",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=prices,
                    y=p_lo,
                    mode="lines",
                    name="Profit range (low units & high cost → high units & low cost)",
                    fill="tonexty",
                    fillcolor="rgba(0, 200, 255, 0.25)",
                    line=dict(width=0),
                )
            )

            mid_prices, mid_profit = sweep_profit_over_price(
                x, float(sell_price_min), float(sell_price_max), n=140, units_override=int(units_sold)
            )
            fig_band.add_trace(
                go.Scatter(
                    x=mid_prices,
                    y=mid_profit,
                    mode="lines",
                    name="Mid scenario",
                    line=dict(width=3, color="#1f77b4"),
                )
            )
            fig_band.add_hline(y=0, line=dict(color="#444", width=1))
            fig_band.update_layout(
                height=420,
                xaxis_title="Selling price (£)",
                yaxis_title="Profit after tax (£)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_band, use_container_width=True)

        st.subheader("Profit landscape (price × units)")
        if price_collapsed or units_collapsed:
            st.warning(
                "Heatmap is hidden when price or units ranges are a single value."
            )
        else:
            prices_h, units_h, z = profit_heatmap_price_units(
                base=x,
                price_min=float(sell_price_min),
                price_max=float(sell_price_max),
                units_min=int(units_min),
                units_max=int(units_max),
                n_price=60,
                n_units=60,
            )

            fig_h = go.Figure(
                data=go.Heatmap(
                    x=prices_h,
                    y=units_h,
                    z=z,
                    colorbar=dict(title="Profit (£)"),
                    hovertemplate="Price £%{x:.2f}<br>Units %{y}<br>Profit £%{z:,.0f}<extra></extra>",
                )
            )
            fig_h.add_trace(
                go.Contour(
                    x=prices_h,
                    y=units_h,
                    z=z,
                    contours=dict(start=0, end=0, size=1, coloring="none"),
                    line=dict(width=3, color="#111"),
                    showscale=False,
                    hoverinfo="skip",
                    name="Break-even boundary",
                )
            )
            fig_h.update_layout(height=520, xaxis_title="Selling price (£)", yaxis_title="Units sold")
            st.plotly_chart(fig_h, use_container_width=True)

# --- TAB 3: SUPPLIER COMPARE ---
with tab_supplier:
    st.subheader("Supplier comparison (same selling assumptions)")

    rows: list[dict] = []
    for quote in quotes:
        supplier_cost, note = pick_unit_cost(quote, order_size_int)
        if supplier_cost is None:
            rows.append(
                {
                    "Supplier": quote.supplier,
                    "Unit Cost (£)": None,
                    "Shipping (£)": quote.shipping_cost or 0.0,
                    "Profit After Tax (£)": None,
                    "ROI on Upfront (%)": None,
                    "Break-even Units": None,
                    "Notes": "Missing unit cost",
                }
            )
            continue

        shipping_cost = quote.shipping_cost or 0.0
        scenario = ScenarioInputs(
            vat_rate=vat_rate,
            grant_gbp=grant,
            sell_price_input=float(sell_price),
            sell_price_is_gross=sell_is_gross,
            units_sold=int(units_sold),
            order_size_units=order_size_units_base,
            unit_cost_gbp=float(supplier_cost),
            shipping_cost_gbp=float(shipping_cost),
            setup_cost_gbp=0.0,
            extra_fixed_costs_gbp=float(extra_fixed),
            extra_variable_cost_per_unit_gbp=float(extra_var_unit),
        )
        result = compute_outputs(scenario)
        rows.append(
            {
                "Supplier": quote.supplier,
                "Unit Cost (£)": supplier_cost,
                "Shipping (£)": shipping_cost,
                "Profit After Tax (£)": result.profit_after_tax_gbp,
                "ROI on Upfront (%)": result.roi_on_upfront_pct,
                "Break-even Units": result.breakeven_units,
                "Notes": note or "",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Profit After Tax (£)", ascending=False, na_position="last")
    st.dataframe(df, use_container_width=True)

    df2 = df.dropna(subset=["Profit After Tax (£)"]).copy()
    if not df2.empty:
        fig_bar = px.bar(
            df2.head(10),
            x="Supplier",
            y="Profit After Tax (£)",
            title="Top suppliers by profit after tax",
            text_auto=".0f",
        )
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 4: DISTRIBUTIONS ---
with tab_distributions:
    st.subheader("Projected Profit Distribution")

    if out.profit_after_tax_gbp <= 0 or dist.get("distributable", 0.0) <= 0:
        st.warning("No distributable profit after tax in this scenario.")
    else:
        labels = [
            "Charity",
            "CEO",
            "Team Member 1",
            "Team Member 2",
            "Team Member 3",
            "Team Member 4",
            "Team Member 5",
        ]
        values = [
            dist["charity"],
            dist["ceo"],
            dist["others"][0],
            dist["others"][1],
            dist["others"][2],
            dist["others"][3],
            dist["others"][4],
        ]

        c_dist_1, c_dist_2 = st.columns(2)

        with c_dist_1:
            fig_pie = px.pie(
                names=labels,
                values=values,
                hole=0.35,
            )
            fig_pie.update_layout(height=360, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_dist_2:
            st.markdown(
                f"""
            ### Payout Values
            * **Distributable (after tax + charity):** {fmt_money(dist['distributable'], 2)}
            * **Charity:** {fmt_money(dist['charity'], 2)}
            * **CEO:** {fmt_money(dist['ceo'], 2)}
            * **Team Member 1:** {fmt_money(dist['others'][0], 2)}
            * **Team Member 2:** {fmt_money(dist['others'][1], 2)}
            * **Team Member 3:** {fmt_money(dist['others'][2], 2)}
            * **Team Member 4:** {fmt_money(dist['others'][3], 2)}
            * **Team Member 5:** {fmt_money(dist['others'][4], 2)}
            """
            )

# --- TAB 5: INFO ---
with tab_info:
    st.header("Glossary & How to Read the Charts")

    with st.expander("Key metrics", expanded=True):
        st.markdown(
            """
**Prices and VAT**
- **Customer price (inc VAT)** is what buyers pay.
- **Net price (ex VAT)** is what the business keeps before costs.
- Net price = Gross price / 1.20 (VAT rate = 20%).

**Revenue, costs, and profit**
- **Revenue (net)** = Net selling price × Units sold
- **COGS** = Variable cost per unit × Units sold
- **Fixed costs** = Shipping + Setup + extra fixed costs
- **Profit before tax** = Revenue + Grant − (COGS + Fixed costs)
- **Profit after tax (PAT)** = Profit before tax − tax

**Unit economics**
- **Contribution per unit** = Net selling price − Variable cost per unit
- **Contribution margin %** = Contribution per unit ÷ Net selling price
- **Profit per unit (PAT)** = Profit after tax ÷ Units sold
- **Effective net margin %** = Profit after tax ÷ Net revenue
- **Inventory left** = Order size (or batches) − Units sold
"""
        )

    with st.expander("Break-even (ignoring grant)", expanded=True):
        st.markdown(
            """
Break-even is the smallest number of units sold where trading alone covers costs.

- **Break-even units** = ceil(Fixed costs ÷ Contribution per unit)
- **Cash break-even** = ceil(Upfront cash ÷ Contribution per unit)
"""
        )

    with st.expander("Demand curve mode", expanded=True):
        st.markdown(
            """
Demand curve mode replaces manual units with a price → units relationship.

We use two anchor points **(p1, u1)** and **(p2, u2)** and linearly interpolate:
- slope = (u2 − u1) / (p2 − p1)
- units(p) = u1 + slope × (p − p1)

Units are then clamped to a min/max range.
"""
        )

    with st.expander("Distribution pipeline", expanded=True):
        st.markdown(
            """
Distribution uses this pipeline:
1. **Profit before tax**
2. **Tax (20%)**
3. **Profit after tax (PAT)**
4. **Charity** = 30% of PAT (if PAT > 0)
5. **Split** the remainder: CEO 45%, team members 11% each (x5)
"""
        )

    with st.expander("Charts explained", expanded=True):
        st.markdown(
            """
- **Profit vs Units (break-even)**: where the line crosses 0 profit is the trading break-even.
- **Waterfall**: shows how revenue is reduced by COGS, fixed costs and tax to arrive at PAT.
- **Profit band**: shaded region shows uncertainty across unit and cost ranges.
- **Heatmap**: profit across price × units; the contour line is the break-even boundary.
- **Supplier bar chart**: compares PAT across suppliers under the same assumptions.
If **demand curve** mode is on, the heatmap is replaced by likely profit vs price and predicted units vs price.
"""
        )

    with st.expander("Ranges, sensitivity, and limitations", expanded=False):
        st.markdown(
            """
- **Ranges** drive the uncertainty band and heatmap. If a range collapses to a single value, charts simplify.
- **Demand curve mode** assumes units fall as price rises (linear between two anchor points).
- **Sensitivity driver** is inferred from relative input range sizes (price vs units vs cost).
- The model supports optional reorders but does **not** model tiered pricing or stepwise shipping changes (v2 candidates).
"""
        )

    with st.expander("Grant cap warning and likelihood estimates", expanded=True):
        st.markdown(
            f"""
### Grant cap warning (why it exists)
We show a warning when **upfront cash needed** exceeds **£{int(GRANT_CAP)}** (the maximum grant).  
This highlights whether the business can place the production order without extra funding.

**Upfront cash needed** (ignoring the grant):
- Upfront cash = (Order size × Variable cost per deck) + Fixed upfront costs

### Likelihood metrics (how they are calculated)
We estimate:
- **Chance of break-even**: % of sampled scenarios where **profit after tax (PAT) ≥ £0**
- **Chance of paying back grant**: % of sampled scenarios where **PAT ≥ £{int(GRANT_CAP)}**
- **Chance upfront cash within grant**: % where upfront cash needed ≤ £{int(GRANT_CAP)}

**Method**: we randomly sample prices/costs/units inside your chosen ranges (Monte Carlo sampling).  
This is a practical risk indicator, not a guarantee.
"""
        )
