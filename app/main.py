import sys
from dataclasses import replace
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
from model.finance import ScenarioInputs, compute_outputs, distribute_profit_after_tax, scenario_for_units, ScenarioOutputs
from model.risk import estimate_likelihoods
from model.quotes import Quote, load_quotes_csv, pick_unit_cost
from model.scenarios import (
    profit_band_over_price,
    profit_curve_with_demand,
    profit_heatmap_price_units,
    sweep_profit_over_price,
    scenario_extrema,
)


def fmt_money(value: float, decimals: int = 0) -> str:
    return f"£{value:,.{decimals}f}"


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def narrative_summary(
    q: Quote,
    x: ScenarioInputs,
    out: ScenarioOutputs,
    out_worst: ScenarioOutputs,
    out_best: ScenarioOutputs,
    dist: dict,
    sell_price_min: float,
    sell_price_max: float,
    units_min: int,
    units_max: int,
    unit_cost_min: float,
    unit_cost_max: float,
) -> list[str]:
    bullets: list[str] = []

    bullets.append(
        f"Using **{q.supplier}** with an order of **{out.ordered_units} decks**, unit cost is about "
        f"**{fmt_money(x.unit_cost_gbp, 2)}** (range {fmt_money(unit_cost_min, 2)}–{fmt_money(unit_cost_max, 2)})."
    )

    price_label = "customer price (inc VAT)" if x.sell_price_is_gross else "net price (ex VAT)"
    bullets.append(
        f"Selling price is **{fmt_money(x.sell_price_input, 2)}** ({price_label}); net price is "
        f"**{fmt_money(out.sell_price_net, 2)}**. Variable cost per unit is "
        f"**{fmt_money(out.variable_cost_per_unit_gbp, 2)}**, so contribution is "
        f"**{fmt_money(out.contribution_margin_gbp, 2)}** per deck."
    )

    trading_be = "not reachable" if out.breakeven_units is None else f"{out.breakeven_units} units"
    cash_be = "not reachable" if out.cash_breakeven_units is None else f"{out.cash_breakeven_units} units"
    if not out.trading_breakeven_feasible:
        trading_be += " (unreachable within purchased inventory)"
    if not out.cash_recovery_feasible:
        cash_be += " (unreachable within purchased inventory)"
    bullets.append(
        f"Upfront cash required is **{fmt_money(out.upfront_cash_gbp)}**; trading break-even is "
        f"**{trading_be}** and cash break-even is **{cash_be}** (before tax and financing)."
    )

    bullets.append(
        f"Projected trading profit before tax is **{fmt_money(out.profit_before_tax_gbp)}**; after illustrative tax it is **{fmt_money(out.profit_after_tax_gbp)}**. Cash-limited allocation pool is **{fmt_money(out.distribution_capacity_gbp)}**; "
        f"charity **{fmt_money(dist['charity'])}**, CEO **{fmt_money(dist['ceo'])}**, "
        f"each team member **{fmt_money(dist['others'][0])}**."
    )

    if units_min == units_max and sell_price_min == sell_price_max and unit_cost_min == unit_cost_max:
        bullets.append(
            f"Single-point scenario at **{x.units_sold} units**: worst/base/best are identical."
        )
    else:
        bullets.append(
            f"Evaluated scenario outcomes (illustrative PAT) span **{fmt_money(out_worst.profit_after_tax_gbp)}** to "
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
        driver_note = f"Widest relative input range is **{driver}**; this is not a measured profit sensitivity."

    bullets.append(f"{driver_note} {floor_note}".strip())

    bullets.append(
        f"Full goodwill-loan repayment after illustrative tax is **{'feasible' if out.loan_repayment_feasible else 'not feasible'}** "
        f"in this funded cash ledger. Any initial funding gap of **{fmt_money(out.funding_gap_gbp)}** requires external capital. "
        "Distributions also reserve that capital. Quotes, demand, VAT and settlement timing remain unverified assumptions; these are not historical results."
    )
    return bullets


GRANT_CAP = 1000.0
# --- Page Config ---
st.set_page_config(page_title="Enterprise Financial Modeller", layout="wide")

# --- Header row (left title, right badge) ---
left, right = st.columns([7, 3])
with left:
    st.title("The Oundle Deck")
    st.caption("Financial modelling · Pricing, inventory and funding decisions")
with right:
    st.markdown(
        """
        <div style="text-align:right; font-size:12px; line-height:1.25; padding-top:6px;">
          <b>Made by Roshan Narayan</b><br/>
          Oundle School · 2026<br/>
          Peter Jones Foundation Tycoon Enterprise Competition
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Educational what-if scenarios · GBP · 20% illustrative profit tax · VAT treatment unverified")

# Use a restrained shared chart theme, including graphs created with graph_objects.
import plotly.io as pio
pio.templates["oundle"] = go.layout.Template(
    layout=go.Layout(template="plotly_white", colorway=["#172e4d", "#607d94", "#9b8365", "#9cabb5"],
                     font=dict(family="Arial, sans-serif", color="#172e4d"),
                     margin=dict(l=30, r=30, t=30, b=40))
)
pio.templates.default = "oundle"
px.defaults.template = "oundle"

# --- Load Data & Sidebar Inputs ---
try:
    quotes = load_quotes_csv(str(ROOT / "data" / "quotes_clean.csv"))
except (OSError, ValueError) as exc:
    st.error(f"Supplier data could not be loaded: {exc}")
    st.stop()
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
            if order_size_int < custom_moq:
                st.error(f"Order size is below this supplier's {custom_moq}-deck minimum.")
                st.stop()
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

            if q.moq is not None and order_size_int < q.moq:
                st.error(f"Order size is below this supplier's {q.moq}-deck minimum. Increase the order or choose another supplier.")
                st.stop()
            if q.moq is None:
                st.caption("Supplier minimum order is unknown; confirm before purchasing.")
            unit_cost, unit_cost_note = pick_unit_cost(q, order_size_int)
            if unit_cost is None:
                st.warning("Unit cost is unknown. Enter a confirmed or explicit illustrative assumption.")
                unit_cost = st.number_input("Missing purchase cost assumption (£)", min_value=0.0, value=1.0)
                unit_cost_note = "manual assumption; original quote incomplete"

            if q.shipping_cost is None:
                st.warning("Shipping is unknown. It is not included in the supplier ranking.")
                if not st.checkbox("Use an explicit shipping assumption for this scenario"):
                    st.stop()
            shipping = q.shipping_cost if q.shipping_cost is not None else 0.0
            shipping = st.number_input(
                "Shipping / Fixed Costs (£)", min_value=0.0, value=float(shipping), step=1.0
            )

        if q.vat_included is None:
            st.caption("Supplier VAT status is unknown; costs use quoted GBP without reclaiming input VAT.")
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

            if demand_min_units > demand_max_units or (demand_p1 == demand_p2 and demand_u1 != demand_u2):
                st.error("Use minimum demand ≤ maximum demand and different anchor prices for different quantities.")
                st.stop()
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
            demand_units = [min(demand_curve.predict_units(float(p)), order_size_int) if not allow_reorder else demand_curve.predict_units(float(p)) for p in demand_prices]
            fig_demand = go.Figure()
            fig_demand.add_trace(
                go.Scatter(x=demand_prices, y=demand_units, mode="lines", name="Demand curve")
            )
            fig_demand.update_layout(height=200, xaxis_title="Price (£)", yaxis_title="Units")
            st.plotly_chart(fig_demand, width="stretch")

    with st.expander("Finance & Distribution", expanded=False):
        grant = st.number_input("Goodwill loan (funding, £)", min_value=0.0, value=1000.0, step=50.0)
        vat_rate = 0.20
        st.caption("20% VAT price conversion is illustrative. Choose net price if no output VAT deduction is intended. No card-processing fees apply to school-bill sales.")

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

# --- Computations ---
x = ScenarioInputs(
    vat_rate=vat_rate,
    grant_gbp=grant,
    sell_price_input=float(sell_price),
    sell_price_is_gross=sell_is_gross,
    units_sold=int(units_sold),
    order_size_units=order_size_int,
    allow_reorder=allow_reorder,
    unit_cost_gbp=float(unit_cost),
    shipping_cost_gbp=float(shipping),
    setup_cost_gbp=0.0,
    extra_fixed_costs_gbp=float(extra_fixed),
    extra_variable_cost_per_unit_gbp=float(extra_var_unit),
)

out = compute_outputs(x)
dist = distribute_profit_after_tax(out.distribution_capacity_gbp, charity_pct, ceo_pct, other_each_pct)
out_worst, out_best = scenario_extrema(
    x, sell_price_min, sell_price_max, units_min, units_max, unit_cost_min, unit_cost_max, demand_curve,
)

# --- Tabs ---
tab_overview, tab_range, tab_supplier, tab_distributions, tab_info = st.tabs(
    ["Overview", "Explore Ranges", "Supplier Compare", "Distributions", "Info"]
)

# --- TAB 1: OVERVIEW ---
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trading profit before tax", fmt_money(out.profit_before_tax_gbp))
    k2.metric("Profit after illustrative tax", fmt_money(out.profit_after_tax_gbp))
    k3.metric("Upfront cash requirement", fmt_money(out.upfront_cash_gbp))
    k4.metric("Cash recovery (units)", "Unreachable" if not out.cash_recovery_feasible else str(out.cash_breakeven_units))
    st.caption(f"Inventory remaining: {out.inventory_left_units:,} decks · Net margin: {fmt_pct(out.effective_net_margin_pct)} · Purchased: {out.ordered_units:,} decks")
    if not out.trading_breakeven_feasible or not out.cash_recovery_feasible:
        st.warning("At least one break-even target is impossible within the purchased inventory at this price and cost.")
    st.caption("Reorders buy whole batches at the same unit cost; shipping is charged once. Upfront cash conservatively reserves all batches, without assuming sales finance later orders.")

    st.subheader("Goodwill loan funding limit")
    upfront_cash = out.upfront_cash_gbp
    ratio = upfront_cash / GRANT_CAP if GRANT_CAP > 0 else 0.0
    st.progress(min(1.0, max(0.0, ratio)))
    if upfront_cash <= GRANT_CAP:
        st.success(
            f"Upfront cash required is {fmt_money(upfront_cash)} — within the £{int(GRANT_CAP)} goodwill loan ceiling."
        )
    else:
        gap = upfront_cash - GRANT_CAP
        st.warning(
            f"Upfront cash required is {fmt_money(upfront_cash)} — exceeds the £{int(GRANT_CAP)} goodwill loan ceiling "
            f"by {fmt_money(gap)}."
        )

    st.subheader("What-if scenario coverage")
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
        base_order_size_units=order_size_int,
        allow_reorder=allow_reorder,
    )

    l1, l2, l3 = st.columns(3)
    l1.metric("Scenarios with trading profit", f"{lik['p_break_even']*100:.0f}%")
    l2.metric("Scenarios repaying loan after tax", f"{lik['p_payback_grant']*100:.0f}%")
    l3.metric("Scenarios within £1,000 loan cap", f"{lik['p_upfront_within_grant']*100:.0f}%")
    st.caption(
        "Shares of uniformly sampled hypothetical scenarios, NOT empirical probabilities. "
        "The Tycoon loan is financing, not profit."
    )

    st.caption(f"Pre-tax expenditure recovered in {lik['p_cash_recovery']*100:.0f}% of samples. Seed 42 · 2,000 scenarios. Repayment assumes external capital fills any initial funding gap.")
    st.markdown("### Scenario narrative")
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
    )
    with st.container(border=True):
        for bullet in bullets:
            st.markdown(f"- {bullet}")

    st.subheader("Waterfall: Where the money goes")
    measures = ["relative", "relative", "relative", "relative", "total"]
    xw = ["Revenue", "COGS", "Fixed costs", "Tax", "Profit after tax"]
    yw = [
        out.revenue_gbp,
        -out.cogs_gbp,
        -out.fixed_costs_gbp,
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
    st.plotly_chart(fig_w, width="stretch")
    st.caption("Repayable goodwill loan proceeds are not trading income. The 20% tax is illustrative.")

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
    st.plotly_chart(fig_unit, width="stretch")

    st.subheader("Upfront cash composition")
    batch_cost = out.variable_cost_per_unit_gbp * out.ordered_units
    fixed_costs = out.fixed_costs_gbp
    cash_df = pd.DataFrame(
        {"Component": ["Batch production cost", "Fixed costs"], "Value": [batch_cost, fixed_costs]}
    )
    if cash_df["Value"].sum() > 0:
        fig_cash = px.pie(cash_df, names="Component", values="Value", hole=0.35)
        fig_cash.update_layout(height=320)
        st.plotly_chart(fig_cash, width="stretch")
    else:
        st.info("No upfront expenditure under these assumptions.")

    st.caption("Manual scenario bounds evaluate all price/volume/cost corners. Demand-linked bounds sample 201 prices; they are evaluated ranges, not confidence intervals.")
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
        st.dataframe(summary, width="stretch")

    with st.expander("Detailed metrics"):
        st.markdown(
            f"""
        - **Revenue (net):** {fmt_money(out.revenue_gbp, 2)}
        - **COGS:** {fmt_money(out.cogs_gbp, 2)}
        - **Fixed costs:** {fmt_money(out.fixed_costs_gbp, 2)}
        - **Goodwill loan:** {fmt_money(x.grant_gbp, 2)}
        - **Profit before tax:** {fmt_money(out.profit_before_tax_gbp, 2)}
        - **Illustrative tax (20%):** {fmt_money(out.tax_gbp, 2)}
        - **Profit after tax:** {fmt_money(out.profit_after_tax_gbp, 2)}

        ---
        - **Net price:** {fmt_money(out.sell_price_net, 2)}
        - **Variable cost / unit:** {fmt_money(out.variable_cost_per_unit_gbp, 2)}
        - **Contribution / unit:** {fmt_money(out.contribution_margin_gbp, 2)}
        - **Contribution margin:** {fmt_pct(out.contribution_margin_pct)}
        """
        )

    st.divider()

    col_chart, col_details = st.columns([2, 1])

    with col_chart:
        st.subheader("Trading profit vs units (loan excluded)")

        max_u = max(order_size_int, units_max) if allow_reorder else order_size_int
        u_grid = np.unique(np.round(np.linspace(0, max_u, 140)).astype(int))
        u_grid = u_grid[u_grid >= 0]

        curve_outputs = [compute_outputs(scenario_for_units(x, int(u))) for u in u_grid]
        profit_curve = [o.profit_after_tax_gbp for o in curve_outputs]
        cash_curve = [o.revenue_gbp-o.upfront_cash_gbp for o in curve_outputs]

        fig_be = go.Figure()
        fig_be.add_trace(
            go.Scatter(x=u_grid, y=profit_curve, mode="lines", name="Trading profit after tax")
        )
        fig_be.add_trace(go.Scatter(x=u_grid, y=cash_curve, mode="lines", name="Net receipts less upfront spend", line=dict(dash="dot")))
        fig_be.add_hline(y=0, line=dict(color="#444", width=1))

        if out.trading_breakeven_feasible:
            fig_be.add_vline(x=out.breakeven_units, line=dict(color="#172e4d", dash="dash"))
            fig_be.add_annotation(
                x=out.breakeven_units,
                y=0,
                text="Trading BE",
                showarrow=True,
                arrowhead=2,
                yshift=18,
            )

        if out.cash_recovery_feasible and not allow_reorder:
            fig_be.add_vline(x=out.cash_breakeven_units, line=dict(color="#d62728", dash="dot"))
            fig_be.add_annotation(
                x=out.cash_breakeven_units,
                y=0,
                text="Pre-tax cash recovery",
                showarrow=True,
                arrowhead=2,
                yshift=-28,
            )

        fig_be.update_layout(
            height=440,
            xaxis_title="Units sold",
            yaxis_title="GBP (illustrative trading PAT / pre-tax cash recovery)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_be, width="stretch")
        st.caption("Both curves exclude financing. The cash curve deducts all purchased inventory once, before illustrative tax. Reorders create steps in required cash.")

    with col_details:
        st.subheader("Assumptions (midpoint)")
        price_label = "Price (inc VAT)" if sell_is_gross else "Price (ex VAT)"
        st.markdown(
            f"""
        - **{price_label}:** {fmt_money(x.sell_price_input, 2)}
        - **Net price:** {fmt_money(out.sell_price_net, 2)}
        - **Units sold:** {x.units_sold}
        - **Purchased inventory:** {out.ordered_units}
        - **Unit cost:** {fmt_money(x.unit_cost_gbp, 2)}
        - **Extra variable / unit:** {fmt_money(x.extra_variable_cost_per_unit_gbp, 2)}
        - **Extra fixed costs:** {fmt_money(x.extra_fixed_costs_gbp, 2)}
        - **Goodwill loan:** {fmt_money(x.grant_gbp, 2)}
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
        st.subheader("Illustrative profit vs price (assumed demand)")
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
                    fillcolor="rgba(96, 125, 148, 0.22)",
                    line=dict(width=0),
                    name="Profit range (cost uncertainty)",
                )
            )

        fig_profit.add_trace(
            go.Scatter(
                x=prices,
                y=profits,
                mode="lines",
                name="Illustrative trading PAT",
                line=dict(width=3, color="#172e4d"),
            )
        )
        fig_profit.add_hline(y=0, line=dict(color="#444", width=1))
        fig_profit.update_layout(
            height=420, xaxis_title="Selling price (£)", yaxis_title="Profit after tax (£)"
        )
        st.plotly_chart(fig_profit, width="stretch")

        st.subheader("Assumed units vs price")
        fig_units = go.Figure()
        fig_units.add_trace(
            go.Scatter(x=prices, y=units_pred, mode="lines", name="Assumed units")
        )
        fig_units.update_layout(height=320, xaxis_title="Selling price (£)", yaxis_title="Units")
        st.plotly_chart(fig_units, width="stretch")
    else:
        st.subheader("Profit across selling price range")

        price_collapsed = sell_price_min == sell_price_max
        units_collapsed = units_min == units_max

        if price_collapsed:
            st.warning(
                "Selling price range is a single value — switch to Range mode to see the band and landscape."
            )
            max_u = max(order_size_int, units_max) if allow_reorder else order_size_int
            units_curve = np.unique(np.round(np.linspace(0, max_u, 80)).astype(int))
            units_curve = units_curve[units_curve >= 0]
            profits = []
            for u in units_curve:
                xi = scenario_for_units(replace(x, sell_price_input=float(sell_price_min)), int(u))
                profits.append(compute_outputs(xi).profit_after_tax_gbp)
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(x=units_curve, y=profits, mode="lines", name="Profit (PAT)")
            )
            fig_line.add_hline(y=0, line=dict(color="#444", width=1))
            fig_line.update_layout(
                height=420, xaxis_title="Units sold", yaxis_title="Profit after tax (£)"
            )
            st.plotly_chart(fig_line, width="stretch")
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
                    name="Maximum evaluated PAT",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=prices,
                    y=p_lo,
                    mode="lines",
                    name="Envelope across volume and cost extremes",
                    fill="tonexty",
                    fillcolor="rgba(96, 125, 148, 0.22)",
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
                    line=dict(width=3, color="#172e4d"),
                )
            )
            fig_band.add_hline(y=0, line=dict(color="#444", width=1))
            fig_band.update_layout(
                height=420,
                xaxis_title="Selling price (£)",
                yaxis_title="Profit after tax (£)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_band, width="stretch")

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
                    colorbar=dict(title="Illustrative PAT (£)"),
                    colorscale="RdBu", zmid=0,
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
            st.plotly_chart(fig_h, width="stretch")

# --- TAB 3: SUPPLIER COMPARE ---
with tab_supplier:
    st.subheader("Supplier comparison (same selling assumptions)")

    st.caption("Indicative historical quotes; unknown shipping/costs and orders below known MOQ are excluded. Fallback price tiers and unverified VAT are shown explicitly. Costs stay on the quoted GBP basis without input VAT reclaim.")
    rows: list[dict] = []
    for quote in quotes:
        supplier_cost, note = pick_unit_cost(quote, order_size_int)
        if supplier_cost is None or quote.shipping_cost is None or (quote.moq is not None and order_size_int < quote.moq):
            rows.append(
                {
                    "Supplier": quote.supplier,
                    "Unit Cost (£)": supplier_cost,
                    "Shipping (£)": quote.shipping_cost,
                    "Profit After Tax (£)": None,
                    "ROI on Upfront (%)": None,
                    "Break-even Units": None,
                    "Notes": "Incomplete cost/shipping or order below MOQ; excluded from ranking",
                }
            )
            continue

        shipping_cost = quote.shipping_cost
        scenario = ScenarioInputs(
            vat_rate=vat_rate,
            grant_gbp=grant,
            sell_price_input=float(sell_price),
            sell_price_is_gross=sell_is_gross,
            units_sold=int(units_sold),
            order_size_units=order_size_int,
            allow_reorder=allow_reorder,
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
                "Notes": "; ".join(filter(None, [note, "VAT unknown" if quote.vat_included is None else "VAT included" if quote.vat_included else "VAT excluded", "MOQ unknown" if quote.moq is None else ""])),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Profit After Tax (£)", ascending=False, na_position="last")
    st.dataframe(df, width="stretch", column_config={
        name: st.column_config.NumberColumn(format="£%.2f")
        for name in ("Unit Cost (£)", "Shipping (£)", "Profit After Tax (£)")
    })

    df2 = df.dropna(subset=["Profit After Tax (£)"]).copy()
    if not df2.empty:
        df2["Supplier label"] = df2["Supplier"].map(lambda name: name if len(name) <= 38 else name[:35] + "…")
        fig_bar = px.bar(
            df2.head(10),
            x="Profit After Tax (£)",
            y="Supplier label",
            orientation="h",
            hover_name="Supplier",
            title="Complete quotes: illustrative profit after tax",
            text_auto=".0f",
        )
        fig_bar.update_layout(height=420, yaxis_title="", yaxis=dict(categoryorder="total ascending"), xaxis=dict(tickprefix="£"))
        st.plotly_chart(fig_bar, width="stretch")

# --- TAB 4: DISTRIBUTIONS ---
with tab_distributions:
    st.subheader("Projected Profit Distribution")

    st.caption(f"Accounting PAT: {fmt_money(out.profit_after_tax_gbp, 2)} · Cash-limited allocation pool: {fmt_money(out.distribution_capacity_gbp, 2)}. Loan repayment and external capital recovery are reserved before charity/team allocations.")
    if out.distribution_capacity_gbp <= 0:
        st.warning("No illustrative payout until trading is profitable and upfront cash is recoverable.")
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
            st.plotly_chart(fig_pie, width="stretch")

        with c_dist_2:
            st.markdown(
                f"""
            ### Payout Values
            * **Team pool (after tax, capital recovery + charity):** {fmt_money(dist['distributable'], 2)}
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
- **Net price (ex VAT)** is the modelled price before costs, after any illustrative VAT conversion.
- Illustrative net price = Gross price / 1.20. VAT registration and actual school treatment are unverified.

**Revenue, costs, and profit**
- **Revenue (net)** = Net selling price × Units sold
- **COGS** = Variable cost per unit × Units sold
- **Fixed costs** = Shipping + Setup + extra fixed costs
- **Profit before tax** = Revenue − (COGS + Fixed costs). The goodwill loan is repayable financing, not profit.
- **Profit after tax (PAT)** = Profit before tax − tax

**Unit economics**
- **Contribution per unit** = Net selling price − Variable cost per unit
- **Contribution margin %** = Contribution per unit ÷ Net selling price
- **Profit per unit (PAT)** = Profit after tax ÷ Units sold
- **Effective net margin %** = Profit after tax ÷ Net revenue
- **Inventory left** = Order size (or batches) − Units sold
"""
        )

    with st.expander("Break-even (excluding loan financing)", expanded=True):
        st.markdown(
            """
Break-even is the smallest number of units sold where trading alone covers costs.

- **Break-even units** = ceil(Fixed costs ÷ Contribution per unit)
- **Cash break-even** = ceil(Upfront cash ÷ Net selling price): the supplier inventory purchase has already been paid upfront, so it must not be deducted again when recovering cash.
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
2. **Illustrative tax (20%)**
3. **Profit after tax (PAT)**
4. **Cash-limited pool** = smaller of positive PAT and cash after tax, full loan repayment and external capital recovery
5. **Charity** = 30% of that pool (default)
6. **Split** the remainder: CEO 45%, team members 11% each (x5)
"""
        )

    with st.expander("Charts explained", expanded=True):
        st.markdown(
            """
- **Profit vs Units (break-even)**: where the line crosses 0 profit is the trading break-even.
- **Waterfall**: shows trading revenue minus COGS, fixed costs and illustrative tax to arrive at PAT. Loan funding is not profit.
- **Profit band**: shaded region shows uncertainty across unit and cost ranges.
- **Heatmap**: profit across price × units; the contour line is the break-even boundary.
- **Supplier bar chart**: compares PAT across suppliers under the same assumptions.
If **demand curve** mode is on, the heatmap is replaced by illustrative profit vs price and assumed units vs price.
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

    with st.expander("Funding and scenario sampling", expanded=True):
        st.markdown(
            f"""
### Goodwill loan funding ceiling (why it exists)
We show a warning when **upfront cash needed** exceeds **£{int(GRANT_CAP)}** (the maximum goodwill loan).
This highlights whether the business can place the production order without extra funding.

**Upfront cash needed** (before loan funding):
- Upfront cash = (Order size × Variable cost per deck) + Fixed upfront costs

### Likelihood metrics (how they are calculated)
We estimate:
- **Trading profit**: % of samples with trading profit before tax ≥ £0.
- **Cash recovery**: % of samples where net sales receipts cover upfront expenditure, before tax.
- **Loan repayment feasible**: % of samples where the funded cash ledger can repay principal after illustrative tax.
- **Within funding cap**: % of samples with initial expenditure ≤ £{int(GRANT_CAP)}.

**Method**: we randomly sample prices/costs/units inside your chosen ranges (Monte Carlo sampling).
These are percentages of hypothetical scenarios, not calibrated real-world probabilities. The 20% tax/VAT assumptions are illustrative and must be checked with school finance.
"""
        )
