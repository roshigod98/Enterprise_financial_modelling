import streamlit as st
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on sys.path when running via `python app/main.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.quotes import load_quotes_csv, pick_unit_cost
from model.finance import ScenarioInputs, compute_outputs, distribute_profit


# --- Page Config ---
st.set_page_config(page_title="Enterprise Financial Modeller", layout="wide")
st.title("Oundle Branded Playing Cards — Financial Modeller")

# --- Load Data & Sidebar Inputs ---
quotes = load_quotes_csv("data/quotes_clean.csv")
supplier_names = [q.supplier for q in quotes]

with st.sidebar:
    st.header("1. Supplier & Production")
    supplier = st.selectbox("Supplier", supplier_names, index=0)
    q = next(x for x in quotes if x.supplier == supplier)

    order_size = st.number_input("Order size (units)", min_value=1, value=500, step=50)
    order_size_int = int(order_size)

    # Unit Cost Logic
    unit_cost, unit_cost_note = pick_unit_cost(q, order_size_int)
    if unit_cost is None:
        st.error("Missing unit cost data for this row.")
        st.stop()
    
    st.caption(f"Base Unit Cost: £{unit_cost:.2f} ({unit_cost_note or 'exact'})")
    
    override_cost_flag = st.checkbox("Override Unit Cost")
    if override_cost_flag:
        unit_cost = st.number_input("Purchase unit cost (£)", min_value=0.0, value=float(unit_cost), step=0.01)

    shipping = q.shipping_cost or 0.0
    shipping = st.number_input("Shipping / Fixed Costs (£)", min_value=0.0, value=float(shipping), step=1.0)
    
    st.header("2. Sales Scenarios")
    sell_is_gross = st.selectbox("Price Type", ["Customer (inc VAT)", "Net (ex VAT)"]) == "Customer (inc VAT)"
    sell_price = st.slider("Selling Price per Deck (£)", min_value=5.0, max_value=50.0, value=15.0, step=0.50)
    max_units_sold = order_size_int
    default_units_sold = min(300, max_units_sold)
    units_step = 10 if max_units_sold >= 10 else 1
    units_sold = st.slider(
        "Projected Sales (Units)",
        min_value=0,
        max_value=max_units_sold,
        value=default_units_sold,
        step=units_step,
    )

    st.header("3. Finance Settings")
    grant = st.number_input("Grant Received (£)", min_value=0.0, value=1000.0, step=50.0)
    vat_rate = 0.20  # Fixed for now, could be input

    st.subheader("Distributions")
    charity_pct = st.slider("Charity Share (%)", 0, 100, 30) / 100.0
    ceo_pct = st.slider("CEO Share (of remaining) (%)", 0, 100, 45) / 100.0
    other_each_pct = (1.0 - ceo_pct) / 5.0
    st.caption(f"Team Members (x5): {other_each_pct*100:.1f}% each")

# --- Computations ---
x = ScenarioInputs(
    vat_rate=vat_rate,
    grant_gbp=grant,
    sell_price_input=sell_price,
    sell_price_is_gross=sell_is_gross,
    units_sold=int(units_sold),
    unit_cost_gbp=float(unit_cost),
    shipping_cost_gbp=float(shipping),
    setup_cost_gbp=0.0,
)

out = compute_outputs(x)
dist = distribute_profit(out.profit_gbp, charity_pct, ceo_pct, other_each_pct)

# --- Tabs ---
tab_dashboard, tab_sensitivity, tab_distributions = st.tabs(["Dashboard", "Sensitivity Analysis", "Distributions"])

# --- TAB 1: DASHBOARD ---
with tab_dashboard:
    # Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Profit", f"£{out.profit_gbp:,.2f}")
    c2.metric("ROI", f"{out.roi_pct:.1f}%" if out.roi_pct is not None else "-")
    c3.metric("Break-even Units", "-" if out.breakeven_units is None else f"{out.breakeven_units} units")
    c4.metric("Margin of Safety", f"{out.margin_of_safety_units} units" if out.margin_of_safety_units is not None else "N/A")

    st.divider()

    col_chart, col_details = st.columns([2, 1])

    with col_chart:
        st.subheader("Break-even Analysis")
        # Generate break-even data
        max_u = max(order_size_int, int(out.breakeven_units or 0) * 2 or 100)
        u_range = np.linspace(0, max_u, 50)
        
        fixed_costs_total = out.fixed_costs_gbp  # Shipping + Setup
        # Revenue = Net Price * Units
        rev_line = out.sell_price_net * u_range
        # Total Cost = Fixed Costs + (Unit Cost * Units) - Grant (Grant offsets costs effectively in this simple view? 
        # Actually profit = Rev + Grant - (Variable + Fixed)
        # So for break-even visual: Cost Line = (Variable * Units) + Fixed - Grant
        # If Grant > Fixed, starting 'cost' is negative?
        # Let's visualize strictly: Revenue vs (Variable + Fixed - Grant)
        # Ideally: Revenue Line AND Cost Line. Intersection is BE.
        
        cost_line = (x.unit_cost_gbp * u_range) + fixed_costs_total - x.grant_gbp
        
        fig_be, ax_be = plt.subplots()
        ax_be.plot(u_range, rev_line, label="Revenue", color="green")
        ax_be.plot(u_range, cost_line, label="Total Costs (Net of Grant)", color="red", linestyle="--")
        
        # Add points for current scenario on each line
        u0 = x.units_sold
        rev0 = out.sell_price_net * u0
        cost0 = (x.unit_cost_gbp * u0) + fixed_costs_total - x.grant_gbp
        ax_be.scatter([u0], [rev0], color="green", zorder=5, label="Current Revenue")
        ax_be.scatter([u0], [cost0], color="red", zorder=5, label="Current Cost")
        # Plot current units on the lines
        ax_be.axvline(x.units_sold, color="gray", linestyle=":", alpha=0.5, label="Current Sales")
        
        ax_be.set_xlabel("Units Sold")
        ax_be.set_ylabel("GBP (£)")
        ax_be.legend()
        ax_be.grid(True, alpha=0.3)
        st.pyplot(fig_be)
        st.caption("Note: the grant is treated as offsetting costs, so the cost line can start below £0.")

    with col_details:
        st.subheader("Financials Details")
        st.markdown(f"""
        **Revenue:** £{out.revenue_gbp:,.2f}  
        **COGS:** £{out.cogs_gbp:,.2f}  
        **Fixed Costs:** £{out.fixed_costs_gbp:,.2f}  
        **Grant:** £{x.grant_gbp:,.2f}  
        
        ---
        **Unit Inputs**
        * Sell Price (Net): £{out.sell_price_net:.2f}
        * Unit Cost: £{x.unit_cost_gbp:.2f}
        * Contribution/Unit: £{out.contribution_margin_gbp:.2f}
        """)

# --- TAB 2: SENSITIVITY ---
with tab_sensitivity:
    st.subheader("Profit sensitivity to selling price")
    
    # 1. Profit vs Price
    prices = np.linspace(5, 50, 50)
    profit_by_price = []
    for p in prices:
        # Create temp input
        xi = ScenarioInputs(**{**x.__dict__, "sell_price_input": p, "sell_price_is_gross": x.sell_price_is_gross})
        profit_by_price.append(compute_outputs(xi).profit_gbp)
    
    fig_sens, ax_sens = plt.subplots(figsize=(10, 4))
    ax_sens.plot(prices, profit_by_price)
    ax_sens.axhline(0, color="black", linewidth=1)
    ax_sens.axvline(x.sell_price_input, color="red", linestyle="--", label="Current Price")
    ax_sens.set_xlabel("Selling Price (£)")
    ax_sens.set_ylabel("Projected Profit (£)")
    ax_sens.set_title(f"Profit vs Price (at {x.units_sold} units sold)")
    ax_sens.legend()
    ax_sens.grid(True, alpha=0.3)
    st.pyplot(fig_sens)

    st.divider()

    st.subheader("Profit sensitivity to volume (Units Sold)")
    # 2. Profit vs Units (Existing)
    units_range = np.arange(0, order_size_int + 1, max(1, order_size_int // 50))
    profit_by_units = []
    for u in units_range:
        xi = ScenarioInputs(**{**x.__dict__, "units_sold": int(u)})
        profit_by_units.append(compute_outputs(xi).profit_gbp)
        
    fig_vol, ax_vol = plt.subplots(figsize=(10, 4))
    ax_vol.plot(units_range, profit_by_units, color="purple")
    ax_vol.axhline(0, color="black", linewidth=1)
    ax_vol.axvline(x.units_sold, color="red", linestyle="--", label="Current Sales")
    ax_vol.set_xlabel("Units Sold")
    ax_vol.set_ylabel("Projected Profit (£)")
    ax_vol.legend()
    ax_vol.grid(True, alpha=0.3)
    st.pyplot(fig_vol)

# --- TAB 3: DISTRIBUTIONS ---
with tab_distributions:
    st.subheader("Projected Profit Distribution")
    
    if out.profit_gbp <= 0:
        st.warning("No profit to distribute in this scenario.")
    else:
        # Data for pie chart
        labels = ["Charity", "CEO", "Team (Total)"]
        sizes = [dist["charity"], dist["ceo"], sum(dist["others"])]
        explode = (0.1, 0, 0)  # explode charity

        c_dist_1, c_dist_2 = st.columns(2)
        
        with c_dist_1:
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)

        with c_dist_2:
            st.markdown(f"""
            ### Payout Values
            * **Charity:** £{dist['charity']:,.2f}
            * **CEO:** £{dist['ceo']:,.2f}
            * **Team Total:** £{sum(dist['others']):,.2f}
              * *Per person (x5):* £{dist['others'][0]:,.2f}
            """)
