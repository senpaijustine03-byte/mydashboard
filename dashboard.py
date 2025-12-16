import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Online Retail Dashboard", layout="wide")
st.title("🛍️ Online Retail Analytics Dashboard")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    # Replace with your dataset
    orders = pd.read_csv("groceries_full.csv", parse_dates=["timestamp"])  # timestamp column needed
    return orders

orders = load_data()

# Normalize columns
orders.columns = orders.columns.str.strip().str.lower()

# -------------------------
# KPI Section
# -------------------------
st.subheader("📌 Key Metrics")
total_sales = (orders['quantity'] * orders['price']).sum()
total_orders = orders['order_id'].nunique()
average_order_value = total_sales / total_orders
repeat_customers = orders.groupby('customer_id')['order_id'].nunique()
repeat_pct = (repeat_customers > 1).mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${total_sales:,.2f}")
c2.metric("Total Orders", f"{total_orders}")
c3.metric("Average Order Value", f"${average_order_value:,.2f}")
c4.metric("Repeat Customers %", f"{repeat_pct:.1f}%")

st.divider()

# -------------------------
# 1️⃣ Sales & Revenue
# -------------------------
st.subheader("💰 Sales & Revenue Over Time")
time_group = st.radio("Aggregate by:", ["Daily", "Monthly", "Yearly"])
if time_group == "Daily":
    sales_time = orders.groupby(orders['timestamp'].dt.date)['quantity','price'].apply(lambda x: (x['quantity']*x['price']).sum()).reset_index()
    x_col = 'timestamp'
elif time_group == "Monthly":
    sales_time = orders.groupby(orders['timestamp'].dt.to_period("M"))['quantity','price'].apply(lambda x: (x['quantity']*x['price']).sum()).reset_index()
    x_col = 'timestamp'
else:
    sales_time = orders.groupby(orders['timestamp'].dt.to_period("Y"))['quantity','price'].apply(lambda x: (x['quantity']*x['price']).sum()).reset_index()
    x_col = 'timestamp'

fig1 = px.line(sales_time, x=x_col, y='price', title="Revenue Over Time")
st.plotly_chart(fig1, use_container_width=True)

# Top Products by Revenue
st.subheader("🛒 Top Products by Revenue")
orders['revenue'] = orders['quantity'] * orders['price']
top_products = orders.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(15)
fig2 = px.bar(top_products, x=top_products.index, y='revenue', title="Top 15 Products by Revenue")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -------------------------
# 2️⃣ Customer Behavior
# -------------------------
st.subheader("👥 Customer Behavior")
top_cities = orders.groupby('city')['order_id'].nunique().sort_values(ascending=False).head(10)
fig3 = px.bar(top_cities, x=top_cities.index, y=top_cities.values, title="Top 10 Cities by Purchase Activity")
st.plotly_chart(fig3, use_container_width=True)

# Repeat vs New customers
repeat_status = orders.groupby('customer_id')['order_id'].nunique().apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
fig4 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values, title="Repeat vs First-time Customers")
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# -------------------------
# 3️⃣ Product Performance
# -------------------------
st.subheader("📦 Product Performance")
if 'return' in orders.columns:
    return_rates = orders.groupby('product_name')['return'].mean().sort_values(ascending=False).head(15)
    fig5 = px.bar(return_rates, x=return_rates.index, y='return', title="Top 15 Products by Return Rate")
    st.plotly_chart(fig5, use_container_width=True)

if 'stock' in orders.columns:
    out_of_stock = orders[orders['stock'] == 0].groupby('product_name')['order_id'].count().sort_values(ascending=False).head(10)
    fig6 = px.bar(out_of_stock, x=out_of_stock.index, y=out_of_stock.values, title="Top Products Out of Stock")
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# -------------------------
# 4️⃣ Marketing & Engagement (simulated)
# -------------------------
st.subheader("📣 Marketing & Engagement")
if 'promotion' in orders.columns:
    promo_sales = orders.groupby('promotion')['revenue'].sum()
    fig7 = px.bar(promo_sales, x=promo_sales.index, y='revenue', title="Revenue by Promotion Type")
    st.plotly_chart(fig7, use_container_width=True)

st.caption("📘 Online Retail Dashboard: Sales, Customer Behavior, Product Performance, Marketing Insights")
