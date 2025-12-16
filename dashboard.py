import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Online Retail Analytics Dashboard", layout="wide")
st.title("🛍️ Online Retail Analytics Dashboard")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    # Load your CSVs
    basket = pd.read_csv("groceries_basket.csv", index_col=0)
    rules = pd.read_csv("association_rules.csv")
    rules.columns = rules.columns.str.strip().str.lower()

    # Simulate additional columns if missing
    np.random.seed(42)
    basket['order_id'] = basket.index
    basket['customer_id'] = np.random.randint(1, basket.shape[0]//2, size=basket.shape[0])
    basket['timestamp'] = pd.to_datetime(
        np.random.choice(pd.date_range("2024-01-01", "2024-12-31"), size=basket.shape[0])
    )
    basket['city'] = np.random.choice(['Manila','Cebu','Davao','Quezon','Baguio'], size=basket.shape[0])
    basket['promotion'] = np.random.choice([0, 1], size=basket.shape[0])
    basket['return'] = np.random.choice([0,1], size=basket.shape[0], p=[0.9,0.1])
    basket['price'] = np.random.randint(5, 100, size=basket.shape[0])  # simulate prices

    return basket, rules

basket, rules = load_data()

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Sales & Revenue", "Customer Behavior", "Product Performance", "Marketing & Engagement", "Basket Analysis"])

# -------------------------
# Item columns
# -------------------------
item_cols = [col for col in basket.columns if col not in ['order_id','customer_id','timestamp','city','promotion','return','price']]

# -------------------------
# 1️⃣ Sales & Revenue
# -------------------------
with tabs[0]:
    st.subheader("💰 Sales & Revenue Overview")
    basket['revenue'] = basket[item_cols].sum(axis=1) * basket['price']

    # Aggregate by time
    time_group = st.radio("Aggregate revenue by:", ["Daily", "Monthly", "Yearly"])
    if time_group == "Daily":
        revenue_time = basket.groupby(basket['timestamp'].dt.date)['revenue'].sum().reset_index()
        revenue_time['timestamp'] = pd.to_datetime(revenue_time['timestamp'])
        x_col = 'timestamp'
    elif time_group == "Monthly":
        revenue_time = basket.groupby(basket['timestamp'].dt.to_period("M"))['revenue'].sum().reset_index()
        revenue_time['timestamp'] = revenue_time['timestamp'].astype(str)
        x_col = 'timestamp'
    else:
        revenue_time = basket.groupby(basket['timestamp'].dt.to_period("Y"))['revenue'].sum().reset_index()
        revenue_time['timestamp'] = revenue_time['timestamp'].astype(str)
        x_col = 'timestamp'

    fig = px.line(revenue_time, x=x_col, y='revenue', title="Revenue Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Top products by revenue
    st.subheader("🛒 Top Products by Revenue")
    item_revenue = (basket[item_cols].sum(axis=0) * basket['price'].mean()).sort_values(ascending=False)
    top_n = st.slider("Top products to show:", 5, 30, 10)
    fig2 = px.bar(item_revenue.head(top_n), x=item_revenue.head(top_n).index, y=item_revenue.head(top_n).values, 
                  labels={'x':'Product','y':'Revenue'}, title="Top Products by Revenue")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# 2️⃣ Customer Behavior
# -------------------------
with tabs[1]:
    st.subheader("👥 Customer Behavior")
    cust_orders = basket.groupby('customer_id')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig3 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values, title="Repeat vs First-time Customers")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top Cities by Transactions")
    top_cities = basket.groupby('city')['order_id'].nunique().sort_values(ascending=False)
    fig4 = px.bar(top_cities, x=top_cities.index, y=top_cities.values, title="Top Cities by Transaction Count")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Basket Size Distribution")
    basket_sizes = basket[item_cols].sum(axis=1)
    fig5, ax5 = plt.subplots(figsize=(10,5))
    sns.histplot(basket_sizes, bins=20, kde=True, color="green", ax=ax5)
    ax5.set_xlabel("Number of Items per Basket")
    ax5.set_ylabel("Number of Transactions")
    ax5.set_title("Basket Size Distribution")
    st.pyplot(fig5)

# -------------------------
# 3️⃣ Product Performance
# -------------------------
with tabs[2]:
    st.subheader("📦 Product Performance")
    return_rates = basket[item_cols].multiply(basket['return'], axis=0).sum().sort_values(ascending=False)
    fig6 = px.bar(return_rates.head(top_n), x=return_rates.head(top_n).index, y=return_rates.head(top_n).values, 
                  title="Top Products by Return Rate")
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Item Co-occurrence Heatmap")
    basket_items = basket[item_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    co_occurrence = basket_items.T.dot(basket_items)
    co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
    fig7, ax7 = plt.subplots(figsize=(12,10))
    sns.heatmap(co_occurrence_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax7)
    ax7.set_title("Item Co-occurrence (% of transactions)")
    st.pyplot(fig7)

# -------------------------
# 4️⃣ Marketing & Engagement
# -------------------------
with tabs[3]:
    st.subheader("📣 Marketing & Engagement")
    promo_sales = basket.groupby('promotion')['revenue'].sum()
    fig8 = px.bar(promo_sales, x=promo_sales.index, y=promo_sales.values, labels={'x':'Promotion','y':'Revenue'},
                  title="Revenue by Promotion")
    st.plotly_chart(fig8, use_container_width=True)

# -------------------------
# 5️⃣ Market Basket Recommendations
# -------------------------
with tabs[4]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols)

    if selected_items:
        recommended_rules = rules[
            rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))
        ]
        sort_cols = [col for col in ["confidence", "lift"] if col in recommended_rules.columns]
        if sort_cols:
            recommended_rules = recommended_rules.sort_values(by=sort_cols, ascending=False)

        if not recommended_rules.empty:
            top_recs = recommended_rules.head(10)
            st.write(f"**Top recommendations for ({', '.join(selected_items)}):**")
            st.dataframe(top_recs[["antecedents","consequents","support","confidence","lift"]], use_container_width=True)

            suggested_items = set()
            for cons in top_recs["consequents"]:
                suggested_items.update(cons.split(", "))
            suggested_items = [item for item in suggested_items if item not in selected_items]
            st.markdown(f"**Suggested items:** {', '.join(suggested_items)}")
        else:
            st.info("No recommendations found for the selected items.")
    else:
        st.info("Select items from the basket to get recommendations.")

st.caption("📘 Dashboard: Sales & Revenue | Customer Behavior | Product Performance | Marketing Insights | Recommendations")
