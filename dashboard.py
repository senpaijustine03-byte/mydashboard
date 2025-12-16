import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Groceries Analytics Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard")

# -------------------------
# Load and preprocess data
# -------------------------
@st.cache_data
def load_data():
    # Load raw dataset
    data = pd.read_csv("Groceries_dataset.csv")  # Kaggle dataset
    data['timestamp'] = pd.to_datetime(data['Date'])
    data['order_id'] = data['Member_number']

    # One-hot encode basket for co-occurrence & association rules
    basket = data.groupby(['Member_number', 'itemDescription'])['itemDescription']\
                 .count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    basket.reset_index(inplace=True)

    # Simulate extra columns for demo purposes
    np.random.seed(42)
    basket['price'] = np.random.randint(5, 100, size=basket.shape[0])
    basket['revenue'] = basket[basket.columns[1:-3]].sum(axis=1) * basket['price']
    basket['promotion'] = np.random.choice([0,1], size=basket.shape[0])
    basket['return'] = np.random.choice([0,1], size=basket.shape[0], p=[0.9,0.1])
    basket['city'] = np.random.choice(['Manila','Cebu','Davao','Quezon','Baguio'], size=basket.shape[0])

    # For association rules, generate dummy rules for demo
    unique_items = basket.columns[1:-5]
    rules = pd.DataFrame({
        'antecedents': [', '.join(np.random.choice(unique_items, 1, replace=False)) for _ in range(50)],
        'consequents': [', '.join(np.random.choice(unique_items, 1, replace=False)) for _ in range(50)],
        'support': np.random.rand(50),
        'confidence': np.random.rand(50),
        'lift': np.random.rand(50)*3
    })

    return basket, rules, data

basket, rules, raw_data = load_data()

item_cols = [col for col in basket.columns if col not in ['Member_number','price','revenue','promotion','return','city']]

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Sales & Revenue", "Customer Behavior", "Product Performance", "Marketing & Engagement", "Basket Analysis"])

# -------------------------
# 1️⃣ Sales & Revenue
# -------------------------
with tabs[0]:
    st.subheader("💰 Sales & Revenue Overview")
    # Aggregate by time
    time_group = st.radio("Aggregate revenue by:", ["Daily", "Monthly", "Yearly"])
    raw_data['timestamp'] = pd.to_datetime(raw_data['Date'])
    if time_group == "Daily":
        revenue_time = raw_data.groupby(raw_data['timestamp'].dt.date)['Member_number'].count().reset_index()
        revenue_time.rename(columns={'Member_number':'transactions'}, inplace=True)
        x_col = 'timestamp'
    elif time_group == "Monthly":
        revenue_time = raw_data.groupby(raw_data['timestamp'].dt.to_period("M"))['Member_number'].count().reset_index()
        revenue_time.rename(columns={'Member_number':'transactions'}, inplace=True)
        revenue_time['timestamp'] = revenue_time['timestamp'].astype(str)
        x_col = 'timestamp'
    else:
        revenue_time = raw_data.groupby(raw_data['timestamp'].dt.to_period("Y"))['Member_number'].count().reset_index()
        revenue_time.rename(columns={'Member_number':'transactions'}, inplace=True)
        revenue_time['timestamp'] = revenue_time['timestamp'].astype(str)
        x_col = 'timestamp'

    fig = px.line(revenue_time, x=x_col, y='transactions', title="Transactions Over Time")
    st.plotly_chart(fig, use_container_width=True)

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
    cust_orders = basket.groupby('Member_number')['revenue'].count()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig3 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values, title="Repeat vs First-time Customers")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top Cities by Transactions")
    top_cities = basket.groupby('city')['Member_number'].count().sort_values(ascending=False)
    fig4 = px.bar(top_cities, x=top_cities.index, y=top_cities.values, title="Top Cities by Transaction Count")
    st.plotly_chart(fig4, use_container_width=True)

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
