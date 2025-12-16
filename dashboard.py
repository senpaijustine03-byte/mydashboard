import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Groceries Analytics Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard (Kaggle Dataset)")
st.markdown("Clean, interactive dashboard with insights from the Groceries dataset.")

# -------------------------
# Load and preprocess data
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Groceries_dataset.csv")
    data['timestamp'] = pd.to_datetime(data['Date'])
    data['order_id'] = data['Member_number']
    return data

data = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Filters")
min_date = data['timestamp'].min().date()
max_date = data['timestamp'].max().date()
date_range = st.sidebar.date_input("Select date range:", [min_date, max_date])

item_options = data['itemDescription'].unique().tolist()
selected_items = st.sidebar.multiselect("Filter by items:", item_options)

# Apply filters
filtered_data = data[
    (data['timestamp'].dt.date >= date_range[0]) &
    (data['timestamp'].dt.date <= date_range[1])
]
if selected_items:
    filtered_data = filtered_data[filtered_data['itemDescription'].isin(selected_items)]

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Raw Data", "Unique Items & Customers", "Transactions Overview",
    "Top Items", "Customer Behavior", "Seasonal Trends",
    "Item Co-occurrence", "Basket Recommendations", "Top Bundles"
])

# -------------------------
# 0️⃣ Raw Data
# -------------------------
with tabs[0]:
    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(filtered_data.head(50))
    st.write(f"Total rows: {len(filtered_data)}")

# -------------------------
# 1️⃣ Unique Items & Customers
# -------------------------
with tabs[1]:
    st.subheader("🛒 Unique Items")
    st.write(f"Total unique items: {filtered_data['itemDescription'].nunique()}")
    st.dataframe(pd.DataFrame(filtered_data['itemDescription'].unique(), columns=["ItemDescription"]))

    st.subheader("👥 Unique Customers")
    st.write(f"Total unique customers: {filtered_data['Member_number'].nunique()}")
    st.dataframe(pd.DataFrame(filtered_data['Member_number'].unique(), columns=["CustomerID"]))

# -------------------------
# 2️⃣ Transactions Overview
# -------------------------
with tabs[2]:
    st.subheader("📈 Transactions Over Time")
    time_group = st.radio("Aggregate by:", ["Daily", "Weekly", "Monthly"], key="agg_radio")
    if filtered_data.empty:
        st.info("No transactions in selected filters.")
    else:
        if time_group == "Daily":
            trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.date)['order_id'].count().reset_index()
            trans_time.rename(columns={'order_id': 'transactions'}, inplace=True)
            trans_time['timestamp'] = pd.to_datetime(trans_time['timestamp'])
            x_col = 'timestamp'
        elif time_group == "Weekly":
            trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.isocalendar().week)['order_id'].count().reset_index()
            trans_time.rename(columns={'order_id': 'transactions'}, inplace=True)
            trans_time['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
            x_col = 'timestamp'
        else:
            trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.to_period("M"))['order_id'].count().reset_index()
            trans_time.rename(columns={'order_id': 'transactions'}, inplace=True)
            trans_time['timestamp'] = trans_time['timestamp'].astype(str)
            x_col = 'timestamp'

        fig = px.line(trans_time, x=x_col, y='transactions', title="Transactions Over Time", template="plotly_white", key="trans_time")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 3️⃣ Top Items
# -------------------------
with tabs[3]:
    st.subheader("🛒 Top Items by Frequency")
    if filtered_data.empty:
        st.info("No items to display.")
    else:
        basket_oh = filtered_data.groupby(['Member_number', 'itemDescription'])['itemDescription']\
                        .count().unstack().fillna(0)
        basket_oh = basket_oh.applymap(lambda x: 1 if x > 0 else 0)
        top_items = basket_oh.sum().sort_values(ascending=False)
        top_n = st.slider("Top items to show:", 5, 30, 10, key="top_items_slider")
        fig_items = px.bar(top_items.head(top_n), x=top_items.head(top_n).index, y=top_items.head(top_n).values,
                           labels={'x':'Item','y':'Count'}, title="Top Items by Transaction Count", template="plotly_white", key="top_items_chart")
        st.plotly_chart(fig_items, use_container_width=True)

# -------------------------
# 4️⃣ Customer Behavior
# -------------------------
with tabs[4]:
    st.subheader("👥 Customer Insights")
    if filtered_data.empty:
        st.info("No customer data for selected filters.")
    else:
        cust_orders = filtered_data.groupby('Member_number')['order_id'].nunique()
        repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
        fig_repeat = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                            title="Repeat vs First-time Customers", template="plotly_white", key="repeat_customers")
        st.plotly_chart(fig_repeat, use_container_width=True)
