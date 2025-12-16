import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Groceries Analytics Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard (Kaggle Dataset)")

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

item_cols = data['itemDescription'].unique().tolist()
selected_items_sidebar = st.sidebar.multiselect("Filter by items (optional):", options=item_cols, default=[])

filtered_data = data[
    (data['timestamp'].dt.date >= date_range[0]) &
    (data['timestamp'].dt.date <= date_range[1])
]
if selected_items_sidebar:
    filtered_data = filtered_data[filtered_data['itemDescription'].isin(selected_items_sidebar)]

# -------------------------
# One-hot encode basket
# -------------------------
basket_oh = filtered_data.groupby(['Member_number', 'itemDescription'])['itemDescription']\
                .count().unstack().fillna(0)
basket_oh = basket_oh.applymap(lambda x: 1 if x > 0 else 0)
item_cols_filtered = basket_oh.columns.tolist()

# -------------------------
# Generate association rules
# -------------------------
@st.cache_data
def generate_rules(basket_oh):
    if basket_oh.empty:
        return pd.DataFrame()
    frequent_itemsets = apriori(basket_oh, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules

rules = generate_rules(basket_oh)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Raw Data", "Unique Items & Customers", "Transactions Overview",
    "Top Items", "Customer Behavior", "Seasonal Trends",
    "Item Co-occurrence", "Basket Recommendations", "Top Bundles & Item Combos"
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
    st.subheader("🛒 All Unique Items")
    st.dataframe(pd.DataFrame(filtered_data['itemDescription'].unique(), columns=["ItemDescription"]))
    st.write(f"Total unique items: {filtered_data['itemDescription'].nunique()}")

    st.subheader("👥 All Unique Customers")
    st.dataframe(pd.DataFrame(filtered_data['Member_number'].unique(), columns=["CustomerID"]))
    st.write(f"Total unique customers: {filtered_data['Member_number'].nunique()}")

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
            trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
            trans_time['timestamp'] = pd.to_datetime(trans_time['timestamp'])
        elif time_group == "Weekly":
            filtered_data['week_start'] = filtered_data['timestamp'] - pd.to_timedelta(filtered_data['timestamp'].dt.weekday, unit='d')
            trans_time = filtered_data.groupby('week_start')['order_id'].count().reset_index()
            trans_time.rename(columns={'order_id':'transactions','week_start':'timestamp'}, inplace=True)
        else:  # Monthly
            filtered_data['month'] = filtered_data['timestamp'].dt.to_period('M').dt.to_timestamp()
            trans_time = filtered_data.groupby('month')['order_id'].count().reset_index()
            trans_time.rename(columns={'order_id':'transactions','month':'timestamp'}, inplace=True)

        fig = px.line(trans_time, x='timestamp', y='transactions',
                      title="Transactions Over Time", template="plotly_white",
                      markers=True, key="trans_time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📅 Peak Transaction Days")
        peak_days = filtered_data['timestamp'].dt.day_name().value_counts()
        fig_peak = px.bar(peak_days, x=peak_days.index, y=peak_days.values,
                          labels={'x':'Day of Week','y':'Transactions'},
                          title="Transactions by Day of Week", template="plotly_white", key="peak_days")
        st.plotly_chart(fig_peak, use_container_width=True)

# -------------------------
# Remaining tabs remain the same, add unique keys to every Plotly chart/widget
# -------------------------
