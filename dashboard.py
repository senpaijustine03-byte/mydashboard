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
st.markdown("""
<style>
body {background-color: #f9f9f9;}
h1 {color: #2c3e50;}
</style>
""", unsafe_allow_html=True)
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
item_cols = data['itemDescription'].unique().tolist()
selected_items_sidebar = st.sidebar.multiselect(
    "Filter by items (optional):",
    options=item_cols,
    default=[]
)
filtered_data = data[
    (data['timestamp'].dt.date >= date_range[0]) &
    (data['timestamp'].dt.date <= date_range[1])
]
if selected_items_sidebar:
    filtered_data = filtered_data[filtered_data['itemDescription'].isin(selected_items_sidebar)]
if filtered_data.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

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
    if rules.empty:
        return pd.DataFrame()
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules

rules = generate_rules(basket_oh)

# -------------------------
# Key Metrics with color
# -------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Total Transactions", value=len(filtered_data['order_id'].unique()), delta_color="inverse")
col2.metric(label="Total Customers", value=filtered_data['Member_number'].nunique())
col3.metric(label="Total Unique Items", value=filtered_data['itemDescription'].nunique())
col4.metric(label="Avg Items / Transaction", value=f"{basket_oh.sum(axis=1).mean():.2f}")
st.markdown("---")

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
    st.info(f"Total rows: {len(filtered_data)}")

# -------------------------
# 1️⃣ Unique Items & Customers
# -------------------------
with tabs[1]:
    st.subheader("🛒 Unique Items")
    st.dataframe(pd.DataFrame(filtered_data['itemDescription'].unique(), columns=["ItemDescription"]))
    st.metric("Total unique items", filtered_data['itemDescription'].nunique())

    st.subheader("👥 Unique Customers")
    st.dataframe(pd.DataFrame(filtered_data['Member_number'].unique(), columns=["CustomerID"]))
    st.metric("Total unique customers", filtered_data['Member_number'].nunique())

# -------------------------
# 2️⃣ Transactions Overview
# -------------------------
with tabs[2]:
    st.subheader("📈 Transactions Over Time")
    time_group = st.radio("Aggregate by:", ["Daily", "Weekly", "Monthly"])
    if time_group == "Daily":
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.date)['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = pd.to_datetime(trans_time['timestamp'])
    elif time_group == "Weekly":
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.isocalendar().week)['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
    else:
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.to_period("M"))['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = trans_time['timestamp'].astype(str)

    fig = px.line(trans_time, x='timestamp', y='transactions', 
                  title="Transactions Over Time", template="plotly_white", markers=True, color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Peak Transaction Days")
    peak_days = filtered_data['timestamp'].dt.day_name().value_counts()
    fig_peak = px.bar(peak_days, x=peak_days.index, y=peak_days.values,
                      labels={'x':'Day of Week','y':'Transactions'}, 
                      title="Transactions by Day of Week", template="plotly_white",
                      color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig_peak, use_container_width=True)

# -------------------------
# 3️⃣ Top Items
# -------------------------
with tabs[3]:
    st.subheader("🛒 Top Items")
    item_freq = basket_oh.sum().sort_values(ascending=False)
    top_n = st.slider("Top items to show:", 5, 30, 10)
    fig2 = px.bar(item_freq.head(top_n), x=item_freq.head(top_n).index, y=item_freq.head(top_n).values,
                  labels={'x':'Item','y':'Count'}, title="Top Items by Transaction Count", template="plotly_white",
                  color_discrete_sequence=['#ff7f0e'])
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# 4️⃣ Customer Behavior
# -------------------------
with tabs[4]:
    st.subheader("👥 Customer Behavior")
    cust_orders = filtered_data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig3 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                  title="Repeat vs First-time Customers", template="plotly_white",
                  color_discrete_sequence=['#d62728', '#9467bd'])
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# 5️⃣ Seasonal Trends
# -------------------------
with tabs[5]:
    st.subheader("📅 Monthly Trends")
    monthly_sales = filtered_data.groupby(filtered_data['timestamp'].dt.month)['order_id'].count()
    fig4 = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values,
                   labels={'x':'Month', 'y':'Number of Transactions'}, 
                   title="Monthly Transaction Trend", template="plotly_white",
                   markers=True, color_discrete_sequence=['#17becf'])
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# 6️⃣ Item Co-occurrence
# -------------------------
with tabs[6]:
    st.subheader("📊 Top 20 Item Co-occurrence")
    if not basket_oh.empty:
        basket_items = basket_oh.astype(float)
        top_items = basket_items.sum().sort_values(ascending=False).head(20).index
        co_occurrence = basket_items[top_items].T.dot(basket_items[top_items])
        co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
        fig5, ax5 = plt.subplots(figsize=(12,10))
        sns.heatmap(co_occurrence_pct, annot=False, cmap="YlGnBu", ax=ax5)
        ax5.set_title("Item Co-occurrence (% of transactions)")
        st.pyplot(fig5)
    else:
        st.info("No data available for co-occurrence heatmap.")

# -------------------------
# 7️⃣ Basket Recommendations
# -------------------------
with tabs[7]:
    st.subheader("🛍️ Basket Recommendations")
    selected_items = st.multiselect("Select items:", options=item_cols_filtered)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3)
    min_lift = st.slider("Minimum lift", 0.0, 5.0, 1.0)
    if not rules.empty and selected_items:
        recommended_rules = rules[
            (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ]
        if not recommended_rules.empty:
            st.dataframe(recommended_rules[['antecedents','consequents','support','confidence','lift']])
        else:
            st.info("No matching recommendations found.")
    else:
        st.info("No rules available or select items.")

# -------------------------
# 8️⃣ Top Bundles
# -------------------------
with tabs[8]:
    st.subheader("📦 Top Bundles")
    if not basket_oh.empty:
        selected_item_combo = st.selectbox("Select item for bundles:", options=item_cols_filtered)
        top_n_bundles = st.slider("Top bundles to show:", 3, 15, 5)
        transactions_with_item = basket_oh[basket_oh[selected_item_combo]==1]
        co_occurring_counts = transactions_with_item.sum().sort_values(ascending=False)
        co_occurring_counts = co_occurring_counts.drop(labels=[selected_item_combo])
        if not co_occurring_counts.empty:
            top_co_occurring = co_occurring_counts.head(top_n_bundles)
            fig_bundle = px.bar(top_co_occurring, x=top_co_occurring.index, y=top_co_occurring.values,
                                labels={'x':'Item','y':'Count'}, title=f"Top {top_n_bundles} items bought with {selected_item_combo}",
                                template="plotly_white", color_discrete_sequence=['#bcbd22'])
            st.plotly_chart(fig_bundle, use_container_width=True)
        else:
            st.info("No co-occurring items found.")
    else:
        st.info("No transactions available for bundles.")

st.caption("📘 Dashboard Tabs: Raw Data | Unique Items & Customers | Transactions | Top Items | Customer Behavior | Seasonal Trends | Co-occurrence | Recommendations | Top Bundles")
