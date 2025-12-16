import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Groceries Analytics Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard (Kaggle Dataset)")
st.markdown("Clean, interactive dashboard with insights from the Groceries dataset.")

# -------------------------
# Load Data
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
min_date = data['timestamp'].min()
max_date = data['timestamp'].max()
date_range = st.sidebar.date_input("Select date range:", [min_date, max_date])
item_cols = data['itemDescription'].unique().tolist()
selected_items_sidebar = st.sidebar.multiselect("Filter by items:", item_cols, default=[])

# Apply filters
filtered_data = data[(data['timestamp'].dt.date >= date_range[0]) &
                     (data['timestamp'].dt.date <= date_range[1])]
if selected_items_sidebar:
    filtered_data = filtered_data[filtered_data['itemDescription'].isin(selected_items_sidebar)]

# -------------------------
# One-hot encode basket
# -------------------------
basket_oh = filtered_data.groupby(['Member_number', 'itemDescription'])['itemDescription']\
            .count().unstack().fillna(0)
basket_oh = basket_oh.applymap(lambda x: 1 if x>0 else 0)
item_cols_filtered = basket_oh.columns.tolist()

# -------------------------
# Generate Association Rules
# -------------------------
@st.cache_data
def generate_rules(basket):
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules

rules = generate_rules(basket_oh)

# -------------------------
# Key Metrics
# -------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", len(filtered_data['order_id'].unique()))
col2.metric("Total Customers", filtered_data['Member_number'].nunique())
col3.metric("Total Unique Items", filtered_data['itemDescription'].nunique())
col4.metric("Avg Items / Transaction", f"{basket_oh.sum(axis=1).mean():.2f}")
st.markdown("---")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Raw Data", "Transactions Overview", "Top Items", 
    "Customer Behavior", "Seasonal Trends", "Item Co-occurrence",
    "Basket Recommendations", "Top Bundles", "Customer Insights"
])

# -------------------------
# 0️⃣ Raw Data
# -------------------------
with tabs[0]:
    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(filtered_data.head(50))
    st.write(f"Total rows: {len(filtered_data)}")

# -------------------------
# 1️⃣ Transactions Overview
# -------------------------
with tabs[1]:
    st.subheader("📈 Transactions Over Time")
    col1, col2 = st.columns(2)
    time_group = st.radio("Aggregate by:", ["Daily", "Weekly", "Monthly"], horizontal=True)

    if time_group=="Daily":
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.date)['order_id'].count()
    elif time_group=="Weekly":
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.isocalendar().week)['order_id'].count()
    else:
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.to_period("M"))['order_id'].count()

    fig = px.line(trans_time, x=trans_time.index.astype(str), y=trans_time.values,
                  labels={'x':'Time','y':'Transactions'},
                  title=f"Transactions ({time_group})", template="plotly_white")
    col1.plotly_chart(fig, use_container_width=True)

    # Peak transaction days
    peak_days = filtered_data['timestamp'].dt.day_name().value_counts()
    fig_peak = px.bar(peak_days, x=peak_days.index, y=peak_days.values,
                      labels={'x':'Day','y':'Transactions'},
                      title="Transactions by Day of Week", template="plotly_white")
    col2.plotly_chart(fig_peak, use_container_width=True)

# -------------------------
# 2️⃣ Top Items
# -------------------------
with tabs[2]:
    st.subheader("🛒 Top Items by Frequency")
    col1, col2 = st.columns(2)
    item_freq = basket_oh.sum().sort_values(ascending=False)
    top_n = st.slider("Top items to show:", 5, 30, 10)
    fig_items = px.bar(item_freq.head(top_n), x=item_freq.head(top_n).index, y=item_freq.head(top_n).values,
                       labels={'x':'Item','y':'Count'}, title="Top Items", template="plotly_white")
    col1.plotly_chart(fig_items, use_container_width=True)

    # Item popularity over time
    selected_item = st.selectbox("Select an item to see trend:", item_cols_filtered)
    item_trend = filtered_data[filtered_data['itemDescription']==selected_item].groupby(filtered_data['timestamp'].dt.to_period("M"))['order_id'].count()
    fig_trend = px.line(item_trend, x=item_trend.index.astype(str), y=item_trend.values,
                        labels={'x':'Month','y':'Transactions'}, title=f"{selected_item} Monthly Trend", template="plotly_white")
    col2.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# 3️⃣ Customer Behavior
# -------------------------
with tabs[3]:
    st.subheader("👥 Customer Behavior Overview")
    col1, col2 = st.columns(2)
    cust_orders = filtered_data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig_repeat = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                        title="Repeat vs First-time Customers", template="plotly_white")
    col1.plotly_chart(fig_repeat, use_container_width=True)

    top_buyers = cust_orders.sort_values(ascending=False).head(10)
    fig_top_buyers = px.bar(top_buyers, x=top_buyers.index, y=top_buyers.values,
                            labels={'x':'CustomerID','y':'Transactions'},
                            title="Top 10 Customers", template="plotly_white")
    col2.plotly_chart(fig_top_buyers, use_container_width=True)

# -------------------------
# 4️⃣ Seasonal Trends
# -------------------------
with tabs[4]:
    st.subheader("📅 Monthly Transaction Trends")
    monthly_sales = filtered_data.groupby(filtered_data['timestamp'].dt.month)['order_id'].count()
    fig_monthly = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values,
                          labels={'x':'Month','y':'Transactions'}, title="Monthly Trend", template="plotly_white")
    st.plotly_chart(fig_monthly, use_container_width=True)

# -------------------------
# 5️⃣ Item Co-occurrence
# -------------------------
with tabs[5]:
    st.subheader("📊 Top 20 Item Co-occurrence Heatmap")
    basket_items = basket_oh.astype(float)
    top_items = basket_items.sum().sort_values(ascending=False).head(20).index
    co_occurrence = basket_items[top_items].T.dot(basket_items[top_items])
    co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
    st.write("Heatmap of top items bought together (% of transactions)")
    fig_heatmap = px.imshow(co_occurrence_pct, text_auto=True, aspect="auto",
                            labels=dict(x="Item", y="Item", color="% Co-occurrence"),
                            title="Item Co-occurrence Heatmap", template="plotly_white")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------
# 6️⃣ Basket Recommendations
# -------------------------
with tabs[6]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols_filtered)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3)
    min_lift = st.slider("Minimum lift", 0.0, 5.0, 1.0)

    if selected_items and not rules.empty:
        recommended_rules = rules[
            (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ]
        if not recommended_rules.empty:
            top_recs = recommended_rules.sort_values(by=['confidence','lift'], ascending=False).head(10)
            st.dataframe(top_recs[['antecedents','consequents','support','confidence','lift']])
            suggested_items = set()
            for cons in top_recs['consequents']:
                suggested_items.update(cons.split(", "))
            suggested_items = [item for item in suggested_items if item not in selected_items]
            st.markdown(f"**Suggested items:** {', '.join(suggested_items)}")
        else:
            st.info("No recommendations found for selected items.")
    else:
        st.info("Select items to get recommendations.")

# -------------------------
# 7️⃣ Top Bundles
# -------------------------
with tabs[7]:
    st.subheader("📦 Top Bundles & Frequently Bought Together")
    selected_item_combo = st.selectbox("Select an item:", options=item_cols_filtered)
    top_n_bundles = st.slider("Top bundles to show:", 3, 15, 5)
    transactions_with_item = basket_oh[basket_oh[selected_item_combo]==1]
    co_occurring_counts = transactions_with_item.sum().sort_values(ascending=False)
    co_occurring_counts = co_occurring_counts.drop(labels=[selected_item_combo])
    if not co_occurring_counts.empty:
        top_co_occurring = co_occurring_counts.head(top_n_bundles)
        fig_bundle = px.bar(top_co_occurring, x=top_co_occurring.index, y=top_co_occurring.values,
                            labels={'x':'Item','y':'Co-occurrences'},
                            title=f"Top {top_n_bundles} items with {selected_item_combo}",
                            template="plotly_white")
        st.plotly_chart(fig_bundle, use_container_width=True)
    else:
        st.info(f"No co-occurring items found for '{selected_item_combo}'.")

# -------------------------
# 8️⃣ Customer Insights
# -------------------------
with tabs[8]:
    st.subheader("👤 Top Customer Insights")
    cust_orders = filtered_data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig_repeat = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                        title="Repeat vs First-time Customers", template="plotly_white")
    st.plotly_chart(fig_repeat, use_container_width=True)

    avg_items_customer = basket_oh.sum(axis=1).mean()
    st.metric("Average items per customer transaction", f"{avg_items_customer:.2f}")

    top_customers = cust_orders.sort_values(ascending=False).head(10)
    fig_top_cust = px.bar(top_customers, x=top_customers.index, y=top_customers.values,
                          labels={'x':'CustomerID','y':'Transactions'},
                          title="Top 10 Customers", template="plotly_white")
    st.plotly_chart(fig_top_cust, use_container_width=True)

    unique_items_per_cust = filtered_data.groupby('Member_number')['itemDescription'].nunique()
    avg_unique_items = unique_items_per_cust.mean()
    st.metric("Average unique items per customer", f"{avg_unique_items:.2f}")
