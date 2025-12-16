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
st.title("🛍️ Groceries Analytics Dashboard")
st.markdown("Clean, interactive dashboard for market basket analysis with top metrics and insights.")

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

# One-hot encode basket
basket_oh = data.groupby(['Member_number', 'itemDescription'])['itemDescription'].count().unstack().fillna(0)
basket_oh = basket_oh.applymap(lambda x: 1 if x > 0 else 0)
item_cols = basket_oh.columns.tolist()

# Generate association rules
@st.cache_data
def generate_rules(basket_oh):
    frequent_itemsets = apriori(basket_oh, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules

rules = generate_rules(basket_oh)

# -------------------------
# Consistent color palette
# -------------------------
PRIMARY_COLOR = "#1f77b4"  # soft blue
SECONDARY_COLOR = "#2ca02c"  # soft green

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Overview", "Top Items", "Customer Behavior",
    "Seasonal Trends", "Item Co-occurrence", 
    "Basket Recommendations", "Top Bundles"
])

# -------------------------
# 1️⃣ Overview
# -------------------------
with tabs[0]:
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(data['order_id'].unique()))
    col2.metric("Total Customers", data['Member_number'].nunique())
    col3.metric("Total Unique Items", data['itemDescription'].nunique())
    st.markdown("---")

    st.subheader("📈 Transactions Over Time")
    trans_monthly = data.groupby(data['timestamp'].dt.to_period("M"))['order_id'].count()
    fig = px.line(trans_monthly, x=trans_monthly.index.astype(str), y=trans_monthly.values,
                  labels={'x':'Month','y':'Transactions'}, title="")
    fig.update_layout(template="plotly_white", line_color=PRIMARY_COLOR)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 2️⃣ Top Items
# -------------------------
with tabs[1]:
    st.subheader("🛒 Top Items by Frequency")
    top_n = st.slider("Top items to show:", 5, 20, 10)
    item_freq = basket_oh.sum().sort_values(ascending=False).head(top_n)
    fig_top_items = px.bar(item_freq, x=item_freq.index, y=item_freq.values,
                           labels={'x':'Item','y':'Count'}, title="")
    fig_top_items.update_layout(template="plotly_white", coloraxis_showscale=False)
    fig_top_items.update_traces(marker_color=SECONDARY_COLOR)
    st.plotly_chart(fig_top_items, use_container_width=True)
    st.markdown("---")

    st.subheader("📈 Item Trend Over Time")
    selected_item = st.selectbox("Select an item:", item_cols)
    item_trend = data[data['itemDescription'] == selected_item]\
                    .groupby(data['timestamp'].dt.to_period("M"))['order_id'].count()
    fig_trend = px.line(item_trend, x=item_trend.index.astype(str), y=item_trend.values,
                        labels={'x':'Month','y':'Transactions'}, title=f"{selected_item} Trend")
    fig_trend.update_layout(template="plotly_white", line_color=PRIMARY_COLOR)
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# 3️⃣ Customer Behavior
# -------------------------
with tabs[2]:
    st.subheader("👥 Customer Behavior")
    cust_orders = data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig_cust = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values)
    fig_cust.update_layout(template="plotly_white", legend_title_text="Customer Type")
    st.plotly_chart(fig_cust, use_container_width=True)

    st.subheader("🏆 Top Buyers")
    top_buyers = cust_orders.sort_values(ascending=False).head(10)
    fig_top_buyers = px.bar(top_buyers, x=top_buyers.index, y=top_buyers.values,
                            labels={'x':'CustomerID','y':'Transactions'})
    fig_top_buyers.update_layout(template="plotly_white", marker_color=PRIMARY_COLOR)
    st.plotly_chart(fig_top_buyers, use_container_width=True)
    st.markdown("---")

# -------------------------
# 4️⃣ Seasonal Trends
# -------------------------
with tabs[3]:
    st.subheader("📅 Monthly Transactions Trend")
    monthly_sales = data.groupby(data['timestamp'].dt.month)['order_id'].count()
    fig_monthly = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values,
                          labels={'x':'Month','y':'Transactions'})
    fig_monthly.update_layout(template="plotly_white", line_color=SECONDARY_COLOR)
    st.plotly_chart(fig_monthly, use_container_width=True)

# -------------------------
# 5️⃣ Item Co-occurrence
# -------------------------
with tabs[4]:
    st.subheader("📊 Top Item Co-occurrence")
    top_items = basket_oh.sum().sort_values(ascending=False).head(20).index
    co_occurrence = basket_oh[top_items].T.dot(basket_oh[top_items])
    co_occurrence_pct = (co_occurrence / basket_oh.shape[0] * 100).astype(float)
    fig_co, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(co_occurrence_pct, annot=False, cmap="YlGnBu", ax=ax)
    st.pyplot(fig_co)

# -------------------------
# 6️⃣ Basket Recommendations
# -------------------------
with tabs[5]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items:", options=item_cols)
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.3)
    min_lift = st.slider("Min lift", 0.0, 5.0, 1.0)

    if selected_items:
        rec_rules = rules[
            (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ]
        if not rec_rules.empty:
            top_rec = rec_rules.sort_values(by=['confidence','lift'], ascending=False).head(10)
            st.dataframe(top_rec[['antecedents','consequents','support','confidence','lift']])
        else:
            st.info("No recommendations found.")
    else:
        st.info("Select items to get recommendations.")

# -------------------------
# 7️⃣ Top Bundles
# -------------------------
with tabs[6]:
    st.subheader("📦 Top Bundles & Frequently Bought Together")
    selected_item_combo = st.selectbox("Select item:", options=item_cols)
    top_n_bundles = st.slider("Top bundles to show:", 3, 15, 5)

    transactions_with_item = basket_oh[basket_oh[selected_item_combo] == 1]
    co_occurring_counts = transactions_with_item.sum().sort_values(ascending=False)
    co_occurring_counts = co_occurring_counts.drop(labels=[selected_item_combo])

    if not co_occurring_counts.empty:
        top_co_occurring = co_occurring_counts.head(top_n_bundles)
        fig_bundle = px.bar(
            top_co_occurring,
            x=top_co_occurring.index,
            y=top_co_occurring.values,
            labels={'x':'Item','y':'Count'},
            title=f"Top {top_n_bundles} items bought with '{selected_item_combo}'"
        )
        fig_bundle.update_layout(template="plotly_white", marker_color=SECONDARY_COLOR)
        st.plotly_chart(fig_bundle, use_container_width=True)
    else:
        st.info("No co-occurring items found.")
