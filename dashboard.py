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
# One-hot encode basket
# -------------------------
basket_oh = data.groupby(['Member_number', 'itemDescription'])['itemDescription']\
                .count().unstack().fillna(0)
basket_oh = basket_oh.applymap(lambda x: 1 if x > 0 else 0)
item_cols = basket_oh.columns.tolist()

# -------------------------
# Generate association rules
# -------------------------
@st.cache_data
def generate_rules(basket_oh):
    frequent_itemsets = apriori(basket_oh, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
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
    st.dataframe(data.head(50))
    st.write(f"Total rows: {len(data)}")

# -------------------------
# 1️⃣ Unique Items & Customers
# -------------------------
with tabs[1]:
    st.subheader("🛒 All Unique Items")
    st.dataframe(pd.DataFrame(data['itemDescription'].unique(), columns=["ItemDescription"]))
    st.write(f"Total unique items: {data['itemDescription'].nunique()}")

    st.subheader("👥 All Unique Customers")
    st.dataframe(pd.DataFrame(data['Member_number'].unique(), columns=["CustomerID"]))
    st.write(f"Total unique customers: {data['Member_number'].nunique()}")

# -------------------------
# 2️⃣ Transactions Overview (Fixed Date Aggregation)
# -------------------------
with tabs[2]:
    st.subheader("📈 Transactions Over Time")
    time_group = st.radio("Aggregate by:", ["Daily", "Weekly", "Monthly"])

    filtered_data = data.copy()
    
    if time_group == "Daily":
        trans_time = filtered_data.groupby(filtered_data['timestamp'].dt.date)['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = pd.to_datetime(trans_time['timestamp'])
        x_col = 'timestamp'

    elif time_group == "Weekly":
        filtered_data['week_start'] = filtered_data['timestamp'] - pd.to_timedelta(filtered_data['timestamp'].dt.weekday, unit='d')
        trans_time = filtered_data.groupby('week_start')['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions','week_start':'timestamp'}, inplace=True)
        x_col = 'timestamp'

    else:  # Monthly
        filtered_data['month'] = filtered_data['timestamp'].dt.to_period('M').dt.to_timestamp()
        trans_time = filtered_data.groupby('month')['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions','month':'timestamp'}, inplace=True)
        x_col = 'timestamp'

    fig = px.line(trans_time, x=x_col, y='transactions', title="Transactions Over Time", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Peak Transaction Days")
    peak_days = filtered_data['timestamp'].dt.day_name().value_counts()
    fig_peak = px.bar(peak_days, x=peak_days.index, y=peak_days.values,
                      labels={'x':'Day of Week','y':'Transactions'}, title="Transactions by Day of Week", template="plotly_white")
    st.plotly_chart(fig_peak, use_container_width=True)

# -------------------------
# 3️⃣ Top Items
# -------------------------
with tabs[3]:
    st.subheader("🛒 Top Items by Frequency")
    item_freq = basket_oh.sum().sort_values(ascending=False)
    top_n = st.slider("Top items to show:", 5, 30, 10)
    fig2 = px.bar(item_freq.head(top_n), x=item_freq.head(top_n).index, y=item_freq.head(top_n).values,
                  labels={'x':'Item','y':'Count'}, title="Top Items by Transaction Count", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Item Popularity Over Time")
    selected_item = st.selectbox("Select an item to see trends:", item_cols)
    item_trend = data[data['itemDescription'] == selected_item].groupby(data['timestamp'].dt.to_period("M"))['order_id'].count()
    fig_item_trend = px.line(item_trend, x=item_trend.index.astype(str), y=item_trend.values,
                             labels={'x':'Month','y':'Transactions'}, title=f"{selected_item} Monthly Trend", template="plotly_white")
    st.plotly_chart(fig_item_trend, use_container_width=True)

# -------------------------
# 4️⃣ Customer Behavior
# -------------------------
with tabs[4]:
    st.subheader("👥 Customer Behavior")
    cust_orders = data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig3 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                  title="Repeat vs First-time Customers", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🏆 Top Buyers")
    top_buyers = cust_orders.sort_values(ascending=False).head(10)
    fig_top_buyers = px.bar(top_buyers, x=top_buyers.index, y=top_buyers.values,
                            labels={'x':'CustomerID','y':'Number of Transactions'}, title="Top 10 Buyers", template="plotly_white")
    st.plotly_chart(fig_top_buyers, use_container_width=True)

    st.subheader("🛒 Average Items per Transaction")
    avg_items = basket_oh.sum(axis=1).mean()
    st.metric("Average items per transaction", f"{avg_items:.2f}")

# -------------------------
# 5️⃣ Seasonal Trends
# -------------------------
with tabs[5]:
    st.subheader("📅 Monthly Seasonal Trends")
    monthly_sales = data.groupby(data['timestamp'].dt.month)['order_id'].count()
    fig4 = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values,
                   labels={'x':'Month', 'y':'Number of Transactions'}, title="Monthly Transaction Trend", template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# 6️⃣ Item Co-occurrence
# -------------------------
with tabs[6]:
    st.subheader("📊 Top 20 Item Co-occurrence Heatmap")
    basket_items = basket_oh.astype(float)
    top_items = basket_items.sum().sort_values(ascending=False).head(20).index
    co_occurrence = basket_items[top_items].T.dot(basket_items[top_items])
    co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
    fig5, ax5 = plt.subplots(figsize=(12,10))
    sns.heatmap(co_occurrence_pct, annot=False, fmt=".1f", cmap="YlGnBu", ax=ax5)
    ax5.set_title("Item Co-occurrence (% of transactions)")
    st.pyplot(fig5)

# -------------------------
# 7️⃣ Basket Recommendations
# -------------------------
with tabs[7]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3)
    min_lift = st.slider("Minimum lift", 0.0, 5.0, 1.0)

    if selected_items:
        recommended_rules = rules[
            (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ]
        if not recommended_rules.empty:
            top_recs = recommended_rules.sort_values(by=['confidence','lift'], ascending=False).head(10)
            st.write(f"**Top recommendations for ({', '.join(selected_items)}):**")
            st.dataframe(top_recs[['antecedents','consequents','support','confidence','lift']])
            suggested_items = set()
            for cons in top_recs['consequents']:
                suggested_items.update(cons.split(", "))
            suggested_items = [item for item in suggested_items if item not in selected_items]
            st.markdown(f"**Suggested items:** {', '.join(suggested_items)}")
        else:
            st.info("No recommendations found for the selected items.")
    else:
        st.info("Select items from the basket to get recommendations.")

# -------------------------
# 8️⃣ Top Bundles & Item Combos
# -------------------------
with tabs[8]:
    st.subheader("📦 Top Bundles & Frequently Bought Together")
    selected_item_combo = st.selectbox("Select an item to see popular bundles:", options=item_cols)
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
            labels={'x':'Item', 'y':'Number of Co-occurrences'},
            title=f"Top {top_n_bundles} items bought together with '{selected_item_combo}'",
            template="plotly_white"
        )
        st.plotly_chart(fig_bundle, use_container_width=True)
    else:
        st.info(f"No co-occurring items found for '{selected_item_combo}'.")

st.caption("📘 Dashboard: Raw Data | Unique Items & Customers | Transactions | Top Items | Customer Behavior | Seasonal Trends | Co-occurrence | Recommendations | Top Bundles & Item Combos")
