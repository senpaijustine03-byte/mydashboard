import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Groceries Analytics Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard")

# -------------------------
# Load and preprocess data
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Groceries_dataset.csv")  # Kaggle dataset
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
tabs = st.tabs(["Transactions Overview", "Customer Behavior", "Product Performance", "Basket Analysis"])

# -------------------------
# 1️⃣ Transactions Overview
# -------------------------
with tabs[0]:
    st.subheader("📈 Transactions Over Time")
    time_group = st.radio("Aggregate by:", ["Daily", "Monthly", "Yearly"])
    if time_group == "Daily":
        trans_time = data.groupby(data['timestamp'].dt.date)['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = pd.to_datetime(trans_time['timestamp'])
        x_col = 'timestamp'
    elif time_group == "Monthly":
        trans_time = data.groupby(data['timestamp'].dt.to_period("M"))['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = trans_time['timestamp'].astype(str)
        x_col = 'timestamp'
    else:
        trans_time = data.groupby(data['timestamp'].dt.to_period("Y"))['order_id'].count().reset_index()
        trans_time.rename(columns={'order_id':'transactions'}, inplace=True)
        trans_time['timestamp'] = trans_time['timestamp'].astype(str)
        x_col = 'timestamp'

    fig = px.line(trans_time, x=x_col, y='transactions', title="Transactions Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛒 Top Items by Frequency")
    item_freq = basket_oh.sum().sort_values(ascending=False)
    top_n = st.slider("Top items to show:", 5, 30, 10)
    fig2 = px.bar(item_freq.head(top_n), x=item_freq.head(top_n).index, y=item_freq.head(top_n).values,
                  labels={'x':'Item','y':'Count'}, title="Top Items by Transaction Count")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# 2️⃣ Customer Behavior
# -------------------------
with tabs[1]:
    st.subheader("👥 Customer Behavior")
    cust_orders = data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig3 = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                  title="Repeat vs First-time Customers")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# 3️⃣ Product Performance
# -------------------------
with tabs[2]:
    st.subheader("📊 Item Co-occurrence Heatmap")
    basket_items = basket_oh.astype(float)
    co_occurrence = basket_items.T.dot(basket_items)
    co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
    fig4, ax4 = plt.subplots(figsize=(12,10))
    sns.heatmap(co_occurrence_pct, annot=False, fmt=".1f", cmap="YlGnBu", ax=ax4)
    ax4.set_title("Item Co-occurrence (% of transactions)")
    st.pyplot(fig4)

# -------------------------
# 4️⃣ Market Basket Recommendations
# -------------------------
with tabs[3]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols)

    if selected_items:
        recommended_rules = rules[
            rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))
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

st.caption("📘 Dashboard: Transactions | Customer Behavior | Product Performance | Basket Recommendations")
