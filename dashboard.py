import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Groceries Dashboard", layout="wide")
st.title("🛍️ Groceries Analytics Dashboard")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Groceries_dataset.csv")
    data['timestamp'] = pd.to_datetime(data['Date'])
    data['order_id'] = data['Member_number']
    return data

data = load_data()

# One-hot encode for basket analysis
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
# Tabs
# -------------------------
tabs = st.tabs([
    "Overview", "Top Items", "Customer Behavior", 
    "Item Co-occurrence", "Basket Recommendations"
])

# -------------------------
# 1️⃣ Overview Tab
# -------------------------
with tabs[0]:
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(data['order_id'].unique()))
    col2.metric("Total Customers", data['Member_number'].nunique())
    col3.metric("Total Unique Items", data['itemDescription'].nunique())

    st.subheader("📈 Monthly Transactions")
    trans_monthly = data.groupby(data['timestamp'].dt.to_period("M"))['order_id'].count()
    fig = px.line(trans_monthly, x=trans_monthly.index.astype(str), y=trans_monthly.values,
                  labels={'x':'Month','y':'Transactions'}, title="")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 2️⃣ Top Items Tab
# -------------------------
with tabs[1]:
    st.subheader("🛒 Top Items")
    top_n = st.slider("Top items to show:", 5, 20, 10)
    item_freq = basket_oh.sum().sort_values(ascending=False).head(top_n)
    fig_top_items = px.bar(item_freq, x=item_freq.index, y=item_freq.values,
                           labels={'x':'Item','y':'Count'}, title="")
    fig_top_items.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig_top_items, use_container_width=True)

# -------------------------
# 3️⃣ Customer Behavior Tab
# -------------------------
with tabs[2]:
    st.subheader("👥 Customer Analysis")
    cust_orders = data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig_cust = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values)
    fig_cust.update_layout(template="plotly_white")
    st.plotly_chart(fig_cust, use_container_width=True)

# -------------------------
# 4️⃣ Item Co-occurrence Tab
# -------------------------
with tabs[3]:
    st.subheader("📊 Top Item Co-occurrence")
    top_items = basket_oh.sum().sort_values(ascending=False).head(20).index
    co_occurrence = basket_oh[top_items].T.dot(basket_oh[top_items])
    co_occurrence_pct = (co_occurrence / basket_oh.shape[0] * 100).astype(float)
    st.dataframe(co_occurrence_pct.round(2))

# -------------------------
# 5️⃣ Basket Recommendations Tab
# -------------------------
with tabs[4]:
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
            st.dataframe(top_recs[['antecedents','consequents','support','confidence','lift']])
        else:
            st.info("No recommendations found for selected items.")
    else:
        st.info("Select items from the basket to get recommendations.")
