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
    data['timestamp'] = pd.to_datetime(data['Date'], errors='coerce')
    data['order_id'] = data['Member_number']
    data = data.dropna(subset=['timestamp'])
    return data

data = load_data()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Filters")
min_date = data['timestamp'].min()
max_date = data['timestamp'].max()
date_range = st.sidebar.date_input("Select date range:", [min_date, max_date])

item_cols = data['itemDescription'].dropna().unique().tolist()
selected_items_sidebar = st.sidebar.multiselect(
    "Filter by items (optional):",
    options=item_cols,
    default=[]
)

# Apply filters
filtered_data = data[
    (data['timestamp'].dt.date >= date_range[0]) &
    (data['timestamp'].dt.date <= date_range[1])
]

if selected_items_sidebar:
    filtered_data = filtered_data[filtered_data['itemDescription'].isin(selected_items_sidebar)]

if filtered_data.empty:
    st.warning("No data available for the selected filters. Please adjust your date range or items.")
    st.stop()  # Stop execution to avoid errors downstream

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
    if basket_oh.empty or basket_oh.sum().sum() == 0:
        return pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift'])
    frequent_itemsets = apriori(basket_oh, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift'])
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    if rules.empty:
        return rules
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
# 6️⃣ Item Co-occurrence (Safe)
# -------------------------
with tabs[6]:
    st.subheader("📊 Top 20 Item Co-occurrence Heatmap")
    if basket_oh.empty:
        st.info("Not enough data to generate co-occurrence heatmap.")
    else:
        basket_items = basket_oh.astype(float)
        top_items = basket_items.sum().sort_values(ascending=False).head(20).index
        co_occurrence = basket_items[top_items].T.dot(basket_items[top_items])
        co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
        fig5, ax5 = plt.subplots(figsize=(12,10))
        sns.heatmap(co_occurrence_pct, annot=False, cmap="YlGnBu", ax=ax5)
        ax5.set_title("Item Co-occurrence (% of transactions)")
        st.pyplot(fig5)

# -------------------------
# 7️⃣ Basket Recommendations (Safe)
# -------------------------
with tabs[7]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols_filtered)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3)
    min_lift = st.slider("Minimum lift", 0.0, 5.0, 1.0)

    if selected_items:
        if rules.empty:
            st.info("No association rules available for the current data selection.")
        else:
            recommended_rules = rules[
                (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
                (rules["confidence"] >= min_conf) &
                (rules["lift"] >= min_lift)
            ]
            if not recommended_rules.empty:
                top_recs = recommended_rules.sort_values(by=['confidence','lift'], ascending=False).head(10)
                st.dataframe(top_recs[['antecedents','consequents','support','confidence','lift']])
            else:
                st.info("No recommendations found for the selected items.")
    else:
        st.info("Select items from the basket to get recommendations.")
