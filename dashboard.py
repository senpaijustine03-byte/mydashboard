import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Groceries Transaction Dashboard",
    layout="wide"
)

st.title("🛒 Groceries Transaction Dataset Dashboard")
st.markdown("Actionable insights on grocery transactions")

# --------------------------------
# Load Data
# --------------------------------
@st.cache_data
def load_data():
    basket = pd.read_csv("groceries_basket.csv", index_col=0)
    rules = pd.read_csv("association_rules.csv")
    # Normalize column names
    rules.columns = rules.columns.str.strip().str.lower()
    return basket, rules

basket, rules = load_data()

# --------------------------------
# KPI METRICS
# --------------------------------
st.subheader("📌 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", basket.shape[0])
c2.metric("Unique Items", basket.shape[1])
c3.metric("Association Rules", rules.shape[0])

st.divider()

# --------------------------------
# Top-Selling Items
# --------------------------------
st.subheader("📦 Top-Selling Items")

item_counts = basket.sum().sort_values(ascending=False)
top_n = st.slider("Top items to display", 5, 30, 15)

fig1, ax1 = plt.subplots(figsize=(10, 5))
item_counts.head(top_n).plot(kind="bar", ax=ax1, color='skyblue')
ax1.set_ylabel("Number of Transactions")
ax1.set_title(f"Top {top_n} Most Purchased Items")
st.pyplot(fig1)

st.divider()

# --------------------------------
# Basket Size Analysis
# --------------------------------
st.subheader("🛒 Basket Size Analysis")
basket_sizes = basket.sum(axis=1)  # number of items per transaction

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.histplot(basket_sizes, bins=20, kde=True, color='orange', ax=ax2)
ax2.set_xlabel("Number of Items per Basket")
ax2.set_ylabel("Number of Transactions")
ax2.set_title("Distribution of Basket Sizes")
st.pyplot(fig2)

st.write(f"Average basket size: **{basket_sizes.mean():.2f}** items per transaction")

st.divider()

# --------------------------------
# Item Pair Heatmap (Co-occurrence)
# --------------------------------
st.subheader("📊 Item Co-occurrence Heatmap (Top Items)")

# Select top items for heatmap
heatmap_top_n = st.slider("Number of top items to include in heatmap", 5, 30, 15)
top_items = item_counts.head(heatmap_top_n).index.tolist()
basket_top = basket[top_items]

# Compute co-occurrence matrix
co_occurrence = basket_top.T.dot(basket_top)
# Normalize to percentage of transactions
co_occurrence_pct = co_occurrence / basket.shape[0] * 100

fig3, ax3 = plt.subplots(figsize=(12, 10))
sns.heatmap(co_occurrence_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3)
ax3.set_title(f"Item Co-occurrence Heatmap (% of transactions)")
st.pyplot(fig3)

st.divider()

# --------------------------------
# Optional: Show association rules filtered
# --------------------------------
st.subheader("📜 Association Rules Overview")
min_support = st.slider("Minimum Support", 0.0, 1.0, 0.01, 0.01)
min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.05)
min_lift = st.slider("Minimum Lift", 0.0, 5.0, 1.0, 0.1)

filtered_rules = rules[
    (rules["support"] >= min_support) &
    (rules["confidence"] >= min_confidence) &
    (rules["lift"] >= min_lift)
]

st.write(f"📊 Showing **{len(filtered_rules)}** rules")
st.dataframe(filtered_rules, use_container_width=True)

st.divider()
st.caption("📘 Practical insights: Top-selling items, basket size distribution, and item co-occurrence heatmap")
