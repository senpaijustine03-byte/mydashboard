import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Groceries Dashboard", layout="wide")
st.title("🛒 Groceries Transaction Dashboard")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    basket = pd.read_csv("groceries_basket.csv", index_col=0)
    rules = pd.read_csv("association_rules.csv")
    rules.columns = rules.columns.str.strip().str.lower()
    return basket, rules

basket, rules = load_data()

# -------------------------
# KPI Metrics
# -------------------------
st.subheader("📌 Key Metrics")
total_transactions = basket.shape[0]
total_items = basket.shape[1]
avg_basket_size = basket.sum(axis=1).mean()

c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", total_transactions)
c2.metric("Total Items", total_items)
c3.metric("Average Basket Size", f"{avg_basket_size:.2f}")

st.divider()

# -------------------------
# Top Items by Frequency
# -------------------------
st.subheader("📦 Top Items by Frequency")
item_counts = basket.sum().sort_values(ascending=False)
top_n = st.slider("Select number of top items to show", 5, 30, 15)

fig1, ax1 = plt.subplots(figsize=(10,5))
item_counts.head(top_n).plot(kind="bar", ax=ax1, color="skyblue")
ax1.set_ylabel("Frequency")
ax1.set_title(f"Top {top_n} Items Purchased")
st.pyplot(fig1)

st.divider()

# -------------------------
# Basket Size Distribution
# -------------------------
st.subheader("🛒 Basket Size Distribution")
basket_sizes = basket.sum(axis=1)

fig2, ax2 = plt.subplots(figsize=(10,5))
sns.histplot(basket_sizes, bins=20, kde=True, color="orange", ax=ax2)
ax2.set_xlabel("Number of Items per Basket")
ax2.set_ylabel("Number of Transactions")
ax2.set_title("Basket Size Distribution")
st.pyplot(fig2)

st.write(f"Average basket size: **{avg_basket_size:.2f}** items per transaction")
st.divider()

# -------------------------
# Item Pair Co-occurrence Heatmap
# -------------------------
st.subheader("📊 Item Co-occurrence Heatmap")
heatmap_top_n = st.slider("Top items for co-occurrence", 5, 30, 15)
top_items = item_counts.head(heatmap_top_n).index.tolist()
basket_top = basket[top_items]

co_occurrence = basket_top.T.dot(basket_top)
co_occurrence_pct = co_occurrence / basket.shape[0] * 100

fig3, ax3 = plt.subplots(figsize=(12,10))
sns.heatmap(co_occurrence_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3)
ax3.set_title("Item Co-occurrence (% of transactions)")
st.pyplot(fig3)

st.divider()

# -------------------------
# Market Basket Recommendations
# -------------------------
st.subheader("🛍️ Market Basket Recommendations")
selected_items = st.multiselect("Select items in your basket:", options=basket.columns)

if selected_items:
    # Filter rules with selected items in antecedents
    recommended_rules = rules[
        rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))
    ]
    sort_cols = [col for col in ["confidence", "lift"] if col in recommended_rules.columns]
    if sort_cols:
        recommended_rules = recommended_rules.sort_values(by=sort_cols, ascending=False)

    if not recommended_rules.empty:
        top_recs = recommended_rules.head(10)
        st.write(f"**Top recommendations for ({', '.join(selected_items)}):**")
        st.dataframe(top_recs[["antecedents","consequents","support","confidence","lift"]], use_container_width=True)

        suggested_items = set()
        for cons in top_recs["consequents"]:
            suggested_items.update(cons.split(", "))
        suggested_items = [item for item in suggested_items if item not in selected_items]
        st.markdown(f"**Suggested items:** {', '.join(suggested_items)}")
    else:
        st.info("No recommendations found for the selected items.")
else:
    st.info("Select items from the basket to get recommendations.")

st.caption("📘 Dashboard: Basket Analysis | Co-occurrence Heatmap | Recommendations")
