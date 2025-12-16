import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Groceries Transaction Dashboard",
    layout="wide"
)

st.title("🛒 Groceries Transaction Dataset Dashboard")
st.markdown("Interactive analysis of grocery transactions using Association Rule Mining")

# --------------------------------
# Load Data
# --------------------------------
@st.cache_data
def load_data():
    basket = pd.read_csv("groceries_basket.csv", index_col=0)
    rules = pd.read_csv("association_rules.csv")
    return basket, rules

basket, rules = load_data()

# --------------------------------
# Clean rule text
# --------------------------------
for col in ["antecedents", "consequents"]:
    rules[col] = (
        rules[col]
        .astype(str)
        .str.replace("frozenset({", "", regex=False)
        .str.replace("})", "", regex=False)
        .str.replace("'", "", regex=False)
    )

# --------------------------------
# KPI METRICS
# --------------------------------
st.subheader("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", basket.shape[0])
col2.metric("Total Unique Items", basket.shape[1])
col3.metric("Association Rules", rules.shape[0])

st.divider()

# --------------------------------
# Item Frequency Analysis
# --------------------------------
st.subheader("📦 Item Frequency Analysis")

item_counts = basket.sum().sort_values(ascending=False)

top_n = st.slider("Select number of top items", 5, 30, 15)

fig1, ax1 = plt.subplots(figsize=(10, 5))
item_counts.head(top_n).plot(kind="bar", ax=ax1)
ax1.set_ylabel("Frequency")
ax1.set_title(f"Top {top_n} Most Purchased Items")
st.pyplot(fig1)

# --------------------------------
# Item Search
# --------------------------------
st.subheader("🔍 Search Item Frequency")

item_name = st.selectbox("Select an item", item_counts.index)

st.write(
    f"**{item_name}** appears in "
    f"**{int(item_counts[item_name])}** transactions "
    f"({item_counts[item_name] / basket.shape[0]:.2%})"
)

st.divider()

# --------------------------------
# Association Rules Table
# --------------------------------
st.subheader("📊 Association Rules Explorer")

with st.expander("Filter Rules"):
    min_support = st.slider("Minimum Support", 0.0, 1.0, 0.01, 0.01)
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.05)
    min_lift = st.slider("Minimum Lift", 0.0, 5.0, 1.0, 0.1)

filtered_rules = rules[
    (rules["support"] >= min_support) &
    (rules["confidence"] >= min_confidence) &
    (rules["lift"] >= min_lift)
]

st.write(f"Showing **{len(filtered_rules)}** rules")
st.dataframe(filtered_rules, use_container_width=True)

# --------------------------------
# Download Button
# --------------------------------
st.download_button(
    "⬇️ Download Filtered Rules",
    filtered_rules.to_csv(index=False),
    file_name="filtered_association_rules.csv",
    mime="text/csv"
)

st.divider()

# --------------------------------
# Visualization: Confidence vs Lift
# --------------------------------
st.subheader("📈 Rule Strength Visualization")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=filtered_rules,
    x="confidence",
    y="lift",
    size="support",
    hue="support",
    palette="viridis",
    legend=False,
    ax=ax2
)

ax2.set_title("Confidence vs Lift (Bubble size = Support)")
ax2.set_xlabel("Confidence")
ax2.set_ylabel("Lift")

st.pyplot(fig2)

# --------------------------------
# Footer
# --------------------------------
st.caption("📘 Educational Dashboard | Association Rule Mining using Apriori Algorithm")
