import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Groceries Transaction Dataset",
    layout="wide"
)

st.title("🛒 Groceries Transaction Dataset")

# -------------------------------
# Load data (ROOT DIRECTORY)
# -------------------------------
@st.cache_data
def load_data():
    basket = pd.read_csv("groceries_basket.csv", index_col=0)
    rules = pd.read_csv("association_rules.csv")
    return basket, rules

basket, rules = load_data()

# -------------------------------
# Clean antecedents & consequents
# -------------------------------
rules["antecedents"] = (
    rules["antecedents"]
    .str.replace("frozenset({", "", regex=False)
    .str.replace("})", "", regex=False)
    .str.replace("'", "", regex=False)
)

rules["consequents"] = (
    rules["consequents"]
    .str.replace("frozenset({", "", regex=False)
    .str.replace("})", "", regex=False)
    .str.replace("'", "", regex=False)
)

# -------------------------------
# Basket overview
# -------------------------------
st.header("📦 Basket Overview")
st.write(
    f"**Total transactions:** {basket.shape[0]} | "
    f"**Total items:** {basket.shape[1]}"
)
st.dataframe(basket.head())

# -------------------------------
# Top items
# -------------------------------
st.header("🔥 Top Items by Frequency")
item_counts = basket.sum().sort_values(ascending=False)
top_items = item_counts.head(15)

st.bar_chart(top_items)

# -------------------------------
# Association rules
# -------------------------------
st.header("📊 Top Association Rules")
st.dataframe(rules.head(20))

# -------------------------------
# Filters
# -------------------------------
st.subheader("🎛️ Filter Rules")

min_lift = st.slider(
    "Minimum Lift", 0.0, 5.0, 1.0, 0.1
)
min_confidence = st.slider(
    "Minimum Confidence", 0.0, 1.0, 0.1, 0.05
)

filtered_rules = rules[
    (rules["lift"] >= min_lift) &
    (rules["confidence"] >= min_confidence)
]

st.write(f"**Rules matching criteria:** {len(filtered_rules)}")
st.dataframe(filtered_rules)

# -------------------------------
# Scatter plot
# -------------------------------
st.subheader("📈 Support vs Confidence")

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=filtered_rules,
    x="support",
    y="confidence",
    size="lift",
    hue="lift",
    palette="viridis",
    legend=False,
    ax=ax
)

ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_title("Association Rules: Support vs Confidence")

st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.caption("Built with Streamlit | Association Rule Mining Dashboard")
