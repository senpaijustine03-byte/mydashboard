import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Groceries Transaction Dataset",
    layout="wide"
)

st.title("🛒 Groceries Transaction Dataset")

# -------------------------------
# Load preprocessed data (RELATIVE PATHS)
# -------------------------------
@st.cache_data
def load_data():
    basket = pd.read_csv("data/groceries_basket.csv", index_col=0)
    rules = pd.read_csv("data/association_rules.csv")
    return basket, rules

basket, rules = load_data()

# -------------------------------
# Clean antecedents and consequents
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
# Basket Overview
# -------------------------------
st.header("📦 Basket Overview")
st.write(
    f"**Total transactions:** {basket.shape[0]} &nbsp;&nbsp; "
    f"**Total items:** {basket.shape[1]}"
)
st.dataframe(basket.head())

# -------------------------------
# Top Items by Frequency
# -------------------------------
st.header("🔥 Top Items by Frequency")
item_counts = basket.sum().sort_values(ascending=False)
top_items = item_counts.head(15)

st.bar_chart(top_items)

# -------------------------------
# Association Rules Overview
# -------------------------------
st.header("📊 Top Association Rules")
st.dataframe(rules.head(20))

# -------------------------------
# Interactive Filters
# -------------------------------
st.subheader("🎛️ Filter Rules")

min_lift = st.slider(
    "Minimum Lift",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1
)

min_confidence = st.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
)

filtered_rules = rules[
    (rules["lift"] >= min_lift) &
    (rules["confidence"] >= min_confidence)
]

st.write(f"**Rules matching criteria:** {len(filtered_rules)}")
st.dataframe(filtered_rules)

# -------------------------------
# Visualization: Support vs Confidence
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

ax.set_title("Association Rules: Support vs Confidence")
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")

st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.caption("📌 Built with Streamlit | Association Rule Mining Dashboard")
