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
st.markdown("Automatic insights and visualization of grocery transactions using Association Rule Mining")

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

c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", basket.shape[0])
c2.metric("Unique Items", basket.shape[1])
c3.metric("Association Rules", rules.shape[0])

st.divider()

# --------------------------------
# Item Frequency
# --------------------------------
st.subheader("📦 Item Frequency")

item_counts = basket.sum().sort_values(ascending=False)
top_n = st.slider("Top items to display", 5, 30, 15)

fig1, ax1 = plt.subplots(figsize=(10, 5))
item_counts.head(top_n).plot(kind="bar", ax=ax1)
ax1.set_ylabel("Frequency")
ax1.set_title(f"Top {top_n} Most Purchased Items")
st.pyplot(fig1)

st.divider()

# --------------------------------
# Rule Filters
# --------------------------------
st.subheader("🎛️ Filter Association Rules")

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

# --------------------------------
# 🤖 AUTOMATIC INSIGHTS SECTION
# --------------------------------
st.subheader("🤖 Automatic Insights")

if len(filtered_rules) == 0:
    st.warning("No rules match the selected criteria.")
else:
    strongest_rule = filtered_rules.sort_values("lift", ascending=False).iloc[0]
    most_confident_rule = filtered_rules.sort_values("confidence", ascending=False).iloc[0]
    most_frequent_rule = filtered_rules.sort_values("support", ascending=False).iloc[0]

    st.markdown(
        f"""
### 🔎 Key Insights

**1️⃣ Strongest Relationship (Highest Lift)**  
Customers who buy **{strongest_rule['antecedents']}** are  
**{strongest_rule['lift']:.2f}× more likely** to also buy  
**{strongest_rule['consequents']}** than average.

**2️⃣ Most Reliable Rule (Highest Confidence)**  
When **{most_confident_rule['antecedents']}** is purchased,  
**{most_confident_rule['confidence']:.1%}** of the time customers also buy  
**{most_confident_rule['consequents']}**.

**3️⃣ Most Frequent Rule (Highest Support)**  
The combination **{most_frequent_rule['antecedents']} → {most_frequent_rule['consequents']}**  
appears in **{most_frequent_rule['support']:.1%}** of all transactions.
        """
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

ax2.set_xlabel("Confidence")
ax2.set_ylabel("Lift")
ax2.set_title("Confidence vs Lift (Bubble Size = Support)")

st.pyplot(fig2)

# --------------------------------
# Footer
# --------------------------------
st.caption("📘 Association Rule Mining Dashboard | Automatic Insight Generation")
