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
c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", basket.shape[0])
c2.metric("Unique Items", basket.shape[1])
c3.metric("Association Rules", rules.shape[0])

st.divider()

# --------------------------------
# Item Frequency
# --------------------------------
st.subheader("📦 Item Frequency Analysis")
item_counts = basket.sum().sort_values(ascending=False)
top_n = st.slider("Top items to display", 5, 30, 15)

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
# Automatic Insights
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
# Interactive Rule Strength Visualization (Plotly)
# --------------------------------
st.subheader("📈 Interactive Rule Strength Visualization")
if len(filtered_rules) > 0:
    fig2 = px.scatter(
        filtered_rules,
        x="confidence",
        y="lift",
        size="support",
        color="support",
        hover_data=["antecedents", "consequents", "support", "confidence", "lift"],
        color_continuous_scale="viridis",
        size_max=30,
        template="plotly_white"
    )
    fig2.update_layout(
        title="Confidence vs Lift (Bubble Size = Support)",
        xaxis_title="Confidence",
        yaxis_title="Lift",
        legend_title="Support"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No rules to display in the plot with current filters.")

st.divider()

# --------------------------------
# Rule Recommendation System (Market Basket Assistant)
# --------------------------------
st.subheader("🛍️ Market Basket Recommendations")

selected_items = st.multiselect(
    "Select items you have in your basket:",
    options=basket.columns
)

if selected_items:
    recommended_rules = filtered_rules[
        filtered_rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))
    ].sort_values(by=["confidence", "lift"], ascending=False)

    if not recommended_rules.empty:
        top_recommendations = recommended_rules.head(10)
        st.write(f"**Top recommendations based on your selection ({', '.join(selected_items)}):**")
        st.dataframe(top_recommendations[["antecedents", "consequents", "support", "confidence", "lift"]], use_container_width=True)

        # Aggregate suggested items
        suggested_items = set()
        for cons in top_recommendations["consequents"]:
            suggested_items.update(cons.split(", "))
        suggested_items = [item for item in suggested_items if item not in selected_items]

        st.markdown(f"**Suggested items to consider:** {', '.join(suggested_items)}")
    else:
        st.info("No recommendations found for the selected items with current filters.")
else:
    st.info("Select one or more items from the basket to get recommendations.")

# --------------------------------
# Footer
# --------------------------------
st.caption("📘 Association Rule Mining Dashboard | Automatic Insights | Market Basket Assistant | Interactive Plots")
