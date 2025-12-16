import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load preprocessed data
# -------------------------------
basket = pd.read_csv("C:\\Users\\biado\\OneDrive\\Desktop\\New folder\\groceries_basket.csv", index_col=0)
rules = pd.read_csv("C:\\Users\\biado\\OneDrive\\Desktop\\New folder\\association_rules.csv")

# -------------------------------
# Clean antecedents and consequents
# -------------------------------
rules['antecedents'] = rules['antecedents'].apply(lambda x: x.replace("frozenset({", "").replace("})", "").replace("'", ""))
rules['consequents'] = rules['consequents'].apply(lambda x: x.replace("frozenset({", "").replace("})", "").replace("'", ""))

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Groceries Transaction Dataset ", layout="wide")

st.title("🛒 Groceries Transaction Dataset")

# -------------------------------
# Basket Overview
# -------------------------------
st.header("Basket (One-Hot Encoded)")
st.write(f"Total transactions: {basket.shape[0]} Total items: {basket.shape[1]}")
st.dataframe(basket.head())

# -------------------------------
# Top Items by Frequency
# -------------------------------
st.header("Top Items by Frequency")
item_counts = basket.sum().sort_values(ascending=False)
top_items = item_counts.head(15)

st.bar_chart(top_items)

# -------------------------------
# Association Rules Overview
# -------------------------------
st.header("Top Association Rules")
st.dataframe(rules.head(20))

# -------------------------------
# Interactive Filters
# -------------------------------
st.subheader("Filter Rules")
min_lift = st.slider("Minimum Lift", min_value=0.0, max_value=5.0, value=1.0)
min_confidence = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, value=0.1)

filtered_rules = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_confidence)]

st.write(f"Rules matching criteria: {len(filtered_rules)}")
st.dataframe(filtered_rules)

# -------------------------------
# Visualization: Support vs Confidence
# -------------------------------
st.subheader("Support vs Confidence Scatter Plot")
plt.figure(figsize=(10,6))
sns.scatterplot(data=filtered_rules, x="support", y="confidence", size="lift", hue="lift", palette="viridis", legend=False)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence (Bubble size = Lift)")
st.pyplot(plt)
