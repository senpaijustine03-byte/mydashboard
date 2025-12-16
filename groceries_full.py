# -------------------------------
# groceries_full.py
# -------------------------------

# Install dependencies if needed:
# pip install pandas mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# 1. Load dataset
# -------------------------------
file_path = "C:\\Users\\biado\\OneDrive\\Desktop\\New folder\\Groceries_dataset.csv"
df = pd.read_csv(file_path, encoding="latin1")

print("Dataset loaded successfully")
print(df.head())

# -------------------------------
# 2. Preprocess: create Transaction column
# -------------------------------
df["Transaction"] = df["Member_number"].astype(str) + "_" + df["Date"]
df = df[["Transaction", "itemDescription"]]

# -------------------------------
# 3. Convert to basket (one-hot encoded)
# -------------------------------
basket = df.groupby(["Transaction", "itemDescription"])["itemDescription"] \
           .count().unstack().fillna(0)
basket = basket.astype(bool)  # True/False required by mlxtend

print("\nBasket format created")
print(basket.head())

# Save basket for dashboard
basket.to_csv("C:\\Users\\biado\\OneDrive\\Desktop\\New folder\\Groceries_basket.csv")

# -------------------------------
# 4. Find frequent itemsets
# -------------------------------
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
print("\nAll frequent itemsets (including single items)")
print(frequent_itemsets.head())

# -------------------------------
# 5. Generate association rules
# -------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules.sort_values(by="lift", ascending=False)

print("\nTop association rules")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head())

# Save rules for dashboard
rules.to_csv("C:\\Users\\biado\\OneDrive\\Desktop\\New folder\\association_rules.csv", index=False)
print("\nAssociation rules saved to CSV")
