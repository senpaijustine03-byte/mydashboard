import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# 1. Load CSV
# -------------------------------
csv_file = r"C:\Users\biado\OneDrive\Desktop\New folder\Groceries_dataset.csv"
df = pd.read_csv(csv_file, encoding="latin1")

print("Dataset loaded successfully")
print(df.head())

# -------------------------------
# 2. Create transaction ID
# -------------------------------
df["Transaction"] = df["Member_number"].astype(str) + "_" + df["Date"]

# -------------------------------
# 3. Create boolean basket
# -------------------------------
basket = (
    df.groupby(["Transaction", "itemDescription"])["itemDescription"]
    .count()
    .unstack()
    .fillna(0)
)
basket = basket > 0  # convert to boolean

print("\nBasket format created")
print(basket.head())

# -------------------------------
# 4. Find frequent itemsets
# -------------------------------
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

print("\nAll frequent itemsets (including single items)")
print(frequent_itemsets.head())

# -------------------------------
# 5. Generate association rules safely
# -------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules.sort_values(by="lift", ascending=False)

# Optionally display only rules with 2+ items in antecedents/consequents
rules_filtered = rules[rules['antecedents'].apply(lambda x: len(x) > 0) &
                       rules['consequents'].apply(lambda x: len(x) > 0)]

print("\nTop association rules")
print(rules_filtered[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# -------------------------------
# 6. Save to CSV
# -------------------------------
rules_output = r"C:\Users\biado\OneDrive\Desktop\New folder\association_rules.csv"
rules_filtered.to_csv(rules_output, index=False)
print(f"\nAssociation rules saved to: {rules_output}")
