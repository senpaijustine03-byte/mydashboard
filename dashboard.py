# -------------------------
# 4️⃣ Customer Behavior
# -------------------------
with tabs[4]:
    st.subheader("👥 Customer Behavior")
    cust_orders = filtered_data.groupby('Member_number')['order_id'].nunique()
    repeat_status = cust_orders.apply(lambda x: "Repeat" if x>1 else "First-time").value_counts()
    fig_repeat = px.pie(repeat_status, names=repeat_status.index, values=repeat_status.values,
                        title="Repeat vs First-time Customers", template="plotly_white")
    st.plotly_chart(fig_repeat, use_container_width=True, key="tab4_repeat_status")

    st.subheader("🏆 Top Buyers")
    top_buyers = cust_orders.sort_values(ascending=False).head(10)
    fig_top_buyers = px.bar(top_buyers, x=top_buyers.index, y=top_buyers.values,
                            labels={'x':'CustomerID','y':'Number of Transactions'}, title="Top 10 Buyers", template="plotly_white")
    st.plotly_chart(fig_top_buyers, use_container_width=True, key="tab4_top_buyers")

    st.subheader("🛒 Average Items per Transaction")
    avg_items = basket_oh.sum(axis=1).mean()
    st.metric("Average items per transaction", f"{avg_items:.2f}", key="tab4_avg_items")

# -------------------------
# 5️⃣ Seasonal Trends
# -------------------------
with tabs[5]:
    st.subheader("📅 Monthly Seasonal Trends")
    monthly_sales = filtered_data.groupby(filtered_data['timestamp'].dt.month)['order_id'].count()
    fig_monthly = px.line(monthly_sales, x=monthly_sales.index, y=monthly_sales.values,
                          labels={'x':'Month', 'y':'Number of Transactions'}, title="Monthly Transaction Trend", template="plotly_white")
    st.plotly_chart(fig_monthly, use_container_width=True, key="tab5_monthly_trend")

# -------------------------
# 6️⃣ Item Co-occurrence
# -------------------------
with tabs[6]:
    st.subheader("📊 Top 20 Item Co-occurrence Heatmap")
    if not basket_oh.empty:
        basket_items = basket_oh.astype(float)
        top_items = basket_items.sum().sort_values(ascending=False).head(20).index
        co_occurrence = basket_items[top_items].T.dot(basket_items[top_items])
        co_occurrence_pct = (co_occurrence / basket_items.shape[0] * 100).astype(float)
        fig6, ax6 = plt.subplots(figsize=(12,10))
        sns.heatmap(co_occurrence_pct, annot=False, cmap="YlGnBu", ax=ax6)
        ax6.set_title("Item Co-occurrence (% of transactions)")
        st.pyplot(fig6, key="tab6_co_occurrence")
    else:
        st.info("No transactions available for the selected filters.")

# -------------------------
# 7️⃣ Basket Recommendations
# -------------------------
with tabs[7]:
    st.subheader("🛍️ Market Basket Recommendations")
    selected_items = st.multiselect("Select items in your basket:", options=item_cols_filtered, key="tab7_select_items")
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.3, key="tab7_min_conf")
    min_lift = st.slider("Minimum lift", 0.0, 5.0, 1.0, key="tab7_min_lift")

    if selected_items:
        if not rules.empty:
            recommended_rules = rules[
                (rules["antecedents"].apply(lambda x: any(item in x.split(", ") for item in selected_items))) &
                (rules["confidence"] >= min_conf) &
                (rules["lift"] >= min_lift)
            ]
            if not recommended_rules.empty:
                top_recs = recommended_rules.sort_values(by=['confidence','lift'], ascending=False).head(10)
                st.write(f"**Top recommendations for ({', '.join(selected_items)}):**")
                st.dataframe(top_recs[['antecedents','consequents','support','confidence','lift']], key="tab7_top_recs")
                suggested_items = set()
                for cons in top_recs['consequents']:
                    suggested_items.update(cons.split(", "))
                suggested_items = [item for item in suggested_items if item not in selected_items]
                st.markdown(f"**Suggested items:** {', '.join(suggested_items)}")
            else:
                st.info("No recommendations found for the selected items.")
        else:
            st.info("No rules available to generate recommendations.")
    else:
        st.info("Select items from the basket to get recommendations.")

# -------------------------
# 8️⃣ Top Bundles & Item Combos
# -------------------------
with tabs[8]:
    st.subheader("📦 Top Bundles & Frequently Bought Together")
    if not basket_oh.empty:
        selected_item_combo = st.selectbox("Select an item to see popular bundles:", options=item_cols_filtered, key="tab8_select_item")
        top_n_bundles = st.slider("Top bundles to show:", 3, 15, 5, key="tab8_top_n_bundles")

        transactions_with_item = basket_oh[basket_oh[selected_item_combo] == 1]
        co_occurring_counts = transactions_with_item.sum().sort_values(ascending=False)
        co_occurring_counts = co_occurring_counts.drop(labels=[selected_item_combo], errors='ignore')

        if not co_occurring_counts.empty:
            top_co_occurring = co_occurring_counts.head(top_n_bundles)
            fig_bundle = px.bar(
                top_co_occurring,
                x=top_co_occurring.index,
                y=top_co_occurring.values,
                labels={'x':'Item', 'y':'Number of Co-occurrences'},
                title=f"Top {top_n_bundles} items bought together with '{selected_item_combo}'",
                template="plotly_white"
            )
            st.plotly_chart(fig_bundle, use_container_width=True, key="tab8_top_bundles")
        else:
            st.info(f"No co-occurring items found for '{selected_item_combo}'.")
    else:
        st.info("No transactions available for the selected filters.")

st.caption("📘 Dashboard Tabs: Raw Data | Unique Items & Customers | Transactions | Top Items | Customer Behavior | Seasonal Trends | Co-occurrence | Recommendations | Top Bundles")
