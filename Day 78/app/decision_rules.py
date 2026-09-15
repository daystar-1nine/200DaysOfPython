from sklearn.tree import export_text

def export_tree_rules(pipeline, feature_names: list, class_names=None, max_depth=4) -> str:
    if class_names is None:
        class_names = ["Stay (0)", "Churn (1)"]
    clf = pipeline.named_steps["classifier"]
    rules = export_text(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth
    )
    return rules

def explain_customer_path(pipeline, customer_row_df, feature_names: list, class_names=None) -> dict:
    if class_names is None:
        class_names = ["Stay", "Churn"]
        
    clf = pipeline.named_steps["classifier"]
    prep = pipeline.named_steps["preprocessor"]
    
    X_trans = prep.transform(customer_row_df)
    prob = clf.predict_proba(X_trans)[0, 1]
    pred = clf.predict(X_trans)[0]
    
    node_indicator = clf.decision_path(X_trans)
    leaf_id = clf.apply(X_trans)[0]
    
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    rules = []
    for node_id in node_index:
        if leaf_id == node_id:
            continue
        feat_idx = feature[node_id]
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
        thresh_val = threshold[node_id]
        
        if X_trans[0, feat_idx] <= thresh_val:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
            
        rules.append(f"{feat_name} {threshold_sign} {thresh_val:.2f}")
        
    cid = str(customer_row_df["Customer_ID"].values[0]) if "Customer_ID" in customer_row_df.columns else "N/A"
    return {
        "Customer_ID": cid,
        "Predicted_Class": class_names[pred],
        "Churn_Probability": float(prob),
        "Decision_Path": rules,
        "Leaf_ID": int(leaf_id)
    }
