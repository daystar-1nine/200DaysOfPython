"""Challenge 4: Customer-Level Decision Path Explanation."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def solve():
    df = pd.read_csv(r"s:\Programming\Python200days\Day 78\data\processed\cleaned_customer_churn.csv")
    features = ["Tenure_Months", "Support_Calls", "Monthly_Charges", "Late_Payments"]
    X = df[features]
    y = df["Churn"]
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Select customer with high churn probability
    sample_idx = 10
    customer = df.iloc[sample_idx]
    c_x = X.iloc[[sample_idx]]
    
    prob = clf.predict_proba(c_x)[0, 1]
    pred = clf.predict(c_x)[0]
    
    print(f"Customer ID:     {customer['Customer_ID']}")
    print(f"Actual Outcome:  {'Churn' if customer['Churn'] == 1 else 'Stay'}")
    print(f"Predicted Class: {'Churn' if pred == 1 else 'Stay'}")
    print(f"Churn Prob:      {prob:.4f}")
    print("\nDecision Path Tracing:")
    
    node_indicator = clf.decision_path(c_x)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    leaf_id = clf.apply(c_x)[0]
    
    for node_id in node_index:
        if node_id == leaf_id:
            continue
        feat_idx = clf.tree_.feature[node_id]
        thresh = clf.tree_.threshold[node_id]
        val = c_x.iloc[0, feat_idx]
        sign = "<=" if val <= thresh else ">"
        print(f"  Node {node_id}: {features[feat_idx]} ({val}) {sign} {thresh:.2f}")

if __name__ == "__main__":
    solve()
