"""
Day 74 - Practical Task 1: Manual Multiple Regression Prediction
Calculate predicted sales manually using regression coefficients.
Model: Sales = 1000 + 2(TV) + 3(Digital) + 1(Radio)
"""

def predict_sales(tv: float, digital: float, radio: float, intercept: float = 1000.0) -> float:
    return intercept + 2.0 * tv + 3.0 * digital + 1.0 * radio

if __name__ == "__main__":
    print("=== Manual Multiple Regression Prediction ===")
    
    # Scenario A
    tv_a, dig_a, rad_a = 50.0, 30.0, 10.0
    pred_a = predict_sales(tv_a, dig_a, rad_a)
    print(f"Scenario A: TV={tv_a}, Digital={dig_a}, Radio={rad_a} -> Predicted Sales = {pred_a}")
    
    # Scenario B (increase digital spend to 100)
    tv_b, dig_b, rad_b = 50.0, 100.0, 10.0
    pred_b = predict_sales(tv_b, dig_b, rad_b)
    print(f"Scenario B: TV={tv_b}, Digital={dig_b}, Radio={rad_b} -> Predicted Sales = {pred_b}")
    print(f"Delta Sales from +70 Digital Spend: {pred_b - pred_a} (Rate: 3.0 units/spend)")
