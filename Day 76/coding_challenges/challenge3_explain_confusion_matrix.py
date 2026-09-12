# Challenge 3: Explain Confusion Matrix
# Write a custom function that prints out actionable business insights based on a confusion matrix

def business_insights_from_cm(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Of all the times we predicted a positive event, we were right {precision:.2%}")
    print(f"Of all the actual positive events, we captured {recall:.2%}")
    
    print("\nBusiness Impact:")
    print(f"We missed {fn} opportunities (False Negatives).")
    print(f"We wasted resources on {fp} incorrect flags (False Positives).")

if __name__ == "__main__":
    business_insights_from_cm(800, 50, 100, 50)
