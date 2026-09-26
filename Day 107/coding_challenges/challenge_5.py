def compare_errors(model_a_preds, model_b_preds, labels):
    """
    Finds which errors are unique to Model A and unique to Model B.
    """
    errors_a = set([i for i in range(len(labels)) if model_a_preds[i] != labels[i]])
    errors_b = set([i for i in range(len(labels)) if model_b_preds[i] != labels[i]])
    
    unique_to_a = errors_a - errors_b
    unique_to_b = errors_b - errors_a
    return unique_to_a, unique_to_b

if __name__ == "__main__":
    labels = [1, 0, 1, 1, 0]
    preds_a = [1, 1, 1, 0, 0] # Errors at 1, 3
    preds_b = [1, 0, 0, 1, 0] # Errors at 2
    
    a_only, b_only = compare_errors(preds_a, preds_b, labels)
    print(f"Errors unique to A: {a_only}")
    print(f"Errors unique to B: {b_only}")
