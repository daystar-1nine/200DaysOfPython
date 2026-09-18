
def match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5):
    # Mocking returning TP, FP, FN
    return 80, 10, 15
    
def evaluate_thresholds(predictions, ground_truth, thresholds):
    # Mocking
    return [
        {'threshold': t, 'precision': 0.8+t*0.1, 'recall': 0.9-t*0.2, 'f1': 0.85-t*0.05}
        for t in thresholds
    ]
