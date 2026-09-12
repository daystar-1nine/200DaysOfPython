def select_best_model(scores_dict):
    return max(scores_dict, key=scores_dict.get)
