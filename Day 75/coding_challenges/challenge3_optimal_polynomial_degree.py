def find_optimal_degree(degree_scores):
    return min(degree_scores, key=degree_scores.get)
