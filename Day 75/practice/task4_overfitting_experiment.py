def check_overfit(train_err, test_err):
    return test_err > train_err * 1.5
