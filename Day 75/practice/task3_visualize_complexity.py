import matplotlib.pyplot as plt

def plot_complexity():
    degrees = [1, 2, 3, 4, 5]
    errors = [5.2, 3.1, 1.5, 1.4, 1.45]
    plt.plot(degrees, errors)
    plt.title("Complexity vs Error")
