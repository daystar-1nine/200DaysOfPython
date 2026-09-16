# Neuron From Scratch
import numpy as np

def neuron(x, weights, bias):
    z = np.dot(x, weights) + bias
    return z

def relu(z):
    return np.maximum(0, z)
