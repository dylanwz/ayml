import numpy as np

def flatten(arr):
    return np.ravel(arr)

def ltoa(label, dataspace):
    match dataspace.upper():
        case "MNIST":
            arr = [0] * 10
            arr[label] = 1
            return arr
