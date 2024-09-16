
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from numpy import exp, array, append

# Sigmoid activation function
def Sigmoid(x):
    return 1 / (1 + exp(-x))
  
# Graphical representation of our Sigmoid activation function
for i in array([range(-10, 10)]):
    x = append(array([]), i)
    plt.plot(x, Sigmoid(x))
    plt.title("Sigmoid activation function")
    plt.xlabel("x")
    plt.ylabel("S(x)")