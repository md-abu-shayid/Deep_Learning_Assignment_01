import numpy as np
import matplotlib.pyplot as plt

def relulu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x >= 0 else 0

# Visualization with a larger dataset
x = np.linspace(-30, 30, 1000)
y = [relulu(x) for x in x]
y_derivative = [relu_derivative(x) for x in x]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, label='ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label='ReLU Derivative', color='orange')
plt.xlabel('Input')
plt.ylabel('Derivative')
plt.title('ReLU Derivative')
plt.legend()

plt.tight_layout()
plt.show()
