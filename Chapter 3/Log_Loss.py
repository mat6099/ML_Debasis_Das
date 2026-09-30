import numpy as np  
import matplotlib.pyplot as plt 

# Sigmoid (logistic) activation: maps any real z into (0, 1)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # Compute 1 / (1 + e^-z)


# Log loss for a sample whose class is y = 1
def loss_1(z):
    return - np.log(sigmoid(z))  # -log(sigma(z)): small when sigma(z) -> 1, large when -> 0


# Log loss for a sample whose class is y = 0
def loss_0(z):
    return - np.log(1 - sigmoid(z))  # -log(1 - sigma(z)): small when sigma(z) -> 0, large when -> 1


z = np.arange(-10, 10, 0.1)  # Net input values from -10 to 10 in steps of 0.1
sigma_z = sigmoid(z)  # Sigmoid outputs (predicted probabilities) for each z
c1 = [loss_1(X) for X in z]  # Loss values for every z when y = 1
plt.plot(sigma_z, c1, label='L(w, b) if y=1')  # Plot y=1 loss against sigma(z) as a solid line
c0 = [loss_0(X) for X in z]  # Loss values for every z when y = 0
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')  # Plot y=0 loss as a dashed line
plt.ylim(0.0, 5.1)  # Limit y-axis so the steep ends of the curves stay readable
plt.xlim([0,1])  # Sigmoid output lies in [0, 1], so bound the x-axis there
plt.xlabel(r'$\sigma(z)$')  # X-axis label rendered as LaTeX sigma(z)
plt.ylabel('L(w, b)')  # Y-axis label: loss as a function of weights and bias
plt.legend(loc='best')  # Show the legend where it overlaps the curves least
plt.tight_layout()  # Adjust spacing so labels are not clipped
plt.show()  # Display the figure
