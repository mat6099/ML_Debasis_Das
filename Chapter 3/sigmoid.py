import numpy as np 
import matplotlib.pyplot as plt 

# Sigmoid (logistic) activation: maps any real z into (0, 1)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # Compute 1 / (1 + e^-z)

z = np.arange(-7, 7, 0.1)  # Input values from -7 to 7 in steps of 0.1
sigma_z = sigmoid(z)  # Apply sigmoid to every z
plt.plot(z, sigma_z)  # Plot the sigmoid curve
plt.axvline(0.0, color='k')  # Draw a black vertical line at z = 0
plt.ylim(-0.1, 1.1)  # Set y-axis range with a small margin
plt.xlabel('z')  # Label the x-axis
plt.ylabel(r'$\sigma (z)$')  # Label the y-axis (LaTeX sigma symbol)
plt.yticks([0.0, 0.5, 1.0])  # Show y-ticks only at 0, 0.5 and 1
ax = plt.gca()  # Get the current axes
ax.yaxis.grid(True)  # Add horizontal grid lines at the y-ticks
plt.tight_layout()  # Adjust spacing so labels fit
plt.show()  # Display the plot
