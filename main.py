import numpy as np
import matplotlib.pyplot as plt

"""
STEP 1: Define a dataset
"""

input_variables = np.array([1,2,3,4,5]) # x values, independent variables
actual_output = np.array([3,4,2,5,6]) #y values, dependant variables

"""
STEP 2: Initializing model parameters
"""

slope = 0 #m in y = mx + b
intercept = 0 #b in y = mx + b

learning_rate = 0.01
num_iterations = 100

error_history = [] #track the error at each step

"""
STEP 3: Train the model using the gradient inputs
"""

for step in range(num_iterations):
    predicted_output = slope * input_variables + intercept

    error = (actual_output - predicted_output)

    """
    Calculate the gradients
       
    These formulas are the partial derivatives of the Mean Squared Error loss function.
       
        ➤ slope_gradient:
        ∂(Error)/∂slope = (2/n) * Σ(x_i * (y_i - ŷ_i))
        Shows how much the slope contributes to the current error.
    
        ➤ intercept_gradient:
        ∂(Error)/∂intercept = (2/n) * Σ(y_i - ŷ_i)
        Shows how much the intercept contributes to the error.
    """

    slope_gradient = (2/len(input_variables)) * np.dot(input_variables, error)
    intercept_gradient = (2/len(input_variables)) * np.sum(error)

    slope += learning_rate * slope_gradient
    intercept += learning_rate * intercept_gradient

    """
    STEP 3.5 (optional) you can observe the performance of your AI model using the MSE (mean square error) formula
        
    ➤ MSE (Mean Squared Error) is a common way to measure how wrong the model is.
    Formula: MSE = (1/n) * Σ(error_i)^2
    - We square the errors to penalize bigger mistakes more heavily.
    - Then we take the average (mean) so the result represents all predictions.
    
    ➤ np.mean(errors ** 2):
    - `errors ** 2` means we're squaring each error individually.
    - `np.mean(...)` takes the average of all those squared errors.
    - This gives us one number that tells us the model's performance:
    Lower MSE = better predictions.
    """

    mse = np.mean(error ** 2)
    error_history.append(mse)

print("TRAINING COMPLETE")

print(f"Final Result: Learned Line y={round(slope, 2)}x + intercept{round(intercept,2)}, error history = {error_history}")

"""
STEP 4: Plot results onto graph
"""
plt.scatter(input_variables, actual_output, label="Actual Data")
plt.plot(input_variables, slope * input_variables + intercept, color="red", label="Learned Line")
plt.xlabel("Input on X")
plt.ylabel("Input on Y")
plt.grid(True)
plt.show()

"""
STEP 5: Show errors
"""
plt.plot(range(num_iterations), error_history, color="purple")
plt.title("Training error over time (expect to decrese as we train model")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()