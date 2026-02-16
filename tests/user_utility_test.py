from example_interactive_ipro_VS_gp_pref_elicit_experiment import UserUtility
import numpy as np
import matplotlib.pyplot as plt

random_state = np.random.RandomState(42)
num_objectives = 3
user_utility = UserUtility(num_objectives=num_objectives, std_noise=0, seed=42)


cost_vector = np.array([[0.76, 0.31, 0.69]])  # Reshaped into a 2D array
y_values = [func(x) for func, x in zip(user_utility.funcs_1d, cost_vector.flatten())]
utility_value = user_utility.utility_func(cost_vector)
most_promising_obj = user_utility.find_most_promising_objective(cost_vector.flatten())

print("Cost Vector:", cost_vector)
print("Y-Values for each objective:", y_values)
print("Final Utility Value:", utility_value[0])
print("most_promising_obj :", most_promising_obj)

# Visualization
x_values = np.linspace(0, 1, 100)  # Range from 0 to 1

# Generate y-values for each function in funcs_1d
y_values = [
    [func(np.array([x])) for x in x_values] for func in user_utility.funcs_1d
]

plt.figure(figsize=(8, 5))
for i, y in enumerate(y_values):
    x = np.array(x_values)
    y = np.array(y)
    mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    plt.plot(x[mask], y[mask], label=f'Utility Function {i + 1}')

plt.xlabel('Cost Input')
plt.ylabel('Utility Output')
plt.title('Visualization of User Utility Functions')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()