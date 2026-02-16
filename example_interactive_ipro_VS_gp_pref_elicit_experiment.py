from gp_pref_elicit_experiment_extensions.ipro_gp_pref_elicit_experiment import IproGpPrefElicitExperiment
from gp_pref_elicit.gp_utilities.utils_parameters import get_parameter_dict

from gp_pref_elicit_experiment_extensions.user_utility import UserUtility

from ipro.ipro_test import TestOracle, TestSolver
from ipro.outer_loops.ipro import IPRO

import matplotlib.pyplot as plt
import numpy as np

def _stacked_sigmoids(x, a, b, n):
    y = 0
    y_min = 0
    y_max = 0
    for i in range(0, 5*n, 5):
        y += 1. / (1 + np.exp(- x * (a-i) + (b+i)))
        y_min += 1. / (1 + np.exp(- 0 * (a-i) + (b+i)))
        y_max += 1. / (1 + np.exp(- 1 * (a-i) + (b+i)))
    return (y - y_min) / (y_max - y_min)

def inverted_stacked_sigmoids(x, a, b, n):
    y = 0
    y_min = 0
    y_max = 0
    for i in range(0, 5 * n, 5):
        y += 1. / (1 + np.exp(- (-x + 1) * (a - i) + (b + i)))  # flip horizontally between [0,1] with (-x + 1)
        y_min += 1. / (1 + np.exp(- 0 * (a - i) + (b + i)))
        y_max += 1. / (1 + np.exp(- 1 * (a - i) + (b + i)))
    return (y - y_min) / (y_max - y_min)

x_values = np.linspace(0, 1, 500)
a, b, n = 20, 5, 3  # parameters for an example

y_original = np.array([_stacked_sigmoids(x, a, b, n) for x in x_values])
y_inverted = np.array([inverted_stacked_sigmoids(x, a, b, n) for x in x_values])

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_original, label="Original stacked sigmoid", color="blue")
plt.plot(x_values, y_inverted, label="Horizontally flipped sigmoid", color="red", linestyle="--")
plt.xlabel("Objective value")
plt.ylabel("Utility")
plt.title("Stacked Sigmoid and its Horizontal Flip")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid(True)
plt.show()



''''''''''''''''''''''''''
'       EXPERIMENT       '
''''''''''''''''''''''''''
experiment_iterations = 100
experiment_seed = 48

'''
    PARETO FRONT (data)
'''
'''
pf = [
    (2.0, 8.0),
    (3.0, 5.2),
    (6.5, 2.5),
    (4.2, 3.8),
    (5.0, 3.2),
    (5.8, 2.9),
    (3.8, 4.5),
    (7.2, 2.1),
    (2.5, 6.5),
    (8.0, 2.0)
]
'''
pf = [
    (2.0, 8.0), (2.21, 7.38), (2.41, 6.76), (2.62, 6.19), (2.83, 5.65),
    (3.03, 5.17), (3.24, 4.99), (3.45, 4.81), (3.66, 4.63), (3.86, 4.39),
    (4.07, 4.03), (4.28, 3.74), (4.48, 3.59), (4.69, 3.43), (4.90, 3.28),
    (5.10, 3.16), (5.31, 3.08), (5.52, 3.01), (5.72, 2.93), (5.93, 2.83),
    (6.14, 2.71), (6.34, 2.59), (6.55, 2.47), (6.76, 2.35), (6.97, 2.23),
    (7.17, 2.12), (7.38, 2.08), (7.59, 2.05), (7.79, 2.03), (8.0, 2.0)
]

'''
    Find Pareto Front with Ipro (for timing gp_pref_elicit experiment part)
'''

problem_id = 'test'
dimensions = 2

# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*pf)
# Create the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(obj1_values, obj2_values, label='non-dominated solution')
# Labels and title
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto front')
plt.legend()
plt.grid(True)
plt.show()


# upper bounds calculations: taking the max values of the pareto optimal solutions calculated for each objective
# individually
upper_bnds = {'obj1': 8.0, 'obj2': 8.0}  # specified/defined as is for this test case
linear_solver = TestSolver(pf, upper_bnds)
oracle = TestOracle(pf, upper_bnds)


'''
    IPRO without manual user input interaction, but with Gaussian Process preference elicitation to find the optimal
    preferred solution by the user (after a certain amount of queries)
'''

ipro = IPRO(
    problem_id=problem_id,
    dimensions=dimensions,
    oracle=oracle,
    linear_solver=linear_solver,
    direction='minimize',
    max_iterations=100,
    tolerance=0,
    user_interaction_loop=False,
)

ipro_pareto_front = ipro.solve()
print("pf:", ipro_pareto_front)

input_dom = np.vstack([item[0]*-1 for item in ipro_pareto_front])
print(input_dom)
# normalize ipro_pareto_front (input_dom) with objective values between [0,1]
min_vals = np.min(input_dom, axis=0)
max_vals = np.max(input_dom, axis=0)
normalized_data = (input_dom - min_vals) / (max_vals - min_vals)
#denormalized_data = normalized_data * (max_vals - min_vals) + min_vals

'''
    USER PREFERENCES as utility function
'''
#util_noise_experiment = 0.01
util_noise_experiment = 0.01
user_utility = UserUtility(num_objectives=dimensions, std_noise=util_noise_experiment, seed=experiment_seed)
user_utility.rescale_on_input_domain(normalized_data)

'''
    Gaussian Process preference elicitation experiment
'''
params = get_parameter_dict(query_type='pairwise', num_objectives=dimensions,  utility_noise=util_noise_experiment)
params['num queries'] = 50
params['seed'] = experiment_seed
experiment = IproGpPrefElicitExperiment(input_domain=normalized_data, user_util=user_utility, parameters=params)
results = experiment.run(recalculate=True)

print("--------------------------------")
print("RESULTS Gaussian process preference elicitation")
print(" ")
max_utility_per_query = results[0]
print("max_utility_per_query: ", max_utility_per_query)
print("input_domain: ", results[1])
print("true_utility: ", results[2])
print("gp_pred_mean: ", results[3])
print("gp_pred_var: ", results[4])
print("acquirer.history / datapoints: ", results[5])
print("user.get_preference(self.acquirer.history, add_noise=False) / utility datapoints: ", results[6])
utility_datapoints = results[6]
print("--------------------------------")

# plotting max_utility_per_query
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(max_utility_per_query) + 1), max_utility_per_query, marker='o', linestyle='-', color='b', label='Utility')
plt.xlabel("Query")
plt.ylabel("Utility")
plt.title("Max utility per Query")
plt.legend()
plt.grid(True)
plt.show()
# plotting max_utility_per_query
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(utility_datapoints) + 1), utility_datapoints, marker='o', linestyle='-', color='b', label='Utility')
plt.xlabel("Query")
plt.ylabel("Utility")
plt.title("Utility per new datapoint queried")
plt.legend()
plt.grid(True)
plt.show()

# Visualization
x_values = np.linspace(-1, 2, 100)  # Range from 0 to 1
# Generate y-values for each function in funcs_1d
y_values = [
    [func(np.array([x])) for x in x_values] for func in user_utility.funcs_1d
]
# Plot the utility functions
plt.figure(figsize=(8, 5))
for i, y in enumerate(y_values):
    plt.plot(x_values, y, label=f'Utility Function {i+1}')
plt.xlabel('Cost Input')
plt.ylabel('Utility Output')
plt.title('Visualization of User Utility Functions')
plt.legend()
plt.grid(True)
plt.show()


'''
    IPRO with user input interaction experiment
'''

ipro = IPRO(
    problem_id=problem_id,
    dimensions=dimensions,
    oracle=oracle,
    linear_solver=linear_solver,
    direction='minimize',
    max_iterations=100,
    tolerance=0,
    user_interaction_loop=True,  # IPRO with user input interaction
    preferred_objective=1
)

processed_solutions = [[2., 8.], [8., 2.]]  # keeping track of subsolutions for visualisation
utility_subsolutions = []
user_input = None
while True:
    print("Enter ipro solve")
    subsolution = ipro.solve(user_input=user_input) #  subsolution of type list[tuple[np.ndarray, Any]],
                               #  but only returns 1 element (list) from solve for IPRO with user input interaction
    if subsolution is None:
        print("User preferences can't be improved, no better solution exists.")
        break
    elif len(subsolution) != 1:
        # at end of IPRO solve method it returns the whole pareto front,
        # this marks the end of the iteration/interaction loop since all possibilities have been processed
        print("pf:", subsolution)
        print("Done!")
        break
    else:
        pareto_sol_objectives = subsolution[0][0] * -1  # * -1 because it is a minimization problem
        pareto_sol = subsolution[0][1]
        if pareto_sol is None:  # sol returned by oracle should be None if no pareto_sol can be found
            print("Current referent can't find a subsolution, try again for a next_referent...")
            continue
        else:
            processed_solutions.append(pareto_sol_objectives)
            print("subsolution in user loop:", pareto_sol_objectives)
            print("processed_sols: ", processed_solutions)
            # separate objectives to show scatterplot
            obj1_values, obj2_values = zip(*processed_solutions)
            plt.figure(figsize=(6, 6))
            plt.scatter(obj1_values, obj2_values, label='processed non-dominated solutions')
            plt.scatter(pareto_sol_objectives[0], pareto_sol_objectives[1], color='green', label='Current non-dominated solution')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('Pareto front')
            plt.legend()
            plt.grid(True)
            plt.show()
    # Save user utility for the given subsolution
    normalized_pareto_sol_objectives = (pareto_sol_objectives - min_vals) / (max_vals - min_vals)
    utility_subsolutions.append(user_utility.get_preference(np.array(normalized_pareto_sol_objectives), add_noise=False)[0])
    # Query user utility for next objective binary
    obj, direction = user_utility.find_most_promising_objective(
        normalized_pareto_sol_objectives)  # utility model works with domain [0,1]
    print("user_utility.find_most_promising_objective:", obj)
    user_input = (obj, direction)

print("utility_subsolutions :", utility_subsolutions)
# plotting max_utility_per_query
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(utility_subsolutions) + 1), utility_subsolutions, marker='o', linestyle='-', color='b', label='Utility')
plt.xlabel("Query")
plt.ylabel("Utility")
plt.title("Utility per subsolutions")
plt.legend()
plt.grid(True)
plt.show()

