from ipro.ipro_test import TestOracle, TestSolver

import matplotlib.pyplot as plt
import numpy as np


from interactive_ipro_VS_gp_pref_elicit_experiment import InteractiveIproVsGPE

'''''''''''''''''''''''''''''''''''''''''''''
'       EXPERIMENT (test pareto_front)      '
'''''''''''''''''''''''''''''''''''''''''''''
number_of_users = 300
number_of_queries = 15

'''
    PARETO FRONT Problem Setting (artificial data)
'''


balanced_pf_30 = [(2.00, 8.00), (2.10, 7.71), (2.17, 7.39), (2.25, 7.08), (2.34, 6.78), (2.45, 6.48),
                  (2.57, 6.19), (2.70, 5.91), (2.85, 5.64), (3.01, 5.37), (3.18, 5.12), (3.36, 4.87),
                  (3.55, 4.63), (3.75, 4.40), (3.96, 4.17), (4.18, 3.96), (4.41, 3.75), (4.65, 3.56),
                  (4.89, 3.37), (5.15, 3.19), (5.41, 3.02), (5.67, 2.87), (5.95, 2.72), (6.23, 2.58),
                  (6.51, 2.45), (6.80, 2.33), (7.10, 2.23), (7.40, 2.13), (7.70, 2.04), (8.00, 2.00)]

balanced_pf_10 = [
    (2.00, 8.00),
    (2.34, 6.78),
    (2.85, 5.64),
    (3.36, 4.87),
    (3.96, 4.17),
    (4.89, 3.37),
    (5.67, 2.87),
    (6.51, 2.45),
    (7.40, 2.13),
    (8.00, 2.00)
]


# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*balanced_pf_30)
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

# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*balanced_pf_10)
# create the scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(obj1_values, obj2_values, label='non-dominated solution')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto front')
plt.legend()
plt.grid(True)
plt.show()

'''
    IPRO setup
'''
dimensions = 2
utility_noise = .01


# upper bounds calculations: taking the max values of the pareto optimal solutions calculated for each objective
# individually
upper_bnds = {'obj1': 8.0, 'obj2': 8.0}  # specified/defined as is for this test case

# convex PF 30 solutions
linear_solver = TestSolver(balanced_pf_30, upper_bnds)
oracle = TestOracle(balanced_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance')
average_utility_gppe_exp, average_utility_interactive_ipro_exp, average_max_utility_gppe_exp, \
    average_max_utility_interactive_ipro_exp, std_dev_utility_gppe_exp, std_dev_utility_interactive_ipro_exp, \
    std_dev_max_utility_gppe_exp, std_dev_max_utility_interactive_ipro_exp, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


# convex PF 10 solutions
linear_solver = TestSolver(balanced_pf_10, upper_bnds)
oracle = TestOracle(balanced_pf_10, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance')
average_utility_gppe_exp_small, average_utility_interactive_ipro_exp_small, average_max_utility_gppe_exp_small, \
    average_max_utility_interactive_ipro_exp_small, std_dev_utility_gppe_exp_small, std_dev_utility_interactive_ipro_exp_small, \
    std_dev_max_utility_gppe_exp_small, std_dev_max_utility_interactive_ipro_exp_small, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


# plotting Gaussian Process Preference Elicitation versus preference guided (interactive) IPRO results
plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_utility_gppe_exp) + 1),
    average_utility_gppe_exp,
    yerr=std_dev_utility_gppe_exp,
    fmt='-o',
    capsize=5,
    color='b',
    label='GPPE (30 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp) + 1),
    average_utility_interactive_ipro_exp,
    yerr=std_dev_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='r',
    label='PG-IPRO (30 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_gppe_exp_small) + 1),
    average_utility_gppe_exp_small,
    yerr=std_dev_utility_gppe_exp_small,
    fmt='--o',
    capsize=5,
    color='c',
    label='GPPE (10 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_small) + 1),
    average_utility_interactive_ipro_exp_small,
    yerr=std_dev_utility_interactive_ipro_exp_small,
    fmt='--o',
    capsize=5,
    color='m',
    label='PG-IPRO (10 sols PF)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average route utility comparison between GPPE and PG-IPRO for different pareto front density (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_max_utility_gppe_exp) + 1),
    average_max_utility_gppe_exp,
    yerr=std_dev_max_utility_gppe_exp,
    fmt='-o',
    capsize=5,
    color='b',
    label='GPPE (30 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1),
    average_max_utility_interactive_ipro_exp,
    yerr=std_dev_max_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='r',
    label='PG-IPRO (30 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_gppe_exp_small) + 1),
    average_max_utility_gppe_exp_small,
    yerr=std_dev_max_utility_gppe_exp_small,
    fmt='--o',
    capsize=5,
    color='c',
    label='GPPE (10 sols PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_small) + 1),
    average_max_utility_interactive_ipro_exp_small,
    yerr=std_dev_max_utility_interactive_ipro_exp_small,
    fmt='--o',
    capsize=5,
    color='m',
    label='PG-IPRO (10 sols PF)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average maximum utility comparison between GPPE and PG-IPRO for different pareto front density (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()
