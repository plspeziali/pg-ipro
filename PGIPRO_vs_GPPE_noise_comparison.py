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

concave_pf_30 = [
    (2.00, 8.00), (2.29, 7.90), (2.61, 7.83), (2.92, 7.75), (3.22, 7.66), (3.52, 7.55),
    (3.81, 7.43), (4.09, 7.30), (4.36, 7.15), (4.63, 6.99), (4.88, 6.82), (5.13, 6.64),
    (5.37, 6.45), (5.60, 6.25), (5.83, 6.04), (6.04, 5.82), (6.25, 5.59), (6.44, 5.35),
    (6.63, 5.11), (6.81, 4.85), (6.98, 4.59), (7.13, 4.33), (7.28, 4.05), (7.42, 3.77),
    (7.55, 3.49), (7.67, 3.20), (7.77, 2.90), (7.87, 2.60), (7.96, 2.30), (8.00, 2.00)
]

linear_pf_30 = [
    (2.00, 8.00), (2.21, 7.79), (2.41, 7.59), (2.62, 7.38), (2.83, 7.17),
    (3.03, 6.97), (3.24, 6.76), (3.45, 6.55), (3.66, 6.34), (3.86, 6.14),
    (4.07, 5.93), (4.28, 5.72), (4.48, 5.52), (4.69, 5.31), (4.90, 5.10),
    (5.10, 4.90), (5.31, 4.69), (5.52, 4.48), (5.72, 4.28), (5.93, 4.07),
    (6.14, 3.86), (6.34, 3.66), (6.55, 3.45), (6.76, 3.24), (6.97, 3.03),
    (7.17, 2.83), (7.38, 2.62), (7.59, 2.41), (7.79, 2.21), (8.00, 2.00)
]

# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*linear_pf_30)
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
normal_utility_noise = .01
high_utility_noise = .2


# upper bounds calculations: taking the max values of the pareto optimal solutions calculated for each objective
# individually
upper_bnds = {'obj1': 8.0, 'obj2': 8.0}  # specified/defined as is for this test case

# small noise
linear_solver = TestSolver(linear_pf_30, upper_bnds)
oracle = TestOracle(linear_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=normal_utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance',
                                                 tolerance=1e-8)
average_utility_gppe_exp, average_utility_interactive_ipro_exp, average_max_utility_gppe_exp, \
    average_max_utility_interactive_ipro_exp, std_dev_utility_gppe_exp, std_dev_utility_interactive_ipro_exp, \
    std_dev_max_utility_gppe_exp, std_dev_max_utility_interactive_ipro_exp, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


# much noise
linear_solver = TestSolver(linear_pf_30, upper_bnds)
oracle = TestOracle(linear_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=high_utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance',
                                                 tolerance=1e-8)
average_utility_gppe_exp_noisy, average_utility_interactive_ipro_exp_noisy, average_max_utility_gppe_exp_noisy, \
    average_max_utility_interactive_ipro_exp_noisy, std_dev_utility_gppe_exp_noisy, std_dev_utility_interactive_ipro_exp_noisy, \
    std_dev_max_utility_gppe_exp_noisy, std_dev_max_utility_interactive_ipro_exp_noisy, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_utility_gppe_exp) + 1),
    average_utility_gppe_exp,
    yerr=std_dev_utility_gppe_exp,
    fmt='-o',
    capsize=5,
    color='b',
    label='GPPE (noise=0.01)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp) + 1),
    average_utility_interactive_ipro_exp,
    yerr=std_dev_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='r',
    label='PG-IPRO (noise=0.01)'
)
plt.errorbar(
    np.arange(1, len(average_utility_gppe_exp_noisy) + 1),
    average_utility_gppe_exp_noisy,
    yerr=std_dev_utility_gppe_exp_noisy,
    fmt='--o',
    capsize=5,
    color='c',
    label='GPPE (noise=0.2)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_noisy) + 1),
    average_utility_interactive_ipro_exp_noisy,
    yerr=std_dev_utility_interactive_ipro_exp_noisy,
    fmt='--o',
    capsize=5,
    color='m',
    label='PG-IPRO (noise=0.2)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average route utility comparison between GPPE and PG-IPRO for different noise levels")
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
    label='GPPE (noise=0.01)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1),
    average_max_utility_interactive_ipro_exp,
    yerr=std_dev_max_utility_interactive_ipro_exp,
    fmt='-o',
    capsize=5,
    color='r',
    label='PG-IPRO (noise=0.01)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_gppe_exp_noisy) + 1),
    average_max_utility_gppe_exp_noisy,
    yerr=std_dev_max_utility_gppe_exp_noisy,
    fmt='--o',
    capsize=5,
    color='c',
    label='GPPE (noise=0.2)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_noisy) + 1),
    average_max_utility_interactive_ipro_exp_noisy,
    yerr=std_dev_max_utility_interactive_ipro_exp_noisy,
    fmt='--o',
    capsize=5,
    color='m',
    label='PG-IPRO (noise=0.2)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_max_utility_interactive_ipro_exp) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average maximum utility comparison between GPPE and PG-IPRO for different noise levels")
plt.legend()
plt.grid(True)
plt.show()
