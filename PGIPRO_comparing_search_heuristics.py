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


# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*balanced_pf_30)
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

# Manhattan distance for convex PF
linear_solver = TestSolver(balanced_pf_30, upper_bnds)
oracle = TestOracle(balanced_pf_30, upper_bnds, heuristic='manhattan')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize')
_, average_utility_interactive_ipro_exp_manhattan, _, average_max_utility_interactive_ipro_exp_manhattan, _, \
    std_dev_utility_interactive_ipro_exp_manhattan, _, std_dev_max_utility_interactive_ipro_exp_manhattan, _, _, _ = PGIPRO_vs_GPPE_experiment.run()

# Chebyshev distance for convex PF
linear_solver = TestSolver(balanced_pf_30, upper_bnds)
oracle = TestOracle(balanced_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize')
_, average_utility_interactive_ipro_exp_chebyshev, _, average_max_utility_interactive_ipro_exp_chebyshev, _, \
    std_dev_utility_interactive_ipro_exp_chebyshev, _, std_dev_max_utility_interactive_ipro_exp_chebyshev, _, _, _ = PGIPRO_vs_GPPE_experiment.run()

# Manhattan distance for concave PF
linear_solver = TestSolver(concave_pf_30, upper_bnds)
oracle = TestOracle(concave_pf_30, upper_bnds, heuristic='manhattan')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize')
_, average_utility_interactive_ipro_exp_manhattan_concave, _, average_max_utility_interactive_ipro_exp_manhattan_concave, _, \
    std_dev_utility_interactive_ipro_exp_manhattan_concave, _, std_dev_max_utility_interactive_ipro_exp_manhattan_concave, _, _, _ = PGIPRO_vs_GPPE_experiment.run()

# Chebyshev distance for concave PF
linear_solver = TestSolver(concave_pf_30, upper_bnds)
oracle = TestOracle(concave_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize')
_, average_utility_interactive_ipro_exp_chebyshev_concave, _, average_max_utility_interactive_ipro_exp_chebyshev_concave, _, \
    std_dev_utility_interactive_ipro_exp_chebyshev_concave, _, std_dev_max_utility_interactive_ipro_exp_chebyshev_concave, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_manhattan) + 1),
    average_utility_interactive_ipro_exp_manhattan,
    yerr=std_dev_utility_interactive_ipro_exp_manhattan,
    fmt='-o',
    capsize=5,
    color='orange',
    label='Manhattan heuristic (convex PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_chebyshev) + 1),
    average_utility_interactive_ipro_exp_chebyshev,
    yerr=std_dev_utility_interactive_ipro_exp_chebyshev,
    fmt='-o',
    capsize=5,
    color='c',
    label='Chebyshev heuristic (convex PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_manhattan_concave) + 1),
    average_utility_interactive_ipro_exp_manhattan_concave,
    yerr=std_dev_utility_interactive_ipro_exp_manhattan_concave,
    fmt='--o',
    capsize=5,
    color='r',
    label='Manhattan heuristic (concave PF)'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_chebyshev_concave) + 1),
    average_utility_interactive_ipro_exp_chebyshev_concave,
    yerr=std_dev_utility_interactive_ipro_exp_chebyshev_concave,
    fmt='--o',
    capsize=5,
    color='b',
    label='Chebyshev heuristic (concave PF)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_utility_interactive_ipro_exp_manhattan) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average route utility comparison between search heuristics in PG-IPRO (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_manhattan) + 1),
    average_max_utility_interactive_ipro_exp_manhattan,
    yerr=std_dev_max_utility_interactive_ipro_exp_manhattan,
    fmt='-o',
    capsize=5,
    color='orange',
    label='Manhattan heuristic (convex PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_chebyshev) + 1),
    average_max_utility_interactive_ipro_exp_chebyshev,
    yerr=std_dev_max_utility_interactive_ipro_exp_chebyshev,
    fmt='-o',
    capsize=5,
    color='c',
    label='Chebyshev heuristic (convex PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_manhattan_concave) + 1),
    average_max_utility_interactive_ipro_exp_manhattan_concave,
    yerr=std_dev_max_utility_interactive_ipro_exp_manhattan_concave,
    fmt='--o',
    capsize=5,
    color='r',
    label='Manhattan heuristic (concave PF)'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_chebyshev_concave) + 1),
    average_max_utility_interactive_ipro_exp_chebyshev_concave,
    yerr=std_dev_max_utility_interactive_ipro_exp_chebyshev_concave,
    fmt='--o',
    capsize=5,
    color='b',
    label='Chebyshev heuristic (concave PF)'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_max_utility_interactive_ipro_exp_manhattan) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average maximum utility comparison between search heuristics in PG-IPRO (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()