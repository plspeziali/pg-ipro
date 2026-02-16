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
utility_noise = .01

# upper bounds calculations: taking the max values of the pareto optimal solutions calculated for each objective
# individually
upper_bnds = {'obj1': 8.0, 'obj2': 8.0}  # specified/defined as is for this test case

# closest_distance referent selection heuristics for linear PF
linear_solver = TestSolver(linear_pf_30, upper_bnds)
oracle = TestOracle(linear_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='closest_distance',
                                                 tolerance=1e-8)
_, average_utility_interactive_ipro_exp_closest_dist, _, average_max_utility_interactive_ipro_exp_closest_dist, _, \
    std_dev_utility_interactive_ipro_exp_closest_dist, _, std_dev_max_utility_interactive_ipro_exp_closest_dist, _, _, _ = PGIPRO_vs_GPPE_experiment.run()

# middle_distance referent selection heuristics for linear PF
linear_solver = TestSolver(linear_pf_30, upper_bnds)
oracle = TestOracle(linear_pf_30, upper_bnds, heuristic='chebyshev')
PGIPRO_vs_GPPE_experiment = InteractiveIproVsGPE(num_queries=number_of_queries,
                                                 experiment_iterations=number_of_users,
                                                 dimensions=dimensions,
                                                 ipro_oracle=oracle,
                                                 ipro_linear_solver=linear_solver,
                                                 gppe_query_type='pairwise',
                                                 utility_noise=utility_noise,
                                                 direction='minimize',
                                                 referent_selection_heuristic='middle_distance',
                                                 tolerance=1e-8)
_, average_utility_interactive_ipro_exp_middle_dist, _, average_max_utility_interactive_ipro_exp_middle_dist, _, \
    std_dev_utility_interactive_ipro_exp_middle_dist, _, std_dev_max_utility_interactive_ipro_exp_middle_dist, _, _, _ = PGIPRO_vs_GPPE_experiment.run()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_closest_dist) + 1),
    average_utility_interactive_ipro_exp_closest_dist,
    yerr=std_dev_utility_interactive_ipro_exp_closest_dist,
    fmt='-o',
    capsize=5,
    color='b',
    label='Closest_distance heuristic'
)
plt.errorbar(
    np.arange(1, len(average_utility_interactive_ipro_exp_middle_dist) + 1),
    average_utility_interactive_ipro_exp_middle_dist,
    yerr=std_dev_utility_interactive_ipro_exp_middle_dist,
    fmt='-o',
    capsize=5,
    color='r',
    label='Middle_distance heuristic'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_utility_interactive_ipro_exp_closest_dist) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average route utility comparison between referent selection heuristics in PG-IPRO (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_closest_dist) + 1),
    average_max_utility_interactive_ipro_exp_closest_dist,
    yerr=std_dev_max_utility_interactive_ipro_exp_closest_dist,
    fmt='-o',
    capsize=5,
    color='b',
    label='Closest_distance heuristic'
)
plt.errorbar(
    np.arange(1, len(average_max_utility_interactive_ipro_exp_middle_dist) + 1),
    average_max_utility_interactive_ipro_exp_middle_dist,
    yerr=std_dev_max_utility_interactive_ipro_exp_middle_dist,
    fmt='-o',
    capsize=5,
    color='r',
    label='Middle_distance heuristic'
)
plt.ylim(-0.1, 1.1)
plt.xticks(np.arange(1, len(average_max_utility_interactive_ipro_exp_closest_dist) + 1))
plt.xlabel("Solution (querying between points)")
plt.ylabel("Utility")
plt.title("Average maximum utility comparison between referent selection heuristics in PG-IPRO (util_noise=0.01)")
plt.legend()
plt.grid(True)
plt.show()
