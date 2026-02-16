import matplotlib.pyplot as plt

from ipro.outer_loops.ipro import IPRO
from ipro.ipro_test import TestOracle, TestSolver


problem_id = 'test_user_interaction'
dimensions = 4
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

pf_not_so_balanced = [
    (2.0, 8.0), (2.21, 7.38), (2.41, 6.76), (2.62, 6.19), (2.83, 5.65),
    (3.03, 5.17), (3.24, 4.99), (3.45, 4.81), (3.66, 4.63), (3.86, 4.39),
    (4.07, 4.03), (4.28, 3.74), (4.48, 3.59), (4.69, 3.43), (4.90, 3.28),
    (5.10, 3.16), (5.31, 3.08), (5.52, 3.01), (5.72, 2.93), (5.93, 2.83),
    (6.14, 2.71), (6.34, 2.59), (6.55, 2.47), (6.76, 2.35), (6.97, 2.23),
    (7.17, 2.12), (7.38, 2.08), (7.59, 2.05), (7.79, 2.03), (8.0, 2.0)
]

concave_pf_30 = [
    (2.00, 8.00), (2.29, 7.90), (2.61, 7.83), (2.92, 7.75), (3.22, 7.66), (3.52, 7.55),
    (3.81, 7.43), (4.09, 7.30), (4.36, 7.15), (4.63, 6.99), (4.88, 6.82), (5.13, 6.64),
    (5.37, 6.45), (5.60, 6.25), (5.83, 6.04), (6.04, 5.82), (6.25, 5.59), (6.44, 5.35),
    (6.63, 5.11), (6.81, 4.85), (6.98, 4.59), (7.13, 4.33), (7.28, 4.05), (7.42, 3.77),
    (7.55, 3.49), (7.67, 3.20), (7.77, 2.90), (7.87, 2.60), (7.96, 2.30), (8.00, 2.00)
]

concave_pf_10 = [
    (2.00, 8.00),
    (3.22, 7.66),
    (4.36, 7.15),
    (5.13, 6.64),
    (5.83, 6.04),
    (6.25, 5.59),
    (6.63, 5.11),
    (7.13, 4.33),
    (7.67, 3.20),
    (8.00, 2.00)
]

linear_pf_30 = [
    (2.00, 8.00), (2.21, 7.79), (2.41, 7.59), (2.62, 7.38), (2.83, 7.17),
    (3.03, 6.97), (3.24, 6.76), (3.45, 6.55), (3.66, 6.34), (3.86, 6.14),
    (4.07, 5.93), (4.28, 5.72), (4.48, 5.52), (4.69, 5.31), (4.90, 5.10),
    (5.10, 4.90), (5.31, 4.69), (5.52, 4.48), (5.72, 4.28), (5.93, 4.07),
    (6.14, 3.86), (6.34, 3.66), (6.55, 3.45), (6.76, 3.24), (6.97, 3.03),
    (7.17, 2.83), (7.38, 2.62), (7.59, 2.41), (7.79, 2.21), (8.00, 2.00)
]

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

balanced_pf_4obj = [
 [4, 3, 3, 2], [3, 4, 5, 7], [6, 2, 5, 3], [4, 4, 2, 8], [2, 8, 8, 2], [2, 7, 8, 8],
 [5, 2, 2, 5], [7, 8, 2, 4], [8, 6, 2, 3], [4, 2, 8, 2], [4, 2, 3, 7], [7, 2, 3, 4],
 [2, 8, 6, 8], [3, 5, 8, 3], [3, 8, 5, 3], [3, 4, 6, 4], [3, 8, 3, 5], [3, 6, 5, 5],
 [2, 8, 7, 3], [8, 5, 2, 4], [8, 2, 3, 2], [4, 7, 2, 7], [3, 3, 6, 8], [3, 6, 8, 2],
 [3, 8, 7, 2], [3, 7, 6, 3], [3, 2, 8, 5], [3, 4, 4, 8], [4, 2, 6, 6], [5, 2, 4, 4]
]

# separate objectives to show scatterplot
obj1_values, obj2_values = zip(*linear_pf_30)
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


# upper bounds calculations: taking the max values of the pareto optimal paths calculated for each objective
# individually
upper_bnds = {'obj1': 8.0, 'obj2': 8.0, 'obj3': 8.0, 'obj4': 8.0}
print("initial upper_bounds: ", upper_bnds)
linear_solver = TestSolver(balanced_pf_4obj, upper_bnds)
oracle = TestOracle(balanced_pf_4obj, upper_bnds, heuristic='chebyshev')


'''
    IPRO with user input interaction
'''

ipro = IPRO(
    problem_id=problem_id,
    dimensions=dimensions,
    oracle=oracle,
    linear_solver=linear_solver,
    direction='minimize',
    max_iterations=10000,
    tolerance=0,
    user_interaction_loop=False,  # IPRO with user input interaction
    preferred_objective=None,
    referent_selection_heuristic='middle_distance'
)

processed_solutions = []  # keeping track of subsolutions for visualisation
user_input = None
while True:
    print("Enter ipro solve")
    subsolution = ipro.solve(user_input=user_input) #  subsolution of type list[tuple[np.ndarray, Any]],
                               #  but only returns 1 element list from solve for IPRO with user input interaction
    if subsolution is None:
        print("User preferences can't be improved, no better solution exists.")
        #break
    elif len(subsolution) != 1:  # at end of IPRO solve method it returns the whole pareto front,
        # this marks the end of the iteration/interaction loop since all possibilities have been processed
        current_pf = subsolution
        print("pf:", current_pf)
        print("pf size:", len(current_pf))
        objectives_vectors = [pf_sol[0] * -1 for pf_sol in current_pf]
        print("pf objective vectors: ", objectives_vectors)
        #break
    else:
        pareto_sol_objectives = subsolution[0][0] * -1  # * -1 because values returned are negative due to minimization problem semantics in IPRO
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
            # draw the scatterplot
            plt.figure(figsize=(6, 6))
            plt.scatter(obj1_values, obj2_values, label='processed non-dominated solutions')
            # show current solution
            plt.scatter(pareto_sol_objectives[0], pareto_sol_objectives[1], color='green', label='Current non-dominated solution')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('Pareto front')
            plt.xlim(1, 9)
            plt.ylim(1, 9)
            plt.legend()
            plt.grid(True)
            plt.show()
    # Query user input for next objective binary preference
    while True:
        objective = input("Choose which objective to decrease or increase: ")
        if not objective.isdigit():
            print("Invalid input! Please enter a valid objective (integer).")
            continue
        objective = int(objective)
        if not 0 <= objective < dimensions:  # check bounds of the present dimensions (objectives)
            print("Invalid input! Please enter a valid objective.")
            continue
        print("You chose objective :", objective)
        direction = input("Enter '-' to decrease or '+' to increase the objective: ")
        if direction not in ['-', '+']:
            print("Incorrect direction. Please enter either '-' or '+'.")
            continue
        print("You chose direction :", direction)
        user_input = (objective, direction)
        break


