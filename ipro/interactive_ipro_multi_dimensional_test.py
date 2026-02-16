import matplotlib.pyplot as plt

from ipro.outer_loops.ipro import IPRO
from ipro.ipro_test import TestOracle, TestSolver

problem_id = 'test_3D_user_interaction'
dimensions = 3

pf_3d_with_2_different_single_objective_optimizer_sols = [
    (9.0, 1.0, 1.5),
    (8.2, 2.0, 2.5),
    (7.5, 3.5, 4.0),
    (6.5, 4.0, 5.5),
    (5.8, 5.0, 6.2),
    (5.0, 5.5, 7.0),
    (4.0, 6.0, 8.5),
    (3.2, 6.5, 7.8),
    (2.5, 7.0, 6.5),
    (1.8, 7.5, 5.5),
]

pf_3d_with_3_different_single_objective_optimizer_sols = [
    (1.0, 8.0, 8.0),
    (2.0, 7.0, 7.5),
    (3.0, 6.0, 7.0),
    (4.0, 5.0, 6.0),
    (5.0, 4.0, 5.0),
    (6.0, 3.0, 4.0),
    (7.0, 2.5, 3.0),
    (8.0, 2.0, 2.5),
    (8.5, 1.0, 2.0),
    (8.0, 8.5, 1.0),
]

wrong_pareto_front_with_3_different_single_objective_optimizer_sols = [ # problem: Mostly convex (small minor sharp edges near minimized objectives)
    (9.0, 1.0, 1.5),    # Objective 2 small
    (8.2, 2.0, 2.5),
    (7.5, 0.5, 4.0),    # Objective 2 minimized (very small)
    (6.5, 4.0, 5.5),
    (5.8, 5.0, 1.2),    # Objective 3 minimized (very small)
    (5.0, 5.5, 7.0),
    (4.0, 6.0, 8.5),
    (3.2, 2.0, 7.8),    # Objective 2 minimized (again small at this point)
    (2.5, 7.0, 6.5),
    (1.8, 7.5, 5.5),
]

test_pareto_front = [
    (1.0, 8.0, 8.0),#        # minimal obj1
    (2.0, 7.5, 6.5),#
    (3.0, 6.0, 7.5),#
    (4.0, 5.5, 5.0),#
    (5.0, 5.0, 4.5),#
    (6.0, 4.5, 3.5),#
    (7.0, 3.0, 4.0),#
    (8.0, 4.5, 2.5),#
    (8.5, 1.0, 5.5),#        # minimal obj2
    (8.0, 8.5, 1.0),#        # minimal obj3
]

# upper bounds calculations: taking the max values of the pareto optimal paths calculated for each objective
# individually
upper_bnds = {'obj1': 8.5, 'obj2': 8.5, 'obj3': 8.0}
print("initial upper_bounds: ", upper_bnds)
linear_solver = TestSolver(test_pareto_front, upper_bnds)
oracle = TestOracle(test_pareto_front, upper_bnds)


'''
    IPRO with user input interaction
'''

ipro = IPRO(
    problem_id=problem_id,
    dimensions=dimensions,
    oracle=oracle,
    linear_solver=linear_solver,
    direction='minimize',
    max_iterations=1000,
    tolerance=0, #1e-1,
    user_interaction_loop=True,  # IPRO with user input interaction
    preferred_objective=None
)

processed_solutions = []
user_input = None
while True:
    print("Enter ipro solve")
    subsolution = ipro.solve(user_input=user_input) #  subsolution of type list[tuple[np.ndarray, Any]],
                               #  but only returns 1 element list from solve for IPRO with user input interaction
    if subsolution is None:
        print("User preferences can't be improved, no better solution exists.")
        break
    elif len(subsolution) != 1:  # at end of IPRO solve method it returns the whole pareto front,
                               # this marks the end of the iteration/interaction loop since all possibilities have been processed
        print("pf:", subsolution)
        print("Done!")
        break
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
        # ONLY MINIMIZATION '-' WORKS!
        if direction not in ['-', '+']:
            print("Incorrect direction. Please enter either '-' or '+'.")
            continue
        print("You chose direction :", direction)
        user_input = (objective, direction)
        break


