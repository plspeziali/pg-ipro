import numpy as np

from ipro.oracles.oracle import Oracle

from ipro.linear_solvers.linear_solver import LinearSolver


class TestSolver(LinearSolver):
    def __init__(self, pareto_front, upper_bounds):
        super().__init__(problem=pareto_front, direction="minimize")
        self.pareto_front = pareto_front
        self.upper_bounds = upper_bounds

    def solve(self, weight_vec):
        """Every edge has a cost, let's make the dot product of the vector cost to make it into a scalar cost"""
        if np.sum(weight_vec) < 0:
            bounds = np.array(list(self.upper_bounds.values()))
            return bounds, bounds
        weighted_pareto_front = [(np.dot(weight_vec, np.array(pf_sol)), pf_sol) for pf_sol in self.pareto_front]
        vec_return = np.array(min(weighted_pareto_front, key=lambda x: x[0])[1])
        return vec_return, vec_return.copy()


class TestOracle(Oracle):
    def __init__(self, pareto_front, upper_bounds, heuristic='chebyshev', problem=None, seed=None):
        super().__init__(problem=problem, seed=seed)
        self.pareto_front = pareto_front
        self.upper_bounds = upper_bounds
        self.heuristic = heuristic

    def strict_pareto_dominate(self, objectives1, objectives2):
        """Returns True if objectives1 strictly pareto-dominates objectives2."""
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 >= obj2:
                return False
        return True

    def pareto_dominate(self, objectives1, objectives2):
        dominates_in_one = False
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 > obj2:
                return False
            elif obj1 < obj2:
                dominates_in_one = True
        return dominates_in_one

    def manhattan_distance(self, ref, target_obj, pareto_front_sol):
        """
        Calculate the normalized Manhattan (L1) distance between two objective vectors.
        """
        denom = np.maximum(ref - target_obj, 1e-8)  # avoid division by zero
        normalized_diff = np.abs(pareto_front_sol - target_obj) / denom
        return np.sum(normalized_diff)

    def chebyshev_asf(self, ref, target_obj, pareto_front_sol):
        denom = np.maximum(ref - target_obj, 1e-8)  # avoid division by zero
        normalized_diff = np.abs(pareto_front_sol - target_obj) / denom
        return np.max(normalized_diff)

    def solve(self, referent, nadir, ideal, upper_point_target_region):

        ideal_vector = upper_point_target_region if referent.size == 2 else ideal  # Ideal used for higher dimensions
        if self.heuristic == 'chebyshev':
            distances_pareto_solutions = [
                (self.chebyshev_asf(referent, ideal_vector, pareto_front_sol), pareto_front_sol)
                for pareto_front_sol in self.pareto_front
            ]
        elif self.heuristic == 'manhattan':
            distances_pareto_solutions = [
                (self.manhattan_distance(referent, ideal_vector, pareto_front_sol), pareto_front_sol)
                for pareto_front_sol in self.pareto_front
            ]
        else:
            raise ValueError(f'Unknown heuristic {self.heuristic}')
        for distance, pareto_front_sol in sorted(distances_pareto_solutions, key=lambda x: x[0]):
            pareto_front_sol = np.array(pareto_front_sol)
            if self.strict_pareto_dominate(pareto_front_sol, referent):
                #print("oracle solve returns pf_sol: ", pareto_front_sol)
                return pareto_front_sol, pareto_front_sol.copy()
        print("No solution found anymore for this referent/Test oracle initialized with incorrect pareto_front")
        return np.array(referent), None  # sol returned by oracle should be None if no pareto_sol can be found
