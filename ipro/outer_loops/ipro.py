import numpy as np

from typing import Optional

from ipro.oracles.oracle import Oracle
from ipro.linear_solvers.linear_solver import LinearSolver
from ipro.outer_loops.typing import Subproblem, Subsolution
from ipro.outer_loops.outer import OuterLoop
from ipro.outer_loops.box import Box
from ipro.utils.pareto import strict_pareto_dominates, batched_strict_pareto_dominates, extreme_prune, pareto_dominates

class IPRO(OuterLoop):
    """IPRO algorithm for solving multi-objective problems."""

    def __init__(
            self,
            problem_id: str,
            dimensions: int,
            oracle: Oracle,
            linear_solver: LinearSolver,
            direction: str = 'maximize',
            ref_point: Optional[np.ndarray] = None,
            offset: float = 1,
            tolerance: float = 1e-1,
            max_iterations: Optional[int] = None,
            update_freq: int = 1,
            known_pf: Optional[np.ndarray] = None,
            track: bool = False,
            exp_name: Optional[str] = None,
            wandb_project_name: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None,
            extra_config: Optional[dict] = None,
            user_interaction_loop: bool = False,
            preferred_objective: int = None,
            referent_selection_heuristic: str = 'closest_distance'
    ):
        super().__init__(
            problem_id,
            dimensions,
            oracle,
            linear_solver,
            method="IPRO",
            direction=direction,
            ref_point=ref_point,
            offset=offset,
            tolerance=tolerance,
            max_iterations=max_iterations,
            known_pf=known_pf,
            track=track,
            exp_name=exp_name,
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity,
            seed=seed,
            extra_config=extra_config,
            user_interaction_loop=user_interaction_loop,
            preferred_objective=preferred_objective
        )
        self.update_freq = update_freq
        self.lower_points = []
        self.upper_points = []

        self.referent_selection_heuristic = referent_selection_heuristic

        self.rng = rng if rng is not None else np.random.default_rng(seed)

        if user_interaction_loop and preferred_objective is not None and not -1 <= preferred_objective < dimensions:
            # -1 indicates no preferred objective --> random subsolution
            raise ValueError("Unknown preferred_objective")

    def reset(self):
        """Reset the algorithm."""
        self.lower_points = []
        self.upper_points = []
        super().reset()

    def init_phase(self) -> tuple[list[Subsolution], bool]:
        """Run the initialisation phase of the algorithm.

        This phase computes the bounding box of the Pareto front by solving the maximisation and minimisation problems
        for all objectives. For the ideal, this is exact and for the nadir this is guaranteed to be a pessimistic
        estimate. The lower points are then computed and initial hypervolume improvements.

        Returns:
            bool: True if the Pareto front is the ideal point, False otherwise.
        """
        nadir = np.zeros(self.dim)
        ideal = np.zeros(self.dim)
        pf = []
        subsolutions = []
        weight_vecs = np.eye(self.dim)

        for i, weight_vec in enumerate(weight_vecs):
            ideal_vec, ideal_sol = self.linear_solver.solve(weight_vec)
            nadir_vec, _ = self.linear_solver.solve(-1 * weight_vec)
            ideal_vec *= self.sign
            nadir_vec *= self.sign
            ideal[i] = ideal_vec[i]
            nadir[i] = nadir_vec[i]
            pf.append(ideal_vec)
            subsolutions.append((weight_vec, ideal_vec, ideal_sol))

        self.pf = extreme_prune(np.array(pf))
        nadir = nadir - self.offset  # Necessary to ensure every Pareto optimal point strictly dominates the nadir.
        ideal = ideal + self.offset

        self.nadir = np.copy(nadir)
        self.ideal = np.copy(ideal)
        self.ref_point = np.copy(nadir) if self.ref_point is None else np.array(self.ref_point)
        self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)

        if len(self.pf) == 1:  # If the Pareto front is the ideal.
            return subsolutions, True

        self.bounding_box = Box(nadir, ideal)
        self.total_hv = self.bounding_box.volume
        self.lower_points = np.array([nadir])

        for point in self.pf:  # Initialise the lower points.
            self.update_lower_points(np.array(point))

        self.upper_points = np.array([ideal])  # Initialise the upper points.

        self.error = max(ideal - nadir)
        self.compute_hvis()
        self.oracle.init_oracle(nadir=self.sign * self.nadir, ideal=self.sign * self.ideal)  # Initialise the oracle.
        return subsolutions, False

    def compute_hvis(self, num=50):
        """Compute the hypervolume improvements of the lower points.

        Note:
            An optional num parameter can be given as computing the hypervolume for a large number of potential points
            is expensive.

        Args:
            num (int, optional): The number of lower points to compute hypervolume improvements for. Defaults to 50.
        """
        discarded_extrema = np.vstack((self.pf, self.completed))
        hvis = np.zeros(len(self.lower_points))

        for lower_id in self.rng.choice(len(self.lower_points), min(num, len(self.lower_points)), replace=False):
            hv = self.compute_hypervolume(np.vstack((discarded_extrema, self.lower_points[lower_id])), self.ideal)
            hvis[lower_id] = hv  # We don't have to compute the difference as it is proportional to the hypervolume.

        sorted_args = np.argsort(hvis)[::-1]
        self.lower_points = self.lower_points[sorted_args]

    def max_hypervolume_improvement(self):
        """Recompute the hypervolume improvements and return the point that maximises it..

        Returns:
            np.array: The point that maximises the hypervolume improvement.
        """
        self.compute_hvis()
        return self.lower_points[0]

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        if len(self.upper_points) == 0:
            error = 0
        else:
            pf = np.array(list(self.pf))
            diffs = self.upper_points[:, None, :] - pf[None, :, :]
            error = np.max(np.min(np.max(diffs, axis=2), axis=1))
        self.error = error

    def update_upper_points(self, vec):
        """Update the upper set.

        Args:
            vec (np.array): The point that is added to the boundary of the dominated space.
        """
        strict_dominates = batched_strict_pareto_dominates(self.upper_points, vec)
        to_keep = self.upper_points[strict_dominates == 0]
        shifted = np.stack([self.upper_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(shifted > self.nadir, axis=-1)]

        new_upper_points = np.vstack((to_keep, shifted))
        self.upper_points = extreme_prune(new_upper_points)

    def update_lower_points(self, vec):
        """Update the lower set.

        Args:
            vec (np.array): The point that is added to the boundary of the dominated space.
        """
        strict_dominates = batched_strict_pareto_dominates(vec, self.lower_points)
        to_keep = self.lower_points[strict_dominates == 0]
        shifted = np.stack([self.lower_points[strict_dominates == 1]] * self.dim)
        shifted[range(self.dim), :, range(self.dim)] = np.expand_dims(vec, -1)
        shifted = shifted.reshape(-1, self.dim)
        shifted = shifted[np.all(self.ideal > shifted, axis=-1)]

        new_lower_points = np.vstack((to_keep, shifted))
        self.lower_points = -extreme_prune(-new_lower_points)

    def select_referent(self, method='random', user_input: tuple[int, str] = None, heuristic: str ='middle_distance'):
        """The method to select a new referent.

        Args:
            method (str): The method to find a next referent.
            user_input (tuple): A tuple with the objective and the optimize preference to increase '+' or decrease '-'
            it.
        """
        if method == 'random':
            return self.lower_points[self.rng.integers(0, len(self.lower_points))]
        if method == 'first':
            return self.lower_points[0]
        if method == 'user_interaction_loop':
            if user_input is None:  # initialization iteration in Outer loop solve has no queried information from user
                return None
            objective = user_input[0]
            direction = user_input[1]  # only minimization works for now

            improving_set = []
            # form improving set
            for lower_point in self.lower_points:
                if direction == '-' and self.sign * lower_point[objective] <= self.sign * self.last_subsolution_vec[objective]:
                    improving_set.append(lower_point)

            if not improving_set:
                return None
            ideal_objective_i_solution_vec = self.all_init_phase_linear_subsolutions[objective][1]
            reference_point = ((ideal_objective_i_solution_vec + self.last_subsolution_vec) / 2) if \
                self.referent_selection_heuristic == 'middle_distance' else self.last_subsolution_vec
            if self.referent_selection_heuristic == 'middle_distance' or self.referent_selection_heuristic == 'closest_distance':
                ideal_objective_i_solution_vec = self.all_init_phase_linear_subsolutions[objective][1]
                epsilon = 1e-8  # avoid division by zero for floating point precision errors
                range_per_obj = np.abs(ideal_objective_i_solution_vec - self.last_subsolution_vec) + epsilon
                def calculate_distance(lower_point):
                    normalized_diff = np.abs(lower_point - reference_point) / range_per_obj
                    return np.sum(normalized_diff)
                return min(improving_set, key=calculate_distance)
            else:
                raise ValueError(f'Unknown heuristic {self.referent_selection_heuristic}')
        else:
            raise ValueError(f'Unknown method {method}')

    def get_iterable_for_replay(self):
        return np.copy(self.lower_points)

    def maybe_add_solution(
            self,
            subproblem: Subproblem,
            point: np.ndarray,
            lower: np.ndarray,
    ) -> Subproblem | bool:
        if strict_pareto_dominates(point, lower):
            new_subproblem = Subproblem(referent=lower, nadir=self.nadir, ideal=self.ideal)
            self.update_found(new_subproblem, point)
            return new_subproblem
        else:
            return False

    def maybe_add_completed(
            self,
            subproblem: Subproblem,
            point: np.ndarray,
            lower: np.ndarray,
    ) -> Subproblem | bool:
        if pareto_dominates(lower, subproblem.referent):
            new_subproblem = Subproblem(referent=lower, nadir=self.nadir, ideal=self.ideal)
            self.update_not_found(new_subproblem, point)
            return new_subproblem
        else:
            return False

    def update_found(self, subproblem, vec):
        """The update to perform when the Pareto oracle found a new Pareto dominant vector."""
        self.pf = np.vstack((self.pf, vec))
        self.update_lower_points(vec)
        self.update_upper_points(vec)

    def update_not_found(self, subproblem, vec):
        """The update to perform when the Pareto oracle did not find a new Pareto dominant vector."""
        self.completed = np.vstack((self.completed, subproblem.referent))
        self.lower_points = self.lower_points[np.any(self.lower_points != subproblem.referent, axis=1)]
        self.update_upper_points(subproblem.referent)
        if strict_pareto_dominates(vec, self.nadir):
            self.robust_points = np.vstack((self.robust_points, vec))

    def decompose_problem(self, iteration, method='first', user_input: tuple[int, str] = None):
        if iteration % self.update_freq == 0:
            self.compute_hvis()
        referent = self.select_referent(method=method, user_input=user_input)
        subproblem = Subproblem(referent=referent, nadir=self.nadir, ideal=self.ideal)
        return subproblem

    def update_excluded_volume(self):
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.discarded_hv = self.compute_hypervolume(np.vstack((self.pf, self.completed)), self.ideal)

    def find_dominated_referent(self, vec):
        for referent in self.lower_points:
            if strict_pareto_dominates(vec, referent):
                return referent

    def find_dominating_upper_point(self, referent):
        dominating_upper_points = []
        for upper_point in self.upper_points:
            if strict_pareto_dominates(upper_point, referent):
                dominating_upper_points.append(upper_point)
        if dominating_upper_points:
            return self.rng.choice(dominating_upper_points)
        else:
            return None
