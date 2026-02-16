import time
import random
import wandb
import platform
import numpy as np

from typing import Optional, Iterable, Any
from ipro.outer_loops.typing import Subproblem, Subsolution, IPROCallback

from pymoo.indicators.hv import Hypervolume
from pymoo.config import Config

from ipro.oracles.oracle import Oracle
from ipro.linear_solvers.linear_solver import LinearSolver
from ipro.utils.pareto import (
    strict_pareto_dominates,
    batched_strict_pareto_dominates,
    extreme_prune,
    batched_pareto_dominates
)

Config.warnings['not_compiled'] = False


class OuterLoop:
    def __init__(
            self,
            problem_id: str,
            dimensions: int,
            oracle: Oracle,
            linear_solver: LinearSolver,
            method: str = "IPRO",
            direction: str = "maximize",  # "minimize" or "maximize
            ref_point: Optional[np.ndarray] = None,
            offset: float = 1,
            tolerance: float = 1e-1,
            max_iterations: Optional[int] = None,
            known_pf: Optional[np.ndarray] = None,
            track: bool = False,
            exp_name: Optional[str] = None,
            wandb_project_name: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            seed: Optional[int] = None,
            extra_config: Optional[dict] = None,
            user_interaction_loop: bool = False,
            preferred_objective: int = None,
    ):
        self.problem_id = problem_id
        self.dim = dimensions
        self.oracle = oracle
        self.linear_solver = linear_solver
        self.method = method
        self.direction = direction
        self.ref_point = ref_point
        self.offset = offset
        self.tolerance = tolerance
        self.max_iterations = max_iterations if max_iterations is not None else np.inf
        self.known_pf = known_pf

        self.sign = 1 if direction == "maximize" else -1
        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = None
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.hv = 0
        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf
        self.replay_triggered = 0

        self.track = track
        self.run_id = None
        self.exp_name = exp_name
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity

        self.seed = seed

        self.extra_config = extra_config

        # interactive IPRO (Preference-Guided IPRO) state variables
        self.user_interaction_loop = user_interaction_loop
        self.user_interaction_loop_started = False
        self.last_subsolution_vec = None
        self.init_phase_subsolutions = None
        self.all_init_phase_linear_subsolutions = None

        self.linear_subsolutions = []
        self.subsolutions = []
        self.done = False
        self.iteration = 0
        self.start = None

        self.preferred_objective = preferred_objective

    def reset(self):
        self.bounding_box = None
        self.ideal = None
        self.nadir = None
        self.pf = np.empty((0, self.dim))
        self.robust_points = np.empty((0, self.dim))
        self.completed = np.empty((0, self.dim))

        self.hv = 0
        self.total_hv = 0
        self.dominated_hv = 0
        self.discarded_hv = 0
        self.coverage = 0
        self.error = np.inf
        self.replay_triggered = 0

    def config(self) -> dict:
        """Get the config of the algorithm."""
        extra_config = self.extra_config if self.extra_config is not None else {}
        return {
            "method": self.method,
            "problem_id": self.problem_id,
            "dimensions": self.dim,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
            **extra_config
        }

    def setup(self, mode: str = 'online') -> float:
        """Setup wandb."""
        config = self.config()
        config.update(self.oracle.config())

        print(f'Running with config: {config}')

        if self.track:
            location = platform.platform()

            if location.startswith('Linux-6.6.22-frehi12'):  # Hack to check where the code is running.
                location = 'ailab'
            elif location.startswith('macOS'):
                location = 'mac'
            else:
                location = 'vub'

            if location == 'vub':
                wandb.init(
                    settings=wandb.Settings(log_internal=str('/scratch/brussel/103/vsc10340/wandb/null'), ),
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                    mode=mode,
                )
            else:
                wandb.init(
                    project=self.wandb_project_name,
                    entity=self.wandb_entity,
                    config=config,
                    name=self.exp_name,
                    mode=mode,
                )

            wandb.define_metric('iteration')
            wandb.define_metric('outer/hypervolume', step_metric='iteration')
            wandb.define_metric('outer/dominated_hv', step_metric='iteration')
            wandb.define_metric('outer/discarded_hv', step_metric='iteration')
            wandb.define_metric('outer/coverage', step_metric='iteration')
            wandb.define_metric('outer/error', step_metric='iteration')
            self.run_id = wandb.run.id

        return time.time()

    def get_pareto_set(self, subsolutions: list[Subsolution]) -> list[tuple[np.ndarray, Any]]:
        """Get the Pareto set from the subsolutions."""
        pareto_set = []
        for subsolution in subsolutions:
            if np.any(np.all(np.isclose(subsolution[1], self.pf), axis=1)):
                pareto_set.append((subsolution[1], subsolution[2]))
        return pareto_set

    def finish(self, start_time: float, iteration: int):
        """Finish the algorithm."""
        self.pf = extreme_prune(np.vstack((self.pf, self.robust_points)))
        self.dominated_hv = self.compute_hypervolume(-self.pf, -self.nadir)
        self.hv = self.compute_hypervolume(-self.pf, -self.ref_point)
        self.log_iteration(iteration + 1)

        end_str = f'Iterations {iteration + 1} | Time {time.time() - start_time:.2f} | '
        end_str += f'HV {self.hv:.2f} | PF size {len(self.pf)} |'
        print(end_str)

        self.close_wandb()

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            pf_table = wandb.Table(data=self.pf, columns=[f'obj_{i}' for i in range(self.dim)])
            wandb.run.log({'pareto_front': pf_table})
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.finish()

    def log_iteration(
            self,
            iteration: int,
            subproblem: Optional[Subproblem] = None,
            pareto_point: Optional[np.ndarray] = None
    ):
        """Log the iteration."""
        if self.track:
            while True:
                try:
                    wandb.log({
                        'outer/hypervolume': self.hv,
                        'outer/dominated_hv': self.dominated_hv,
                        'outer/discarded_hv': self.discarded_hv,
                        'outer/coverage': self.coverage,
                        'outer/error': self.error,
                        'iteration': iteration
                    })
                    break
                except wandb.Error as e:
                    print(f"wandb got error {e}")
                    time.sleep(random.randint(10, 100))

            if subproblem is not None:
                wandb.run.summary[f"referent_{iteration}"] = self.sign * subproblem.referent
                wandb.run.summary[f"ideal_{iteration}"] = self.sign * subproblem.ideal
                wandb.run.summary[f"pareto_point_{iteration}"] = self.sign * pareto_point

            wandb.run.summary['hypervolume'] = self.hv
            wandb.run.summary['PF_size'] = len(self.pf)
            wandb.run.summary['replay_triggered'] = self.replay_triggered

    def compute_hypervolume(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Compute the hypervolume of a set of points.

        Note:
            This computes the hypervolume assuming all objectives are to be minimized.

        Args:
            points (array_like): List of points.
            ref (np.array): Reference point.

        Returns:
            float: The computed hypervolume.
        """
        points = points[batched_pareto_dominates(ref, points)]
        if points.size == 0:
            return 0
        ind = Hypervolume(ref_point=ref)
        return ind(points)

    def init_phase(self) -> tuple[list[Subsolution], bool]:
        """Initialize the outer loop."""
        raise NotImplementedError

    def is_done(self, step: int) -> bool:
        """Check if the algorithm is done."""
        print("1 - self.coverage <= self.tolerance: ", 1 - self.coverage <= self.tolerance)
        print("self.coverage: ", self.coverage)
        print("self.tolerance: ", self.tolerance)
        return 1 - self.coverage <= self.tolerance or step >= self.max_iterations

    def decompose_problem(self, iteration: int, method: str = 'first') -> Subproblem:
        """Decompose the problem into a subproblem."""
        raise NotImplementedError

    def find_dominated_referent(self, vec):
        """Find dominated referent for the balanced initialization phase subsolution when user_interaction_loop."""
        raise NotImplementedError

    def find_dominating_upper_point(self, referent):
        """Find dominating upper_point for a given referent to do normalization in oracle for a given target region."""
        raise NotImplementedError

    def update_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when a Pareto optimal solution is found."""
        raise NotImplementedError

    def update_not_found(self, subproblem: Subproblem, vec: np.ndarray):
        """The update that is called when no Pareto optimal solution is found."""
        raise NotImplementedError

    def update_excluded_volume(self):
        """Update the dominated and infeasible sets."""
        raise NotImplementedError

    def estimate_error(self):
        """Estimate the error of the algorithm."""
        raise NotImplementedError

    def get_iterable_for_replay(self) -> Iterable[Any]:
        raise NotImplementedError

    def maybe_add_solution(
            self,
            subproblem: Subproblem,
            vec: np.ndarray,
            item: Any,
    ) -> Subproblem | bool:
        raise NotImplementedError

    def maybe_add_completed(
            self,
            subproblem: Subproblem,
            vec: np.ndarray,
            item: Any,
    ) -> Subproblem | bool:
        raise NotImplementedError

    def replay(
            self,
            vec: np.ndarray,
            sol: Any,
            iter_pairs: list[Subsolution]
    ) -> tuple[list[Subsolution], list[Subsolution]]:
        """Replay the algorithm while accounting for the non-optimal Pareto oracle.

        Note:
            This reexecutes the initialisation phase which may trigger expensive compute again. However, we always use
            a given box in the experiments, so this makes no difference **in this specific case**.

        Args:
            vec (ndarray): The vector that causes the conflict.
            iter_pairs (list[Subsolution]): A list of subsolutions.

        Returns:
            An updated list of referent point tuples.
        """
        print('REPLAY TRIGGERED')
        replay_triggered = self.replay_triggered
        self.reset()
        self.replay_triggered = replay_triggered + 1
        new_init_subsolutions, _ = self.init_phase()
        idx = 0
        new_subsolutions = []

        for old_subproblem, old_vec, old_sol in iter_pairs:  # Replay the points that were added correctly
            idx += 1
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                if strict_pareto_dominates(vec, old_vec):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_sol))
            else:
                if strict_pareto_dominates(vec, old_subproblem.referent):
                    self.update_found(old_subproblem, vec)
                    new_subsolutions.append((old_subproblem, vec, sol))
                    break
                else:
                    self.update_not_found(old_subproblem, old_vec)
                    new_subsolutions.append((old_subproblem, old_vec, old_vec))

        for old_subproblem, old_vec, old_sol in iter_pairs[idx:]:  # Process the remaining points to see if we can still add them.
            items = self.get_iterable_for_replay()
            if strict_pareto_dominates(old_vec, old_subproblem.referent):
                maybe_add = self.maybe_add_solution
            else:
                maybe_add = self.maybe_add_completed
            for item in items:
                res = maybe_add(old_subproblem, old_vec, item)
                if res:
                    new_subsolutions.append((res, old_vec, old_sol))
                    break

        return new_init_subsolutions, new_subsolutions

    def solve(self, return_inter_sol: bool = False, callback: Optional[IPROCallback] = None, user_input: Optional[tuple[int, str]] = None) -> Optional[list[tuple[np.ndarray, Any]]]:
        """Solve the problem."""
        if self.done:
            print('The problem is already solved in the initial phase.')
            pareto_set = self.get_pareto_set(self.linear_subsolutions)
            return pareto_set
        init_iter = not self.user_interaction_loop_started
        if not self.user_interaction_loop_started:
            self.start = self.setup()
            self.linear_subsolutions, self.done = self.init_phase()
            self.init_phase_subsolutions = self.linear_subsolutions.copy()
            self.all_init_phase_linear_subsolutions = self.linear_subsolutions.copy()

            self.log_iteration(self.iteration)

            pareto_set = self.get_pareto_set(self.linear_subsolutions)

            if self.done:
                print('The problem is solved in the initial phase.')
                return pareto_set
            elif self.user_interaction_loop:
                self.user_interaction_loop_started = True
                # return one of the first (linear) subsolutions as starting point for the Preference-Guided interaction
                if self.preferred_objective is not None:
                    if self.preferred_objective == -1: # -1 to randomly take one of extrema solutions optimized for one particular objective.
                        obj_idx = random.randrange(self.dim)
                    else:
                        obj_idx = self.preferred_objective
                    self.last_subsolution_vec = pareto_set[obj_idx][0]
                    self.init_phase_subsolutions.pop(obj_idx)
                    return [pareto_set[obj_idx]]

        # In the case of 'user_interaction_loop', PF might return to soon with 'self.is_done', since the
        # linear_subsolutions already account for the coverage but one of these might be the last to be generated in
        # the user_interaction_loop
        if self.user_interaction_loop:
            subproblem = self.decompose_problem(self.iteration, method='user_interaction_loop', user_input=user_input)
            # check for viable linear_subsolution yet to be returned by the user_interaction_loop (after initial
            # iteration of the user_interaction_loop, since first a balanced subsolution will be returned)
            if not init_iter and subproblem.referent is None:
                objective = user_input[0]
                for index, (weight_vec, ideal_vec, ideal_sol) in enumerate(self.init_phase_subsolutions):
                    if ideal_vec[objective] > self.last_subsolution_vec[objective]:
                        self.init_phase_subsolutions.pop(index)
                        self.last_subsolution_vec = ideal_vec
                        return [(ideal_vec, ideal_sol)]
                # User preferences can't be improved, no subsolution is possible in the given optimization direction of an objective
                print("User preferences can't be improved, no referent is available in the given optimization direction of an objective")
                return None

        if return_inter_sol:  # return_inter_sol=True will return each subsolution even for regular IPRO (self.user_interaction_loop=False)
            self.user_interaction_loop_started = True

        while not self.is_done(self.iteration):
            begin_loop = time.time()
            print(f'Iter {self.iteration} - Covered {self.coverage:.5f}% - Error {self.error:.5f}')

            if not self.user_interaction_loop:
                subproblem = self.decompose_problem(self.iteration, method='first', user_input=user_input)

            if self.user_interaction_loop and init_iter and self.preferred_objective is None:
                # first return a (balanced) subsolution based on the search heuristic in the oracle:
                vec, sol = self.oracle.solve(
                    self.sign * subproblem.nadir,
                    nadir=self.sign * subproblem.nadir,
                    ideal=self.sign * subproblem.ideal,
                    upper_point_target_region=self.sign * subproblem.ideal
                )
                vec *= self.sign
                # One of the linear_subsolutions can be the most balanced next subsolution, in case of a non-convex PF.
                # These subsolutions are already processed in the search space update in the init_phase of IPRO:
                for index, (weight_vec, ideal_vec, ideal_sol) in enumerate(self.init_phase_subsolutions):
                    if np.all(ideal_vec == vec):
                        self.init_phase_subsolutions.pop(index)
                        self.last_subsolution_vec = vec
                        return [(vec, sol)]
                # find appropriate lower_point that our new subsolution strict Pareto dominates for correct updates:
                subproblem.referent = self.find_dominated_referent(vec)
            else:  # appropriate for 2D problems
                upper_point_target_region = self.find_dominating_upper_point(subproblem.referent)
                if upper_point_target_region is not None:
                    upper_point_target_region = self.sign * upper_point_target_region
                vec, sol = self.oracle.solve(
                    self.sign * subproblem.referent,
                    nadir=self.sign * subproblem.nadir,
                    ideal=self.sign * subproblem.ideal,
                    upper_point_target_region=upper_point_target_region
                )
                vec *= self.sign

            if strict_pareto_dominates(vec, subproblem.referent):
                if np.any(batched_strict_pareto_dominates(vec, np.vstack((self.pf, self.completed)))):
                    self.linear_subsolutions, self.subsolutions = self.replay(vec, sol, self.subsolutions)
                else:
                    self.last_subsolution_vec = vec
                    self.update_found(subproblem, vec)
                    self.subsolutions.append((subproblem, vec, sol))
            else:
                if np.any(batched_strict_pareto_dominates(vec, self.completed)):
                    self.linear_subsolutions, self.subsolutions = self.replay(vec, sol, self.subsolutions)
                else:
                    print("Oracle did not find a solution which strictly dominates the referent: ", subproblem.referent)
                    self.update_not_found(subproblem, vec)
                    self.subsolutions.append((subproblem, vec, sol))

            self.update_excluded_volume()
            self.estimate_error()
            self.coverage = (self.dominated_hv + self.discarded_hv) / self.total_hv
            self.hv = self.compute_hypervolume(-self.sign * self.pf, -self.sign * self.ref_point)

            self.iteration += 1
            self.log_iteration(self.iteration, subproblem=subproblem, pareto_point=vec)

            if callback is not None:
                callback(self.iteration, self.hv, self.dominated_hv, self.discarded_hv, self.coverage, self.error)

            duration = time.time() - begin_loop

            print(f'Ref {self.sign * subproblem.referent} - Found {self.sign * vec} - Time {duration:.2f}s')
            print('---------------------')

            # Return intermediate result:
            if self.user_interaction_loop:
                return [(vec, sol)]
            if return_inter_sol:
                return [(vec, sol)]

        print("Finished IPRO, returning PF...")
        self.finish(self.start, self.iteration)
        pareto_set = self.get_pareto_set(self.linear_subsolutions + self.subsolutions)
        return pareto_set
