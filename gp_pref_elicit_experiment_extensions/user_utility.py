from gp_pref_elicit.gp_utilities.utils_user import UserPreference
import numpy as np

class UserUtility(UserPreference):

    def __init__(self, num_objectives, std_noise, seed=None):
        super().__init__(num_objectives, std_noise, seed)

    def inverted_stacked_sigmoids(self, x, a, b, n):
        y = 0
        y_min = 0
        y_max = 0
        for i in range(0, 5 * n, 5):
            y += 1. / (1 + np.exp(- (-x + 1) * (a - i) + (b + i)))  # flip horizontally between [0,1] with (-x + 1)
            y_min += 1. / (1 + np.exp(- 0 * (a - i) + (b + i)))
            y_max += 1. / (1 + np.exp(- 1 * (a - i) + (b + i)))
        return (y - y_min) / (y_max - y_min)

    def sigmoid_mon_utilities_1d(self, random_state, num_functions):
        sigmoids = []
        steepness_params = []
        shift_params = []
        stack_nr_params = []
        for _ in range(num_functions):
            a = random_state.uniform(10, 50, 1)[0]
            b = random_state.uniform(1, 20, 1)[0]
            n = random_state.randint(1, 10, 1)[0]
            sigmoids.append(lambda x, a=a, b=b, n=n: self.inverted_stacked_sigmoids(x, a, b, n))
            steepness_params.append(a)
            shift_params.append(b)
            stack_nr_params.append(n)
        return np.array(sigmoids), np.array(steepness_params), np.array(shift_params), np.array(stack_nr_params)

    def _initialise_utility_function(self):
        if self.num_objectives < 2:
            raise ValueError("Utility of the user should be for multiple objectives!")
        else:
            # for each objective, create a randomly monotonic stacked_sigmoid function
            self.funcs_1d, self.steepness_params, self.shift_params, self.stack_nr_params = self.sigmoid_mon_utilities_1d(self.random_state, num_functions=self.num_objectives)

            # weights for the individual preference functions
            # (if num_obj > 2 this reflects how much the user cares about the objectives)
            weight_preference_funcs = self.random_state.uniform(0.1, 0.9, len(self.funcs_1d))  # CHANGED (from 0.2-0.8)
            weight_preference_funcs /= np.sum(weight_preference_funcs)

            def utl_func(x):
                y = 0.
                for d in range(len(self.funcs_1d)):
                    y += weight_preference_funcs[d] * self.funcs_1d[d](x[:, d % self.num_objectives])

                return y

        return utl_func

    def find_most_promising_objective(self, x, perturbation=0.0001):
        best_objective = None
        direction = None  # this procedure can be extended with '+' direction
        max_utility_change = -np.inf
        initial_utility = self.get_preference(np.array([x]), add_noise=True)[0]
        for i in range(len(x)):
            x_perturbed_decrease = x.copy()
            # decrease the i'th objective
            x_perturbed_decrease[i] -= perturbation
            # calculate new utility for the perturbation
            if 0 <= x_perturbed_decrease[i] <= 1:  # the utility function defined for [0,1] domain!
                utility_perturbation_decrease = self.get_preference(np.array([x_perturbed_decrease]), add_noise=True)[0]
            else:
                utility_perturbation_decrease = -np.inf
            # calculate utility difference
            utility_change_perturbation_decrease = utility_perturbation_decrease - initial_utility
            if utility_change_perturbation_decrease > max_utility_change:
                max_utility_change = utility_change_perturbation_decrease
                best_objective = i
                direction = '-'
        return best_objective, direction
