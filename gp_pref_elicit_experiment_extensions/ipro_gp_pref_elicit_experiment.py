from gp_pref_elicit.gp_utilities.utils_experiment import Experiment

import numpy as np
import time

class IproGpPrefElicitExperiment(Experiment):
    """
        This class makes an experiment to show the preference elicitation method of pairwise queries with the
        Ipro returned pareto_front as input domain.
    """

    def __init__(self, input_domain, user_util, parameters):

        self.pareto_front_input_domain = input_domain
        self.user = user_util
        super().__init__(parameters)

    def initialise_input_domain(self):
        return self.pareto_front_input_domain

    def initialise_user(self):
        return self.user

    def run(self, recalculate=True):

        # keep track of the maximum utility found
        max_utility_per_query = np.empty(self.params['num queries'])

        max_util_time = time.time()  # time when max_util solution has been found
        old_y_max = 0
        # loop: ask queries and update the gaussian process
        for q in range(self.params['num queries']):

            print("... query ", q + 1)

            # get the datapoint(s) for the next (first) query
            if q == 0:
                self.curr_x_max, self.curr_x_new = self.acquirer.get_start_points(self.gp)
                print("starting datapoints:", self.curr_x_max, self.curr_x_new)
                max_util_time = time.time()
            else:
                self.curr_x_new = self.acquirer.get_next_point(self.gp, self.dataset)
                print("next datapoint:", self.curr_x_new)

            if self.params['reference min'] == 'full':
                self.dataset.add_single_comparison(self.curr_x_new, np.zeros(self.params['num objectives']))
            elif self.params['reference min'] == 'beginning' and q < 5:
                self.dataset.add_single_comparison(self.curr_x_new, np.zeros(self.params['num objectives']))

            if self.params['reference max'] == 'full':
                self.dataset.add_single_comparison(np.ones(self.params['num objectives']), self.curr_x_new)
            elif self.params['reference max'] == 'beginning' and q < 5:
                self.dataset.add_single_comparison(np.ones(self.params['num objectives']), self.curr_x_new)

            new_x_max = self.make_query(x_max=self.curr_x_max, x_new=self.curr_x_new)

            if self.params['gp prior mean'] == 'linear-zero' and q > 4:
                self.gp.prior_mean_type = 'zero'

            print(self.dataset.comparisons)
            self.gp.update(self.dataset)

            y_max = self.user.get_preference(new_x_max, add_noise=False)[0]
            max_utility_per_query[q] = y_max

            # Update the time when the maximum utility is found
            if y_max > old_y_max:
                max_util_time = time.time()
                old_y_max = y_max

            self.curr_x_max = new_x_max

        results = self.gather_results(max_utility_per_query)
        results.append(max_util_time)
        print("results: ", results)

        return results
