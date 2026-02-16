import time

from gp_pref_elicit_experiment_extensions.ipro_gp_pref_elicit_experiment import IproGpPrefElicitExperiment
from gp_pref_elicit.gp_utilities.utils_parameters import get_parameter_dict

from gp_pref_elicit_experiment_extensions.user_utility import UserUtility
from ipro.outer_loops.ipro import IPRO
import numpy as np


class InteractiveIproVsGPE:
    """
        Interactive (Preference-Guided) IPRO with manual user interaction Versus Gaussian_process Preference Elicitation
        Experiment.
        (running the experiment of the two methods together, because the same utility function is used to compare them)
    """

    def __init__(self, num_queries, experiment_iterations, dimensions, ipro_oracle, ipro_linear_solver,
                 gppe_query_type='pairwise', utility_noise=0.01, direction='minimize',
                 referent_selection_heuristic='closest_distance', tolerance=0):
        self.num_queries = num_queries
        self.experiment_iterations = experiment_iterations
        self.dimensions = dimensions
        self.ipro_oracle = ipro_oracle
        self.ipro_linear_solver = ipro_linear_solver
        self.gppe_query_type = gppe_query_type  # Gaussian Process Preference Elicitation query type
        self.utility_noise = utility_noise
        self.direction = direction  # maximization or minimization problem
        self.referent_selection_heuristic = referent_selection_heuristic
        self.tolerance = tolerance
        self.sign = 1 if direction == 'maximize' else -1
        if self.direction not in ['maximize', 'minimize']:
            raise ValueError("Unknown direction")

    def run(self):
        '''
            IPRO, without manual user input interaction, to find the input domain for the Gaussian Process preference
            elicitation experiment.
            Bayesian approach to find the most preferred solution by the user (after a certain amount of queries)
        '''
        # Find Pareto Front with IPRO (this execution time is used for the runtime result in the gp_pref_elicit part of the experiment)
        start_ipro_generate_pf_time = time.time()
        ipro = IPRO(
            problem_id='gp_pref_elicit_finding_pf',
            dimensions=self.dimensions,
            oracle=self.ipro_oracle,
            linear_solver=self.ipro_linear_solver,
            direction=self.direction,
            max_iterations=1000,
            tolerance=self.tolerance,
            user_interaction_loop=False,
        )
        ipro_pareto_front = ipro.solve()
        ipro_generate_pf_time = time.time() - start_ipro_generate_pf_time  # in seconds
        print("pf:", ipro_pareto_front)
        print("Finding the pareto front with IPRO runtime: ", ipro_generate_pf_time)

        input_dom = np.vstack([item[0]*self.sign for item in ipro_pareto_front])
        # normalize ipro_pareto_front (input_dom) with objective values between [0,1]
        min_vals = np.min(input_dom, axis=0)
        max_vals = np.max(input_dom, axis=0)
        normalized_data = (input_dom - min_vals) / (max_vals - min_vals)  # requirement for gp_pref_elicit and the used utility function
        # denormalized_data = normalized_data * (max_vals - min_vals) + min_vals

        all_utility_gppe = []
        max_utility_gppe = []
        all_utility_interactive_ipro = []
        max_utility_interactive_ipro = []
        interactive_ipro_generate_subsolution_times = []
        user = 1
        for user_utility_seed in range(self.experiment_iterations):  # Simulate different users
            print("user iteration: ", user)
            user += 1

            '''
                USER PREFERENCES as utility function
            '''
            util_noise_experiment = self.utility_noise  # set add_noise in get_preference calls to True
            user_utility = UserUtility(num_objectives=self.dimensions, std_noise=util_noise_experiment, seed=user_utility_seed)
            user_utility.rescale_on_input_domain(normalized_data)

            '''
                Gaussian Process preference elicitation experiment
            '''
            params = get_parameter_dict(query_type=self.gppe_query_type, num_objectives=self.dimensions,
                                        utility_noise=util_noise_experiment)
            params['num queries'] = self.num_queries
            params['seed'] = user_utility_seed

            experiment = IproGpPrefElicitExperiment(input_domain=normalized_data, user_util=user_utility,
                                                    parameters=params)
            results = experiment.run(recalculate=True)
            utility_datapoints = results[6]  # in gppe the first query gives 2 datapoints and afterwards always one new datapoint after a query
            max_utility_per_query = results[0]

            all_utility_gppe.append(utility_datapoints)
            first_sol = min(utility_datapoints[0], utility_datapoints[1])
            max_utility_per_solution = np.concatenate(([first_sol], max_utility_per_query))
            max_utility_gppe.append(max_utility_per_solution)

            '''
                Preference-Guided IPRO with preference guidance/user input interaction experiment
            '''

            preference_guided_ipro = IPRO(
                problem_id="interactive_ipro",
                dimensions=self.dimensions,
                oracle=self.ipro_oracle,
                linear_solver=self.ipro_linear_solver,
                direction='minimize',
                max_iterations=1000,
                tolerance=self.tolerance,
                user_interaction_loop=True,  # IPRO with user input interaction
                preferred_objective=None,
                referent_selection_heuristic=self.referent_selection_heuristic
            )

            utility_subsolutions = []
            max_utility_subsolutions = []
            user_input = None
            query = 0
            curr_x_max = None
            processed_solutions = []
            start_interactive_ipro_generate_subsolution_time = time.time()
            while True:
                print("Enter preference guided IPRO solve")
                subsolution = preference_guided_ipro.solve(user_input=user_input)  # subsolution of type list[tuple[np.ndarray, Any]],
                #  but only returns a list with 1 element from solve for IPRO with user input interaction
                if subsolution is None:
                    print("User preferences can't be improved, no better solution exists.")
                    # Extend the utility_subsolutions with the last utility found since no new query will change this:
                    utility_subsolutions = utility_subsolutions + [utility_subsolutions[-1]] * ((self.num_queries + 1) - len(utility_subsolutions))
                    max_utility_subsolutions = max_utility_subsolutions + [max_utility_subsolutions[-1]] * ((self.num_queries + 1) - len(max_utility_subsolutions))
                    break
                elif len(subsolution) != 1:
                    # at end of IPRO solve method it returns the whole pareto front,
                    # this marks the end of the iteration/interaction loop since all possibilities have been processed
                    print("pf:", subsolution)
                    print("Done!")
                    # Extend the utility_subsolutions with the last utility found since no new query will change this:
                    utility_subsolutions = utility_subsolutions + [utility_subsolutions[-1]] * (
                                (self.num_queries + 1) - len(utility_subsolutions))
                    max_utility_subsolutions = max_utility_subsolutions + [max_utility_subsolutions[-1]] * (
                                (self.num_queries + 1) - len(max_utility_subsolutions))
                    break
                else:
                    print("Preference elicitation loop not yet done, IPRO still not done!")
                    pareto_sol_objectives = subsolution[0][0] * self.sign
                    pareto_sol = subsolution[0][1]
                    if pareto_sol is None:  # sol returned by oracle should be None if no pareto_sol can be found
                        print("Current referent can't find a subsolution, try again for a next_referent...")
                        continue
                    else:
                        # Time for PG-IPRO to find a solution (after possibly trying multiple referents),
                        # this is the time to propose an actual new solution to the user.
                        end_interactive_ipro_generate_subsolution_time = time.time()
                        # maintain average runtime for interactive (PG-IPRO) to find one subsolution:
                        # if query >= 1:  # ignore init_phase setup IPRO
                        interactive_ipro_generate_subsolution_time = end_interactive_ipro_generate_subsolution_time - start_interactive_ipro_generate_subsolution_time
                        interactive_ipro_generate_subsolution_times.append(interactive_ipro_generate_subsolution_time)

                # Save user utility for the given subsolution
                normalized_pareto_sol_objectives = (pareto_sol_objectives - min_vals) / (max_vals - min_vals)
                current_utility = user_utility.get_preference(np.array(normalized_pareto_sol_objectives), add_noise=False)[0]
                # Instead of computing the utility, get it from the actual user by asking him what he prefers

                utility_subsolutions.append(current_utility)

                # only start querying user after first solution has already been found
                if curr_x_max is not None:
                    comp_result = user_utility.pairwise_comparison(curr_x_max, normalized_pareto_sol_objectives, add_noise=True)
                    if comp_result:
                        print("no update required of x_max")
                    else:
                        curr_x_max = normalized_pareto_sol_objectives

                    y_max = user_utility.get_preference(curr_x_max, add_noise=False)[0]
                    max_utility_subsolutions.append(y_max)
                else:  # first time PG-IPRO shows the first solution this is the max reached before query 1
                    max_utility_subsolutions.append(current_utility)
                    curr_x_max = normalized_pareto_sol_objectives

                # End after a pre-defined number of queries:
                if query >= self.num_queries:
                    break

                # Query user utility for next objective binary preference
                # The user defines what is the most promising objective to improve
                # according to him
                obj, direction = user_utility.find_most_promising_objective(
                    normalized_pareto_sol_objectives)  # utility model works with domain [0,1]
                query += 1
                user_input = (obj, direction)
                # reset time:
                start_interactive_ipro_generate_subsolution_time = time.time()

            all_utility_interactive_ipro.append(np.array(utility_subsolutions))
            max_utility_interactive_ipro.append(np.array(max_utility_subsolutions))

        ''' Average mean and standard deviation calculations '''
        all_utility_gppe_data = np.vstack(all_utility_gppe)
        max_utility_gppe_data = np.vstack(max_utility_gppe)
        all_utility_interactive_ipro_data = np.vstack(all_utility_interactive_ipro)
        max_utility_interactive_ipro_data = np.vstack(max_utility_interactive_ipro)

        average_utility_gppe = np.mean(all_utility_gppe_data, axis=0)
        average_utility_interactive_ipro = np.mean(all_utility_interactive_ipro_data, axis=0)
        average_max_utility_gppe = np.mean(max_utility_gppe_data, axis=0)
        average_max_utility_interactive_ipro = np.mean(max_utility_interactive_ipro_data, axis=0)

        std_dev_utility_gppe = np.std(all_utility_gppe_data, axis=0)
        std_dev_utility_interactive_ipro = np.std(all_utility_interactive_ipro_data, axis=0)
        std_dev_max_utility_gppe = np.std(max_utility_gppe_data, axis=0)
        std_dev_max_utility_interactive_ipro = np.std(max_utility_interactive_ipro_data, axis=0)

        # times
        mean_interactive_ipro_generate_subsolution_time = np.mean(interactive_ipro_generate_subsolution_times)
        std_dev_interactive_ipro_generate_subsolution_time = np.std(interactive_ipro_generate_subsolution_times)

        #print("PG-IPRO times: ", interactive_ipro_generate_subsolution_times)

        return average_utility_gppe, average_utility_interactive_ipro, average_max_utility_gppe, \
            average_max_utility_interactive_ipro, std_dev_utility_gppe, std_dev_utility_interactive_ipro, \
            std_dev_max_utility_gppe, std_dev_max_utility_interactive_ipro, ipro_generate_pf_time, \
            mean_interactive_ipro_generate_subsolution_time, std_dev_interactive_ipro_generate_subsolution_time
