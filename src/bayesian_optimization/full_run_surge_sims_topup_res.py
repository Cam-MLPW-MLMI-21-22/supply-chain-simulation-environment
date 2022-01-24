import itertools
from scse.default_run_parameters.national_grid_default_run_parameters import DEFAULT_RUN_PARAMETERS
from loop import *
from tqdm.auto import tqdm
from multiprocess.pool import Pool
from emukit.core.constraints import LinearInequalityConstraint
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.core.loop import ConvergenceStoppingCondition
from emukit.core.loop import FixedIterationsStoppingCondition
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.experimental_design import ExperimentalDesignLoop
from matplotlib.colors import LogNorm
from scse.api.simulation import run_simulation
from emukit.test_functions import branin_function
from skopt.benchmarks import branin as _branin
from GPy.models import GPRegression
from emukit.core.initial_designs import RandomDesign
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.model_wrappers import GPyModelWrapper
import logging
import time
from utils import plot_3d_boundary, plot_3d_observed_rewards
from pylab import *
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import GPy
import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

## Experimental Setup

all_scenarios = {"Baseline (20 % Discount)": {"discharge_discount": 0.8,
                                              "charging_discount": 1.05},
                 "25 % Discount,": {"discharge_discount": 0.75,
                                    "charging_discount": 1.05},
                 "33 % Discount": {"discharge_discount": 0.67,
                                   "charging_discount": 1.05},
                 "33 % Discount +": {"discharge_discount": 0.67,
                                     "charging_discount": 1.0},
                 "50 % Discount": {"discharge_discount": 0.5,
                                   "charging_discount": 1.05}}


# ["Baseline (20 % Discount)", "33 % Discount"]
network_param_scenarios = ["33 % Discount"]
surge_scenarios = ["wind+solar"]

# univariate_surge = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
univariate_surge = [0.1, 0.5, 0.7, 1.0, 1.2, 1.7, 2.0]

bivariate_surge = list(itertools.product([1.5, 2.5], [0.1, 0.3, 0.7]))

# surge vals stored as (wind_mod, solar_mod)

surge_vals = {"wind": [(surge_val, 1.0) for surge_val in univariate_surge],
              "solar": [(1.0, surge_val) for surge_val in univariate_surge],
              "wind+solar": bivariate_surge}

timesteps_per_day = 48


max_num_batteries = 1000

# units in £/
min_battery_penalty = DEFAULT_RUN_PARAMETERS.battery_penalty - 100000
max_battery_penalty = DEFAULT_RUN_PARAMETERS.battery_penalty + 200000

min_battery_capacity = 1
max_battery_capacity = 80

num_data_points = 30

# Basic plotting function


def plot_reward(X, Y, labels):
    """
    Plots reward against a maximum of two dimensions.
    """

    plt.style.use('seaborn')
    fig = plt.figure(figsize=(12, 12))

    order = np.argsort(X[:, 0])

    if X.shape[1] == 1:
        ax = plt.axes()
        ax.plot(X[order, 0], Y[order])
        ax.set_xlabel(labels[0])
        ax.set_ylabel("Cumulative reward")
    elif X.shape[1] == 2:
        ax = plt.axes(projection='3d')
        im = ax.plot_trisurf(X[order, 0].flatten(), X[order, 1].flatten(
        ), Y[order].flatten(), cmap=cm.get_cmap('autumn'))
        fig.colorbar(im)
        ax.view_init(90, 90)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel("Cumulative reward")  # (£/MWh)
    else:
        raise ValueError('X has too many dimensions to plot - max 2 allowed')

    return fig, ax


# ## Run Experiments

# In[10]:

for num_days in [10]:

    time_horizon_value = timesteps_per_day*num_days

    main_save_dir = f"./surge_results_{num_days}"
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    for network_param_scenario in network_param_scenarios:
        for scenario in surge_scenarios:
            # for scenario in surge_scenarios:
            #     for network_param_scenario in network_param_scenarios:
            save_dir = f"{main_save_dir}/{scenario}/{network_param_scenario}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stored_opt_locs = []
            stored_wind_mod = []
            stored_wind_mod = []
            stored_scenarios = []
            stored_solar_mod = []
            stored_param_scenarios = []

            for (wind_mod, solar_mod) in surge_vals[scenario]:

                single_res_file = open(
                    f"{save_dir}/results_{wind_mod}_{solar_mod}.txt", "w")

                print(
                    f"Network Param Scenario: {network_param_scenario}, Surge Scenario: {scenario}\n")
                print(f"Wind: {wind_mod}, Solar: {solar_mod}\n")

                num_batteries = DiscreteParameter(
                    'num_batteries', range(0, max_num_batteries+1))

                # parameters = [num_batteries, network_param_scenario,
                #               scenario, wind_mod, solar_mod]
                parameters = [num_batteries]
                parameter_space = ParameterSpace(parameters)
                # Get and check the parameters of the intial values (X)

                design = RandomDesign(parameter_space)
                X = design.get_samples(num_data_points)
                # overwrite
                X = np.linspace(0, max_num_batteries,
                                num_data_points).reshape([num_data_points, 1])
                print(X)

                def invoke_miniscot(x):
                    """
                    Handling single API call to miniSCOT simulation given some inputs

                    x contains parameter configs x = [x0 x1 ...]
                    - The order of parameters in x should follow the order specified in the parameter_space declaration
                    - E.g. here we specify num_batteries = x[0]
                    """
                    kwargs = {
                        'time_horizon': time_horizon_value,
                        'num_batteries': int(x[0])
                    }

                    kwargs.update(all_scenarios[network_param_scenario])

                    print("Running: ", scenario, wind_mod, solar_mod)

                    kwargs["surge_modulator"] = wind_mod
                    kwargs["solar_surge_modulator"] = solar_mod
                    kwargs["surge_scenario"] = scenario

                    cum_reward = run_simulation(**kwargs)

                    return cum_reward[-1]

                # In[7]:

                def f(X):
                    """
                    Handling multiple API calls to miniSCOT simulation given some inputs

                    X is a matrix of parameters
                    - Each row is a set of parameters
                    - The order of parameters in the row should follow the order specified in the parameter_space declaration
                    """
                    Y = []
                    for x in X:
                        cum_reward = invoke_miniscot(x)

                        # Note that we negate the reward; want to find min
                        Y.append(-cum_reward[-1])

                    Y = np.reshape(np.array(Y), (-1, 1))
                    return Y

                # In[8]:

                def f_multiprocess(X):
                    """
                    Handling multiple API calls to miniSCOT simulation given some inputs using multiprocessing.

                    X is a matrix of parameters
                    - Each row is a set of parameters
                    - The order of parameters in the row should follow the order specified in the parameter_space declaration
                    """

                    # Set to None to use all available CPU
                    max_pool = 5
                    with Pool(max_pool) as p:
                        Y = list(
                            tqdm(
                                p.imap(invoke_miniscot, X),
                                total=X.shape[0]
                            )
                        )

                    # Note that we negate the reward; want to find min
                    Y = -np.reshape(np.array(Y), (-1, 1))
                    return Y

                # ### Get initial data points

                design = RandomDesign(parameter_space)

                start = time.time()
                Y = f_multiprocess(X)
                end = time.time()
                print("Getting {} initial simulation points took {} seconds".format(
                    num_data_points, round(end - start, 0)))

                # ### Check the cum. reward from the initial points to ensure we are seeing reasonable behaviour before fitting

                # In[14]:

                plot_reward(X, Y, parameter_space.parameter_names)

                successful_sample = False
                num_tries = 0
                max_num_tries = 3

                use_default = False
                use_ard = False

                while not successful_sample and num_tries < max_num_tries:

                    print(f"\nCURRENT ATTEMPT #{num_tries}")

                    # emulator model

                    if use_default:
                        gpy_model = GPRegression(X, Y)
                    else:
                        kernel = GPy.kern.RBF(1, lengthscale=1e0,
                                              variance=1e4, ARD=use_ard)
                        gpy_model = GPy.models.GPRegression(
                            X, Y, kernel, noise_var=1e-10)

                    try:
                        gpy_model.optimize()
                        print("\nokay to optimize")
                        model_emukit = GPyModelWrapper(gpy_model)

                        # Load core elements for Bayesian optimization
                        expected_improvement = ExpectedImprovement(
                            model=model_emukit)
                        optimizer = GradientAcquisitionOptimizer(
                            space=parameter_space)

                        # Create the Bayesian optimization object
                        batch_size = 3
                        bayesopt_loop = BayesianOptimizationLoop(model=model_emukit,
                                                                 space=parameter_space,
                                                                 acquisition=expected_improvement,
                                                                 batch_size=batch_size)

                        # Run the loop and extract the optimum;  we either complete 10 steps or converge
                        max_iters = 5
                        stopping_condition = (
                            FixedIterationsStoppingCondition(
                                i_max=max_iters) | ConvergenceStoppingCondition(eps=0.01)
                        )

                        bayesopt_loop.run_loop(
                            f_multiprocess, stopping_condition)
                        print("\nsuccessfully ran loop\n")
                        successful_sample = True

                    except:
                        num_tries += 1

        #                ### Get new points from BO

                new_X, new_Y = bayesopt_loop.loop_state.X, bayesopt_loop.loop_state.Y

                new_order = np.argsort(new_X[:, 0])
                new_X = new_X[new_order, :]
                new_Y = new_Y[new_order]

                # ### Plot results from BO with the GP fit

                x_plot = np.reshape(
                    np.array([i for i in range(0, max_num_batteries)]), (-1, 1))
                mu_plot, var_plot = model_emukit.predict(x_plot)

                plt.figure(figsize=(12, 8))
                LEGEND_SIZE = 15
                plt.plot(new_X, new_Y, "ro", markersize=10,
                         label="All observations")
                plt.plot(X, Y, "bo", markersize=10,
                         label="Initial observations")
                plt.plot(x_plot, mu_plot, "C0", label="Model")
                plt.fill_between(x_plot[:, 0],
                                 mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                                 mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)
                plt.fill_between(x_plot[:, 0],
                                 mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                                 mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
                plt.fill_between(x_plot[:, 0],
                                 mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                                 mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)

                plt.legend(prop={'size': 14})
                plt.xlabel("Number of Batteries", fontsize=14)
                plt.ylabel("Cumulative Reward (£)", fontsize=14)
                plt.grid(True)
                plt.yticks(fontsize=13)
                plt.xticks(fontsize=13)
                plt.savefig(
                    f"{save_dir}/wind_{wind_mod}_solar_{solar_mod}.png")

                results = bayesopt_loop.get_results()
                print(
                    f"\nBest: {results.minimum_location}, Val: {results.minimum_value}\n")
                opt_loc = results.minimum_location[0]
                single_res_file.write(str(opt_loc))
                single_res_file.close()
                stored_opt_locs.append(opt_loc)
                stored_scenarios.append(scenario)
                stored_param_scenarios.append(network_param_scenario)
                stored_wind_mod.append(wind_mod)
                stored_solar_mod.append(solar_mod)

            res_df = pd.DataFrame({"Optimum": stored_opt_locs, "Solar Surge Factor": stored_solar_mod,
                                   "Wind Surge Factor": stored_wind_mod, "Network Parameter Scenario": stored_param_scenarios,
                                   "Surge Scenario": stored_scenarios})
            res_df.to_csv(f"{save_dir}/results.csv")
