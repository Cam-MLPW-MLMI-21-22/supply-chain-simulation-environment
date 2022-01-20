from dataclasses import dataclass

from scse.default_run_parameters.core_default_run_parameters import CoreRunParameters


@dataclass
class _RunParameters(CoreRunParameters):
    # Override certain core parameters
    run_profile = 'national_grid_profile'
    asin_selection = 0  # Use 0 for national grid simulation
    logging_level = 'DEBUG'  #  May need to set to critical occassionally
    time_increment = 'half-hourly'
    time_horizon = 336  #  1 week, if 30 min time increments

    # Bespoke parameters
    simulation_logging_level = 'CRITICAL'
    num_batteries = 1  # 5

    discharge_discount = 0.8
    charging_discount = 1.05

    # Penalty and reward prices, w/ units £/MWh
    source_request_reward_penalty = -36.65
    sink_deposit_reward_penalty = 27.63
    battery_drawdown_reward_penalty = -36.65  # * discharge_discount
    battery_charging_reward_penalty = 27.63  # * charging_discount

    # Other penalties
    transfer_penalty = 0  # 2
    lost_demand_penalty = 0
    holding_cost_penalty = 0  # -20

    # for now, assumes all batteries are of same capacity
    # TODO: modify to handle capacity which scales with cost
    max_battery_capacity = 150  # units in MWh; current sites typically 50 MWh
    # fraction of charge in the batteries at the beginning
    init_battery_charge_frac = 0.5
    battery_penalty = -(200 * 1000)  # units in £/MWh
    lifetime_years = 15  # number of years over which price is spread

    surge_modulator = 1.0  #  baseline (no surge = 1.0)
    solar_surge_modulator = 1.0
    surge_scenario = "wind"  #  options = {"wind", "solar", "wind+solar"}
    timesteps_per_week = 336


DEFAULT_RUN_PARAMETERS = _RunParameters()
