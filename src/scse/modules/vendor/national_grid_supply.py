import logging

from scse.api.module import Agent
from scse.services.service_registry import singleton as registry
from scse.constants.national_grid_constants import DEFAULT_BALANCE_SOURCE, ELECTRICITY_ASIN, ENERGY_GENERATION_ASINS
from scse.default_run_parameters.national_grid_default_run_parameters import (
    DEFAULT_RUN_PARAMETERS
)

logger = logging.getLogger(__name__)


class ElectricitySupply(Agent):
    _DEFAULT_SUPPLY_ASIN = ELECTRICITY_ASIN

    def __init__(self, run_parameters):
        """
        Simulates electricity supply from all sources.

        Supply forecast is provided by a service.

        NOTE: The balance mechanism source is currently modelled as a
        vendor. Therefore, we cannot simply loop through all
        vendors - we must exclude the balance source. A simple method
        of doing this is implemented below, but we may want to find a
        more robust solution.
        """
        self._supply_asin = self._DEFAULT_SUPPLY_ASIN
        self._supply_forecast_service = registry.load_service(
            'electricity_supply_forecast_service', run_parameters)
        self._surge_modulator = run_parameters.get(
                'surge_modulator', DEFAULT_RUN_PARAMETERS.surge_modulator)
        self._solar_surge_modulator = run_parameters.get(
                'solar_surge_modulator', DEFAULT_RUN_PARAMETERS.solar_surge_modulator)
        self._surge_scenario = run_parameters.get(
                'surge_scenario', DEFAULT_RUN_PARAMETERS.surge_scenario)

    def get_name(self):
        return 'vendor'

    def reset(self, context, state):
        self._asin_list = context['asin_list']

    def compute_actions(self, state):
        """
        Simulate a supply of electricity from each source.

        NOTE: Only supports having a single substation currently.
        """
        actions = []
        current_time = state['date_time']
        current_clock = state['clock']

        G = state['network']

        # Get a list of substations - remember that these have the type `port` for now
        substations = []
        for node, node_data in G.nodes(data=True):
            if node_data.get('node_type') in ['port']:
                substations.append(node)

        if len(substations) == 0:
            raise ValueError('Could not identify any substations.')
        elif len(substations) > 1:
            raise ValueError(
                'Identified multiple substations - this is not yet supported.')

        substation = substations[0]

        # Create shipment from every vendor (i.e. electricity supply) to substation
        # NOTE: Do not create demand from balance source. TODO: Find a better method of avoiding.
        for node, node_data in G.nodes(data=True):
            if node_data.get('node_type') == 'vendor' and node != DEFAULT_BALANCE_SOURCE:
                # Determine how the vendor produces electricity
                generation_types = node_data.get('asins_produced')
                if generation_types is None:
                    raise ValueError(
                        'Expect all sources to have `asins_produced` property, indicating their generation type.')
                elif len(generation_types) != 1:
                    raise ValueError(
                        'All sources must have only one generation type - multiple not yet supported.')

                generation_type = generation_types[0]
                forecasted_supply = self._supply_forecast_service.get_forecast(
                    asin=generation_type, clock=current_clock, time=current_time
                )

                wind_surge_timesteps = DEFAULT_RUN_PARAMETERS.timesteps_per_day * \
                    DEFAULT_RUN_PARAMETERS.days_wind_surge
                solar_surge_timesteps = DEFAULT_RUN_PARAMETERS.timesteps_per_day * \
                    DEFAULT_RUN_PARAMETERS.days_solar_surge
                # model variety of supply surge scenarios
                if self._surge_scenario == "wind":
                    if generation_type == ENERGY_GENERATION_ASINS.wind_offshore:
                        # run wind surge for pre-specified number of days
                        #  number of timesteps per day (hard-coded)
                        if current_clock < wind_surge_timesteps:
                            forecasted_supply *= self._surge_modulator
                            forecasted_supply = int(forecasted_supply)
                elif self._surge_scenario == "solar":
                    if generation_type == ENERGY_GENERATION_ASINS.solar:
                        # run solar surge for one week
                        if current_clock < solar_surge_timesteps:
                            forecasted_supply *= self._solar_surge_modulator
                            forecasted_supply = int(forecasted_supply)
                elif self._surge_scenario == "wind+solar":
                    # wind for one week, solar for next
                    if generation_type == ENERGY_GENERATION_ASINS.wind_offshore:
                        if current_clock < wind_surge_timesteps:
                            forecasted_supply *= self._surge_modulator
                            forecasted_supply = int(forecasted_supply)
                    elif generation_type == ENERGY_GENERATION_ASINS.solar:
                        if current_clock >= wind_surge_timesteps and current_clock < wind_surge_timesteps + solar_surge_timesteps:
                            forecasted_supply *= self._solar_surge_modulator
                            forecasted_supply = int(forecasted_supply)

                logger.debug(
                    f"Supply for {forecasted_supply} quantity of ASIN {generation_type}.")

                # Note: The default ASIN is sent (i.e. electricity), regardless of generation type
                action = {
                    'type': 'inbound_shipment',
                    'asin': self._supply_asin,
                    'origin': node,
                    'destination': substation,
                    'quantity': forecasted_supply,
                    'schedule': current_clock
                }

                actions.append(action)

        return actions
