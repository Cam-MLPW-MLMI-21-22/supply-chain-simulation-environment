import logging

from scse.api.module import Agent
from scse.services.service_registry import singleton as registry
from scse.constants.national_grid_constants import (
    ELECTRICITY_ASIN, DEFAULT_CONSUMER
)

logger = logging.getLogger(__name__)


class ElectricityDemand(Agent):
    _DEFAULT_ASIN = ELECTRICITY_ASIN
    _DEFAULT_CUSTOMER = DEFAULT_CONSUMER

    def __init__(self, run_parameters):
        """
        Simulates electricity demand from a single consumer/customer.

        Demand forecast is provided by a service.

        NOTE: The balance mechanism sink is currently modelled as a
        customer. Therefore, we cannot simply loop through all
        customers if we were to add more to the network. However, it
        could be the responsibility of the demand service to identify
        the sink and ensure it always has 0 demand.
        """
        self._demand_forecast_service = registry.load_service('electricity_demand_forecast_service', run_parameters)
        self._asin = run_parameters.get('constant_demand_asin', self._DEFAULT_ASIN)
        self._customer = run_parameters.get('constant_demand_customer', self._DEFAULT_CUSTOMER)

    def get_name(self):
        return 'constant_demand'

    def reset(self, context, state):
        self._asin_list = context['asin_list']

    def compute_actions(self, state):
        """
        Creates a single action/order for an amount of electricity.
        """
        actions = []
        current_time = state['date_time']
        current_clock = state['clock']

        forecasted_demand = self._demand_forecast_service.get_forecast(
            clock=current_clock, time=current_time
        )

        action = {
            'type': 'customer_order',
            'asin': self._asin,
            'origin': None,  # The customer cannot request where the electricity comes from
            'destination': self._customer,
            'quantity': forecasted_demand,
            'schedule': state['clock']
        }
            
        logger.debug(f"{self._customer} requested {forecasted_demand} units of {self._asin}.")

        actions.append(action)

        return actions
