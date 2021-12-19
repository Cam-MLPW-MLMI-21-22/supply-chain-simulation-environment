import networkx as nx

from scse.api.module import Env
from scse.constants.national_grid_constants import ELECTRICITY_ASIN, ENERGY_GENERATION_ASINS


class NationalGridNetwork(Env):
    # Start with nothing, and allow 1 period for transit
    # Technically transfer should be instantaneous, but lets cheat a bit
    _DEFAULT_INITIAL_INVENTORY = 0
    _DEFAULT_TRANSIT_TIME = 1

    _DEFAULT_MAX_BATTERY_CAPACITY = 50  #  size of a unit battery
    _DEFAULT_INIT_BATTERY_CAPACITY = int(_DEFAULT_MAX_BATTERY_CAPACITY * 0.2)

    _DEFAULT_NUM_BATTERIES = 1

    def __init__(self, run_parameters):
        """
        Highly simplified digital twin of the network.
        """
        self._initial_inventory = run_parameters.get(
            'initial_inventory', self._DEFAULT_INITIAL_INVENTORY)
        self._transit_time = run_parameters.get(
            'transit_time', self._DEFAULT_TRANSIT_TIME)
        self._max_battery_capacity = run_parameters.get(
            'max_battery_capacity', self._DEFAULT_MAX_BATTERY_CAPACITY)
        self._init_battery_capacity = run_parameters.get(
            'init_battery_capacity', self._DEFAULT_INIT_BATTERY_CAPACITY)

        self._num_batteries = run_parameters.get(
            'num_batteries', self._DEFAULT_NUM_BATTERIES)

    def get_name(self):
        return 'network'

    def get_initial_state(self, context):
        G = nx.DiGraph()
        asin_list = context['asin_list']

        ##############
        # Define Nodes
        ##############

        # "Vendors": electricity sources
        # These could be further broken down
        # Have only added a subset for now
        # Added `asins_produced` property
        G.add_node("Solar",
                   node_type='vendor',
                   asins_produced=[ENERGY_GENERATION_ASINS.solar],
                   location=(-3.7625904850106417, 50.485070023807836)
                   )
        G.add_node("Wind Onshore",
                   node_type='vendor',
                   asins_produced=[ENERGY_GENERATION_ASINS.wind_onshore],
                   location=(-4.369099752793398, 56.95015978364187)
                   )
        G.add_node("Fossil Gas",
                   node_type='vendor',
                   asins_produced=[ENERGY_GENERATION_ASINS.fossil_gas],
                   location=(-3.4726115079844275, 52.48838509810871)
                   )

        # "Port": electricity substations
        # Could also define as warehouses, but provides differentiation
        # Further analysis of how `node_type` is used required
        # Only one for now - more can be added at later date
        # Added `allow_negative` property
        G.add_node("Substation",
                   node_type='port',
                   location=(-1.47591978069484, 53.02151541873239),
                   inventory=dict.fromkeys(asin_list, self._initial_inventory),
                   allow_negative=True
                   )

        battery_loc = (-1.5549031279170884, 51.42927817167841)

        for battery_idx in range(self._num_batteries):
            # "Warehouse": batteries
            # note, assumes:
            # - batteries all in same loc
            # - batteries all have same max inventory (capacity)
            # - batteries all have same initial inventory
            G.add_node(f"Battery{battery_idx}",
                       node_type='warehouse',
                       location=battery_loc,
                       inventory={
                           ELECTRICITY_ASIN: self._init_battery_capacity},
                       max_inventory={
                           ELECTRICITY_ASIN: self._max_battery_capacity}
                       )
        # Consumers
        # Only one for now - more can be added at later date
        G.add_node("Consumers",
                   node_type='customer',
                   location=(-0.17563780900605935, 51.633920790187155),
                   delivered=0
                   )

        # Balance Mechanism
        # Source and sink for maintaining balance at substations.
        # Note that the source has been modelled as a vendor, and the sink
        # as a customer.
        G.add_node("Balance Source",
                   node_type='vendor',
                   asins_produced=[ELECTRICITY_ASIN],
                   location=(-0.6827303448677813, 54.107893307767526)
                   )
        G.add_node("Balance Sink",
                   node_type='customer',
                   location=(1.0740887603185447, 52.53358756872671),
                   delivered=0
                   )

        ##############
        # Define Edges
        ##############

        # Electricity sources to substation
        G.add_edge("Solar", "Substation",
                   ** {'transit_time': self._transit_time, 'shipments': []})
        G.add_edge("Wind Onshore", "Substation",
                   ** {'transit_time': self._transit_time, 'shipments': []})
        G.add_edge("Fossil Gas", "Substation",
                   ** {'transit_time': self._transit_time, 'shipments': []})

        for battery_idx in range(self._num_batteries):
            # Substation to battery
            # Note: At later date may want to model direct source -> battery storage
            G.add_edge("Substation", f"Battery{battery_idx}",
                       ** {'transit_time': self._transit_time, 'shipments': []})
            # Battery to to substation
            # Could use different substation to prevent cycle
            G.add_edge(f"Battery{battery_idx}", "Substation",
                       ** {'transit_time': self._transit_time, 'shipments': []})

        # Substation to customer
        G.add_edge("Substation", "Consumers",
                   ** {'transit_time': self._transit_time, 'shipments': []})

        # Balance source to substation
        G.add_edge("Balance Source", "Substation",
                   ** {'transit_time': self._transit_time, 'shipments': []})

        # Substation to balance sink
        G.add_edge("Substation", "Balance Sink",
                   ** {'transit_time': self._transit_time, 'shipments': []})

        return G
