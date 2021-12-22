from scse.controller import miniscot as miniSCOT

class miniSCOTnotebook():
    DEFAULT_START_DATE = '2019-01-01'
    DEFAULT_TIME_INCREMENT = 'half-hourly'
    DEFAULT_HORIZON = 100
    DEFAULT_SIMULATION_SEED = 12345
    DEFAULT_ASIN_SELECTION = 0
    DEFAULT_PROFILE = 'national_grid_profile'
    DEFAULT_NUM_BATTERIES = 10

    def __init__(self, time_horizon=DEFAULT_HORIZON, num_batteries=DEFAULT_NUM_BATTERIES):
        self.time_horizon = time_horizon
        self.num_batteries = num_batteries

        self.start(simulation_seed=self.DEFAULT_SIMULATION_SEED,
                   start_date=self.DEFAULT_START_DATE,
                   time_increment=self.DEFAULT_TIME_INCREMENT,
                   time_horizon=self.time_horizon,
                   asin_selection=self.DEFAULT_ASIN_SELECTION,
                   profile=self.DEFAULT_PROFILE,
                   num_batteries=self.num_batteries)

    def start(self, **run_parameters):
        self.horizon = run_parameters['time_horizon']
        self.actions = []
        self.breakpoints = []

        self.env = miniSCOT.SupplyChainEnvironment(**run_parameters)

        self.context, self.state = self.env.get_initial_env_values()
        self.env.reset_agents(self.context, self.state)

    def next(self):
        """Execute a single time unit."""
        self.state, self.actions, self.reward = self.env.step(self.state, self.actions)

    def run(self):
        """Run simulation until the first break-point or, if none are enabled, until the end of time (the specified horizon)."""
        for t in range(self.state['clock'], self.horizon):
            if t in self.breakpoints:
                break
            else:
                self.state, self.actions, self.reward = self.env.step(self.state, self.actions)


m = miniSCOTnotebook()

# Can use the following line to step through simulation
# m.next()

# Example below of injecting an action into the simulation.  Note this will raise an error since CHA1 doesn't have 5 units onhand!
# action={"schedule": 0, "type": "inbound_shipment", "asin": "9780465024759", "origin": "Manufacturer", "destination": "Newsvendor", "quantity": 5}
# actions = [action]
# m.env.step(m.state, actions)
