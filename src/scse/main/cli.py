from __future__ import print_function
import sys
import importlib
import argparse
import cmd2
import pprint
import networkx as nx
from itertools import cycle
from collections import defaultdict
import matplotlib.pyplot as plt

from scse.controller import miniscot as miniSCOT
from scse.default_run_parameters.national_grid_default_run_parameters import DEFAULT_RUN_PARAMETERS


class MiniSCOTDebuggerApp(cmd2.Cmd):

    def __init__(self, **args):
        super().__init__(args)
        print("Welcome to miniSCOT - your Supply Chain in a bottle.")
        print("IMPORTANT: If you want to supply arguments you need to run 'start' along with the desired arguments"
              "to initialize the environment correctly. Then run 'run'.")
        # self.intro = self.colorize("Welcome to miniSCOT - your Supply Chain in a bottle.", 'cyan')

        self._start(simulation_seed=DEFAULT_RUN_PARAMETERS.simulation_seed,
                    start_date=DEFAULT_RUN_PARAMETERS.start_date,
                    time_increment=DEFAULT_RUN_PARAMETERS.time_increment,
                    time_horizon=DEFAULT_RUN_PARAMETERS.time_horizon,
                    asin_selection=DEFAULT_RUN_PARAMETERS.asin_selection,
                    profile=DEFAULT_RUN_PARAMETERS.run_profile,
                    num_batteries=DEFAULT_RUN_PARAMETERS.num_batteries,
                    max_battery_capacity=DEFAULT_RUN_PARAMETERS.max_battery_capacity,
                    battery_penalty=DEFAULT_RUN_PARAMETERS.battery_penalty,
                    discharge_discount=DEFAULT_RUN_PARAMETERS.discharge_discount,
                    charging_discount=DEFAULT_RUN_PARAMETERS.charging_discount,
                    surge_modulator=DEFAULT_RUN_PARAMETERS.surge_modulator,
                    solar_surge_modulator=DEFAULT_RUN_PARAMETERS.solar_surge_modulator,
                    surge_scenario=DEFAULT_RUN_PARAMETERS.surge_scenario)

        self._set_prompt()

    def _set_prompt(self):
        print("miniSCOT (t = {!r}) $ ".format(self._state['clock']))
        # self.prompt = self.colorize("miniSCOT (t = {!r}) $ ".format(self._state['clock']), 'cyan')

    def postcmd(self, stop, line):
        self._set_prompt()
        return stop

    def _start(self, **run_parameters):

        self._horizon = run_parameters['time_horizon']
        self._actions = []
        self._breakpoints = []

        self._env = miniSCOT.SupplyChainEnvironment(**run_parameters)

        self._context, self._state = self._env.get_initial_env_values()
        self._env.reset_agents(self._context, self._state)

    param_parser = argparse.ArgumentParser()
    param_parser.add_argument(
        '--start_date', help="simulation will at date 'yyyy-mm-dd' (default 2019-01-01)", type=str, default=DEFAULT_RUN_PARAMETERS.start_date)
    param_parser.add_argument(
        '--time_increment', help="increment time daily or hourly (default 'daily')", type=str, default=DEFAULT_RUN_PARAMETERS.time_increment)
    param_parser.add_argument(
        '--horizon', help="total time units to simulate (default 100)", type=int, default=DEFAULT_RUN_PARAMETERS.time_horizon)
    param_parser.add_argument(
        '--seed', help="simulation random seed (default 12345)", type=int, default=DEFAULT_RUN_PARAMETERS.simulation_seed)
    param_parser.add_argument(
        '--asin_selection', help="number of ASINs to use (default 10)", type=int, default=DEFAULT_RUN_PARAMETERS.asin_selection)
    param_parser.add_argument(
        '--profile', help="profile (default minimal)", type=str, default=DEFAULT_RUN_PARAMETERS.run_profile)
    param_parser.add_argument(
        '--num_batteries',
        help=f"number of batteries (default {DEFAULT_RUN_PARAMETERS.num_batteries})",
        type=int, default=DEFAULT_RUN_PARAMETERS.num_batteries)
    param_parser.add_argument(
        '--max_battery_capacity',
        help=f"max battery capacity (default {DEFAULT_RUN_PARAMETERS.max_battery_capacity})",
        type=int, default=DEFAULT_RUN_PARAMETERS.max_battery_capacity)
    param_parser.add_argument(
        '--battery_penalty',
        help=(
            f"penalty for the addition of a new battery (default {DEFAULT_RUN_PARAMETERS.battery_penalty})"
        ),
        type=int, default=DEFAULT_RUN_PARAMETERS.battery_penalty)
    param_parser.add_argument(
        '--surge_modulator',
        help=(
            f"scalar factor on offshore wind supply (default {DEFAULT_RUN_PARAMETERS.surge_modulator})"
        ),
        type=float, default=DEFAULT_RUN_PARAMETERS.surge_modulator)
    param_parser.add_argument(
        '--solar_surge_modulator',
        help=(
            f"scalar factor on solar supply  (default {DEFAULT_RUN_PARAMETERS.solar_surge_modulator})"
        ),
        type=float, default=DEFAULT_RUN_PARAMETERS.solar_surge_modulator)
    param_parser.add_argument(
        '--surge_scenario',
        help=(
            f"which scenario to in options wind, solar, wind+solar (default {DEFAULT_RUN_PARAMETERS.surge_scenario})"
        ),
        type=str, default=DEFAULT_RUN_PARAMETERS.surge_scenario)
    param_parser.add_argument(
        '--charging_discount',
        help=(
            f"discount on the charging penalty for a battery (default {DEFAULT_RUN_PARAMETERS.charging_discount})"
        ),
        type=float, default=DEFAULT_RUN_PARAMETERS.charging_discount)
    param_parser.add_argument(
        '--discharge_discount',
        help=(
            f"discount on the discharging reward for a battery (default {DEFAULT_RUN_PARAMETERS.discharge_discount})"
        ),
        type=float, default=DEFAULT_RUN_PARAMETERS.discharge_discount)
    #param_parser.add_argument('-asin', help="list of ASINs.", action='append', default=_DEFAULT_ASIN_LIST)

    @cmd2.with_argparser(param_parser)
    def do_start(self, args):
        """Start (or restart) environment, resetting all variables and state."""
        self._start(simulation_seed=args.seed,
                    start_date=args.start_date,
                    time_increment=args.time_increment,
                    time_horizon=args.horizon,
                    asin_selection=args.asin_selection,
                    profile=args.profile,
                    num_batteries=args.num_batteries,
                    max_battery_capacity=args.max_battery_capacity,
                    surge_modulator=args.surge_modulator,
                    surge_scenario=args.surge_scenario,
                    solar_surge_modulator=args.solar_surge_modulator,
                    charging_discount=args.charging_discount,
                    discharge_discount=args.discharge_discount)

    def do_next(self, arguments):
        """Execute a single time unit."""
        self._state, self._actions, self._reward = self._env.step(
            self._state, self._actions)

    def do_run(self, arguments, visual=False):
        """
        Run simulation until the first break-point or, if none are enabled, until the end of time (the specified horizon).

        If `visual=True` then call `do_visualise` on each step. Press q to iterate through steps.
        """

        if visual:
            self.do_visualise(None)
        for t in range(self._state['clock'], self._horizon):
            if t in self._breakpoints:
                break
            else:
                self._state, self._actions, self._reward = self._env.step(
                    self._state, self._actions)
                if visual:
                    self.do_visualise(None)

    def do_visual_run(self, arguments):
        """
        Executes `do_run` with `visual=True` argument.
        """
        self.do_run(arguments, visual=True)

    def do_print(self, args):
        """Print variables."""
        msg = ""
        if args == 'nodes':
            msg = self._print_nodes()
        elif args == 'edges':
            msg = self._print_edges()
        elif args == 'actions':
            msg = pprint.pformat(self._actions)
        elif args == 'orders':
            msg = pprint.pformat(self._state['customer_orders'])
        elif args == 'order_hist':
            msg = pprint.pformat(self._state['customer_order_history'])
        elif args == "POs":
            msg = pprint.pformat(self._state['purchase_orders'])
        elif args == 'po_hist':
            msg = pprint.pformat(self._state['purchase_order_history'])
        elif args == 'inbound_shipments':
            msg = pprint.pformat(self._state['inbound_shipment_history'])
        elif args == 'outbound_shipments':
            msg = pprint.pformat(self._state['outbound_shipment_history'])
        elif args == 'transfer_shipments':
            msg = pprint.pformat(self._state['transfer_shipment_history'])
        elif args == "reward":
            msg = pprint.pformat(self._reward)
        else:
            msg = "Valid options are: nodes | edges | actions | orders | order_hist | POs | po_hist | " + \
                "inbound_shipments | outbound_shipments | transfer_shipments | reward"

        self.poutput(msg)

    def do_visualise(self, _):
        """Visualise the network"""

        # Make sure any pre-existing figures are closed
        plt.close()

        # If cartopy is available then use it
        if importlib.util.find_spec("cartopy"):
            import cartopy.crs as ccrs
            fig, ax = plt.subplots(1, 1, figsize=(
                10, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))

            # Add coastlines
            ax.coastlines()

            # Limit axes to the UK
            ax.set_extent([-8., 3., 49., 60.])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Get the graph object and coordinates of each node in the network
        G = self._state['network']
        pos = {node_name: node_data['location']
               for node_name, node_data in G.nodes(data=True)}

        # Plot the nodes and node labels
        # Each node type has its own color
        # Nodes with negative capacity or capacity >= the max have red label
        node_type_dict = defaultdict(list)
        for node_name, node_details in G.nodes.items():
            # This will be used when plotting the nodes
            node_type_dict[node_details['node_type']].append(node_name)

            # Determine the total inventory onhand
            onhand = 0
            onhand += sum(node_details.get('inventory', {None: 0}).values())
            onhand += node_details.get('delivered', 0)

            # Very rudimentary way of determining a node is over capacity
            # Could be that it is over capacity for one ASIN, but below capacity for all the others
            text_col = 'k'
            if onhand < 0 or onhand >= sum(node_details.get('max_inventory', {None: 10e10}).values()):
                text_col = 'r'

            # Add label with quantity of product held
            nx.draw_networkx_labels(G, pos=pos, ax=ax, labels={
                                    node_name: onhand}, font_color=text_col, clip_on=False)

        # Plot the nodes, with a different colour for each type
        color_cycle = cycle(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        for node_type, node_names in node_type_dict.items():
            nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=node_names, node_color=next(
                color_cycle), label=node_type)

        # Add node names as labels
        shifted_pos = {k: [v[0], v[1]+.35] for k, v in pos.items()}
        nx.draw_networkx_labels(G, shifted_pos, clip_on=False)

        # Identify uni and bidirectional edges; record the total quantity of product in transit along each edge
        unidirectional_edges = {}
        bidirectional_edges = {}
        for edge_start, edge_end, edges_details in G.edges(data=True):
            if (edge_end, edge_start) in unidirectional_edges:
                # Move entry from unidirectional edge dict to bidirectional dict
                bidirectional_edges[(edge_end, edge_start)] = unidirectional_edges[(
                    edge_end, edge_start)]
                del unidirectional_edges[(edge_end, edge_start)]
                edge_dict = bidirectional_edges
            else:
                edge_dict = unidirectional_edges

            edge_dict[(edge_start, edge_end)] = 0
            for shipment in edges_details['shipments']:
                edge_dict[(edge_start, edge_end)] += shipment['quantity']

        # Plot the unidirectional edges as straight lines and add labels
        nx.draw_networkx_edges(
            G, pos=pos, ax=ax, edgelist=unidirectional_edges, arrows=True)
        nx.draw_networkx_edge_labels(
            G, pos=pos, ax=ax, edge_labels=unidirectional_edges)

        # Plot the bidirectional edges as curves and add labels
        for k, v in bidirectional_edges.items():
            edge_start, edge_end = k

            # Edge is plotted in red if there is flow along it, else it is dashed
            if v != 0:
                nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=[
                                       k], arrows=True, connectionstyle='arc3,rad=0.2', width=1.5, edge_color='r')
            else:
                nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=[
                                       k], arrows=True, connectionstyle='arc3,rad=0.2', style='--')

            # If there is flow in both directions then indicate this using x/y label; else only label with single quantity
            # There will be no label if there is no flow in either direction
            if v != 0 and bidirectional_edges[(edge_end, edge_start)] > v:
                nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels={
                                             k: f'{v}/{bidirectional_edges[(edge_end, edge_start)]}'}, bbox={'alpha': 0})
            elif v != 0 and bidirectional_edges[(edge_end, edge_start)] == 0:
                nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels={
                                             k: v}, bbox={'alpha': 0})

        # Add title showing current clock and time values
        current_clock = self._state['clock']
        current_time = self._state['date_time']
        ax.set_title(f"Clock: {current_clock}; Time: {current_time}")

        # Display the legend
        plt.legend()

        # Display the plot
        plt.show(block=True)

    def _print_nodes(self):
        return pprint.pformat([(node, node_data) for node, node_data in self._state['network'].nodes(data=True)])

    def _print_edges(self):
        return pprint.pformat([(source_node, dest_node, edge_data) for source_node, dest_node, edge_data in self._state["network"].edges(data=True)])

    bp_parser = argparse.ArgumentParser()
    bp_parser.add_argument('time', help='time when to break')

    @cmd2.with_argparser(bp_parser)
    def do_breakpoint(self, args):
        """Set break-point for the specified time-step."""
        self._breakpoints.append(int(args.time))


def main():
    app = MiniSCOTDebuggerApp()
    sys.exit(app.cmdloop())


if __name__ == "__main__":
    main()
