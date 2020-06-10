#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#    P L O T T E R
# ==============================================================================
"""
* File name: plotter.py
* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)

The Plotter class provides three basic plotting methods:

* 'plot_operation': A mixed bar and line plot that visualizes the operation of
  a component on the basis of a selected commodity.

* 'plot_objective': Bar chart that summarizes the cost contributions of each
  component to the overall objective function value.

* 'quick_plot': Quick visualization for the values of one component variable as
  a line, scatter, or bar plot.

Note: The results of the optimization are exported to dictionaries and stored
as strings in a json-file to easily handle multidimensional indices
(e.g. tuples). To evaluate the Python strings we use the function "literal_eval"
from the python built in library "ast". (the strings can only consist of:
strings, bytes, numbers, tuples, lists, dicts, sets, booleans, and None)
[`Ref <https://stackoverflow.com/questions/4547274/
convert-a-python-dict-to-a-string-and-back>`_]
"""
import os
import copy
import json
import ast
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
from aristopy import utils


# Option: Add a method for capet plotting on a requested component variable.

class Plotter:
    def __init__(self, json_file):
        """


        :param json_file: Path to the optimization results file in JSON-Format
        """

        # Leave if no results file available
        if not os.path.isfile(json_file):
            self.data = None
            return

        self.json_file = json_file  # name / path to the json-file

        # Read the data from the json-file
        with open(self.json_file, encoding='utf-8') as f:
            self.data = json.loads(f.read())

        # Read general values from the data dict
        self.nbr_of_ts = self.data['number_of_time_steps']
        self.dt = self.data['hours_per_time_step']
        self.is_clustered = self.data['is_data_clustered']
        self.nbr_of_typ_periods = self.data['number_of_typical_periods']
        self.nbr_of_ts_per_period = self.data['number_of_time_steps_per_period']
        self.nbr_of_periods = self.data['total_number_of_periods']
        self.periods_order = ast.literal_eval(self.data['periods_order'])

        # Init values:
        self.single_period = None  # if clustered: plot only period with idx 'X'
        self.level_of_detail = 2  # 1 (simple) or 2 (more detailed)
        self.comp = ''  # name of the component of interest
        self.model_class = None  # string, class name of comp, e.g. 'Storage'

        # Values used for plotting -> can be changed by 'plot_operation'
        self.dt_plot = self.dt  # init
        self.scale_plot = 1 / self.dt  # init

        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        self.line_colors = ['black', 'red', 'blue', 'green', 'orange' 'brown']
        # 'tab10' contains 20 discrete bar_colors (first from 0 to 0.049, ...)
        # Todo: Simply use a colormap and divide it according to needed colors
        # Extended the available colors with Dark and Accent...
        self.bar_colors = np.r_[plt.get_cmap('tab20')(np.linspace(0, 1, 20)),
                                plt.get_cmap('Dark2')(np.linspace(0, 1, 8)),
                                plt.get_cmap('Accent')(np.linspace(0, 1, 8))]
        # Global properties dictionary:
        self.props = {'fig_width': 10, 'fig_height': 6,
                      'bar_width': 1, 'bar_lw': 0, 'line_lw': 2, 'period_lw': 0,
                      'xlabel': 'Time steps [-]', 'ylabel': '',
                      'xticks_rotation': 0,
                      'grid': False, 'lgd_ncol': 1, 'lgd_pos': 'best',
                      'save_pgf': False, 'save_pdf': False,
                      'save_png': True, 'dpi': 200, 'pad_inches': None}

    # ==========================================================================
    #    P L O T   O B J E C T I V E
    # ==========================================================================
    def plot_objective(self, show_plot=False, save_plot=True,
                       file_name='objective_plot', **kwargs):
        """
        Method to create a bar chart that summarizes the cost contributions of
        each component of the EnergySystem instance to the overall objective
        function value.

        :param show_plot: State whether the plot should be shown once finalized
            |br| *Default: False*
        :type show_plot: bool

        :param save_plot: State whether the plot should be saved once finalized
            |br| *Default: True*
        :type save_plot: bool

        :param file_name: Name of the file (if saved); no file-ending required
            |br| *Default: 'objective_plot'*
        :type file_name: str

        :param kwargs: Additional keyword arguments to manipulate the plot
            (e.g., labels, figure size, legend position, ...).
            See dict 'props' of the Plotter class.
        """
        if self.data is None:
            return

        # Get the global plotting properties of the Plotter class (defaults)
        props = copy.copy(self.props)
        props['xlabel'] = None  # default
        props['ylabel'] = 'Objective function value (contribution)'  # default
        props['bar_width'] = 0.8  # default
        # Overwrite props with local kwargs if specified and found.
        for key, val in kwargs.items():
            if key in props.keys():
                props[key] = val
            else:
                warn('Keyword argument "{}" is unknown and ignored'.format(key))

        # Get the plotting data:
        obj_data = {}
        for comp_name, comp_data in self.data['components'].items():
            # ORDER: capex, opex, start_up, commodity_cost, commodity_revenues
            data = comp_data['comp_obj_dict']

            # Skip the component in the plot if all obj. entries are zero:
            if sum(abs(i) for i in data.values()) <= 0.01:  # rounding errors
                continue

            obj_data[comp_name] = \
                [data['capex_capacity'] + data['capex_exist'],
                 data['opex_capacity'] + data['opex_exist'] + data[
                     'opex_operation'],
                 data['start_up_cost'],
                 data['commodity_cost'],
                 data['commodity_revenues']]

        names = list(obj_data.keys())
        # to vertically stacked and transposed array (1. row: capex, 2. opex,..)
        values = np.vstack(list(obj_data.values())).transpose()
        labels = ['CAPEX', 'OPEX', 'Start up cost',
                  'Commodity cost', 'Commodity revenues']

        # If objective function contributions have been added via method
        # 'add_objective_function_contribution' in EnergySystem:
        added_obj = self.data['added_objective_function_contributions']
        # if dict is not empty and if the sum of all (abs) entries is not zero
        if added_obj and sum(abs(i) for i in added_obj.values()) != 0:
            names.append('Added')
            labels.extend(added_obj.keys())
            add_rows = np.zeros(shape=(len(added_obj.keys()), values.shape[1]))
            add_col = np.append(np.zeros(values.shape[0]),
                                list(added_obj.values()))
            values = np.append(values, values=add_rows, axis=0)
            values = np.insert(values, values.shape[1], values=add_col, axis=1)
            # values = np.c_[values, tot]  # --> faster but bad readability

        # Plot the Total as an overall sum:
        total = values.sum()
        names.append('Total')
        labels.append('Total')
        add_row = np.zeros(shape=(1, values.shape[1]))
        add_col = np.append(np.zeros(values.shape[0]), total)
        values = np.append(values, values=add_row, axis=0)
        values = np.insert(values, values.shape[1], values=add_col, axis=1)

        # ----------------------
        # https://stackoverflow.com/questions/35979852/stacked-bar-charts-using-python-matplotlib-for-positive-and-negative-values
        # Take negative and positive data apart and cumulate
        def get_cumulated_array(data, **kwargs):
            cum = data.clip(**kwargs)
            cum = np.cumsum(cum, axis=0)
            d = np.zeros(np.shape(data))
            d[1:] = cum[:-1]
            return d

        cumulated_data = get_cumulated_array(values, min=0)
        cumulated_data_neg = get_cumulated_array(values, max=0)
        # Re-merge negative and positive data.
        row_mask = (values < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data
        # ----------------------

        # Plot stacked bars for all components and the total
        fig, ax = plt.subplots(figsize=(props['fig_width'],
                                        props['fig_height']))
        for i, category in enumerate(labels):
            ax.bar(names, values[i], props['bar_width'],
                   label=category, bottom=data_stack[i],
                   color=self.bar_colors[i], zorder=10,
                   edgecolor='black', linewidth=props['bar_lw'])

        # Add horizontal line at y=0
        ax.axhline(0, color='black', lw=0.8)

        ax.tick_params(axis='x', labelrotation=props['xticks_rotation'])
        ax.set_xlabel(props['xlabel'])
        ax.set_ylabel(props['ylabel'])
        ax.legend(ncol=props['lgd_ncol'], loc=props['lgd_pos'],
                  framealpha=0.8, edgecolor='black').set_zorder(100)
        if props['grid']:
            ax.grid(which='major', linestyle='--', zorder=0)
        fig.tight_layout(pad=0.0, w_pad=0.2)
        if show_plot:
            plt.show()
        if save_plot:
            if props['save_png']:
                fig.savefig(file_name+'.png', bbox_inches="tight",
                            pad_inches=props['pad_inches'], dpi=props['dpi'])
            if props['save_pdf']:
                fig.savefig(file_name+'.pdf', bbox_inches="tight",
                            pad_inches=props['pad_inches'])
            if props['save_pgf']:
                fig.savefig(file_name+'.pgf', bbox_inches="tight",
                            pad_inches=props['pad_inches'])
        plt.close()

    # ==========================================================================
    #    Q U I C K   P L O T
    # ==========================================================================
    def quick_plot(self, component_name, variable_name, kind='bar',
                   save_plot=False, file_name=None):
        """
        Method to create a quick visualization for the values of one component
        variable as a line, scatter, or bar plot.

        :param component_name: Name of the component that holds the variable
            of interest.
        :type component_name: str

        :param variable_name: Name of the variable (or parameter) that should
            be plotted.
        :type variable_name: str

        :param kind: States the kind of plot. Possible options are:
            'plot' (line plot), 'scatter', 'bar'.
            |br| *Default: 'bar'*
        :type kind: str

        :param save_plot: State whether the plot should be saved once finalized
            |br| *Default: False*
        :type save_plot: bool

        :param file_name: Name of the file (if saved); no file-ending required.
            Name is auto-generated if None is provided and plot should be saved.
            |br| *Default: None*
        :type file_name: str
        """
        if self.data is None:
            return

        # Set the component and try to find values for the requested var / param
        self.comp = component_name
        data = self._get_values(variable_name)
        if data is None:  # return with warning if not successful
            return warn('Could not find variable {} in component {}'
                        .format(variable_name, component_name))

        fig, ax = plt.subplots(figsize=(self.props['fig_width'],
                                        self.props['fig_height']))
        if kind == 'plot':
            ax.plot(range(len(data)), list(data.values()),
                    label=variable_name, zorder=10)
        elif kind == 'scatter':
            ax.scatter(range(len(data)), list(data.values()),
                       label=variable_name, zorder=10)
        elif kind == 'bar':
            ax.bar(range(len(data)), list(data.values()),
                   label=variable_name, zorder=10)

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data.keys()))
        ax.tick_params(axis='x', labelrotation=self.props['xticks_rotation'])
        ax.set_title('Quickplot for component "{}"'.format(component_name),
                     size=16, color='black', ha='center')
        ax.set_xlabel('Time index [period, time step]')
        ax.set_ylabel('Quantity of variable "{}"'.format(variable_name))
        # ax.grid(which='major', linestyle='--', zorder=0)
        ax.legend(framealpha=0.8, edgecolor='black').set_zorder(100)
        fig.tight_layout()
        if save_plot:
            f_name = file_name + '.png' if file_name is not None \
                else '{}_{}.png'.format(component_name, variable_name)
            fig.savefig(f_name, dpi=200)
        else:
            plt.show()

    # ==========================================================================
    #    P L O T   O P E R A T I O N
    # ==========================================================================
    def plot_operation(self, component_name, commodity, level_of_detail=2,
                       scale_to_hourly_resolution=False,
                       plot_single_period_with_index=None, show_plot=False,
                       save_plot=True, file_name='operation_plot', **kwargs):
        """
        Method to create a mixed bar and line plot that visualizes the
        operation of a component on the basis of a selected commodity.

        :param component_name: Name of the component that holds the commodity
            of interest.
        :type component_name: str

        :param commodity: Name of the commodity that should be plotted.
        :type commodity: str

        :param level_of_detail: Specifies the level of plotting detail. Only the
            commodity in the component itself is plotted if 1 is selected.
            The composition of the commodity (from which sources formed and to
            which destinations sent) is visualized if 2 is selected.
            |br| *Default: 2*
        :type level_of_detail: int (1 or 2)

        :param scale_to_hourly_resolution: States if the data should be scaled
            to hourly resolution before plotting. This might be useful, if the
            optimization was performed with a value for the EnergySystem keyword
            argument 'hours_per_time_step' larger than 1.
            |br| *Default: False*
        :type scale_to_hourly_resolution: bool

        :param plot_single_period_with_index: States if only one period with the
            given index number should be plotted. This is only possible if the
            optimization was performed with aggregated time series data.
            |br| *Default: None*
        :type plot_single_period_with_index: int or None

        :param show_plot: State whether the plot should be shown once finalized
            |br| *Default: False*
        :type show_plot: bool

        :param save_plot: State whether the plot should be saved once finalized
            |br| *Default: True*
        :type save_plot: bool

        :param file_name: Name of the file (if saved); no file-ending required
            |br| *Default: 'operation_plot'*
        :type file_name: str

        :param kwargs: Additional keyword arguments to manipulate the plot
            (e.g., labels, figure size, legend position, ...).
            See dict 'props' of the Plotter class.
        """
        if self.data is None:
            return

        # Check the user input:
        utils.check_plot_operation_input(
            self.data, component_name, commodity, scale_to_hourly_resolution,
            plot_single_period_with_index, level_of_detail,
            show_plot, save_plot, file_name)

        self.single_period = plot_single_period_with_index
        self.level_of_detail = level_of_detail
        self.comp = component_name
        self.model_class = self.data['components'][self.comp]['model_class']

        # Get the global plotting properties of the Plotter class (defaults)
        props = copy.copy(self.props)

        # Set 'dt_plot' and 'dt_scale' according to 'scale_to_hourly_resolution'
        # If scaling is requested: Adjust the index and scale the plotted
        # variable values (except of the SOC).
        if scale_to_hourly_resolution:
            self.dt_plot, self.scale_plot = self.dt, 1 / self.dt
            props['bar_width'] = props['bar_width'] * self.dt_plot  # default
            props['xlabel'] = 'Hours of the year [h]'  # default
        else:
            self.dt_plot, self.scale_plot = 1, 1

        # Overwrite props with local kwargs if specified and found.
        for key, val in kwargs.items():
            if key in props.keys():
                props[key] = val
            else:
                warn('Keyword argument "{}" is unknown and ignored'.format(key))

        # **********************************************************************
        #    Plotting
        # **********************************************************************
        fig, ax = plt.subplots(figsize=(props['fig_width'],
                                        props['fig_height']))
        try:
            # 1. Find the required commodity in the inlets and / or outlets of
            # the component and get the associated port variables.
            var_in_name, var_out_name = None, None  # init
            if commodity in self.data['components'][self.comp][
                    'inlet_commod_and_var_names'].keys():
                var_in_name = self.data['components'][self.comp][
                    'inlet_commod_and_var_names'][commodity]
            if commodity in self.data['components'][self.comp][
                    'outlet_commod_and_var_names'].keys():
                var_out_name = self.data['components'][self.comp][
                    'outlet_commod_and_var_names'][commodity]

            if level_of_detail == 1:
                # --------------------------------------------------------------
                #    Only plot the commodity in the component itself
                # --------------------------------------------------------------
                # Get the commodity data for inlets ad outlets
                _, var_in_data = self._get_and_convert_variable(var_in_name)
                _, var_out_data = self._get_and_convert_variable(var_out_name)

                # Plot commodity data on inlet port:
                if var_in_data is not None:
                    idx = self._get_index(additional_time_step=False).flatten()
                    ax.bar(idx, var_in_data.flatten() * self.scale_plot,
                           props['bar_width'],
                           align='edge', label=var_in_name, zorder=5,
                           color=self.bar_colors[0], edgecolor='black',
                           linewidth=props['bar_lw'])

                # Plot commodity data on outlet port:
                if var_out_data is not None:
                    # If commodity also on inlet port -> multiply outlet with -1
                    if var_in_data is not None:
                        var_out_data *= -1

                    idx = self._get_index(additional_time_step=False).flatten()
                    ax.bar(idx, var_out_data.flatten() * self.scale_plot,
                           props['bar_width'],
                           align='edge', label=var_out_name, zorder=5,
                           color=self.bar_colors[1], edgecolor='black',
                           linewidth=props['bar_lw'])

            else:  # level_of_detail == 2
                # --------------------------------------------------------------
                #    Plot the composition of the commodity
                #    (from which sources formed / to which destinations sent)
                # --------------------------------------------------------------
                # Get the connected arc names
                arc_in_names, arc_out_names = [], []  # init
                if var_in_name is not None:
                    arc_in_names = self.data['components'][self.comp][
                        'var_connections'][var_in_name]
                if var_out_name is not None:
                    arc_out_names = self.data['components'][self.comp][
                        'var_connections'][var_out_name]
                  
                # Get the data for the connected arcs at inlets and outlets
                arc_in_data, arc_out_data = [], []  # init
                for arc_name in arc_in_names:
                    _, data = self._get_and_convert_variable(arc_name)
                    arc_in_data.append(data.flatten())
                for arc_name in arc_out_names:
                    _, data = self._get_and_convert_variable(arc_name)
                    arc_out_data.append(data.flatten())

                # Order the data (on each side) according to their sum
                # => easier to read if series with high occurrence is at bottom.
                # Rearrange inlet data (order from large to small and stacked):
                if len(arc_in_data) > 0:
                    order = np.array([sum(v) for v in arc_in_data]).argsort()
                    order = order[::-1]  # reverse order --> from large to small
                    # Set new order for 'arc_in_data' and 'arc_in_names'
                    arc_in_data = np.vstack(
                        [arc_in_data[i] for i in order]) * self.scale_plot
                    arc_in_names = [arc_in_names[i] for i in order]
                # Rearrange outlet data (order from large to small and stacked):
                if len(arc_out_data) > 0:
                    order = np.array([sum(v) for v in arc_out_data]).argsort()
                    order = order[::-1]  # reverse order --> from large to small
                    # Set new order for 'arc_out_data' and 'arc_out_names'
                    arc_out_data = np.vstack(
                        [arc_out_data[i] for i in order]) * self.scale_plot
                    arc_out_names = [arc_out_names[i] for i in order]

                # Create index
                idx = self._get_index(additional_time_step=False).flatten()

                # If commodity also on inlet port -> multiply outlet with -1
                if len(arc_out_data) > 0 and len(arc_in_data) > 0:
                    arc_out_data *= -1

                # Plot stacked bars on inlet port:
                for i, val in enumerate(arc_in_data):
                    if abs(np.sum(val)) <= 0.01:  # skip components with zeros
                        continue
                    ax.bar(idx, val, props['bar_width'],
                           bottom=arc_in_data[:i].sum(axis=0), align='edge',
                           label=arc_in_names[i], zorder=5,
                           color=self.bar_colors[i],
                           edgecolor='black', linewidth=props['bar_lw'])

                # Plot stacked bars on outlet port:
                for i, val in enumerate(arc_out_data):
                    if abs(np.sum(val)) <= 0.01:  # skip components with zeros
                        continue
                    ax.bar(idx, val, props['bar_width'],
                           bottom=arc_out_data[:i].sum(axis=0), align='edge',
                           label=arc_out_names[i], zorder=5,
                           color=self.bar_colors[len(arc_in_names)+i],
                           edgecolor='black', linewidth=props['bar_lw'])

            # ------------------------------------------------------------------
            #    Storage: Add the SOC line and a horizontal line at y=0
            # ------------------------------------------------------------------
            if self.model_class == 'Storage':

                # Get the data for the state of charge variables:
                name, var_soc = self._get_and_convert_variable('soc_variable')
                _, var_soc_inter = self._get_and_convert_variable(
                    'soc_inter_variable')

                # Only in case the data is clustered and should be scaled to the
                # full series (not only one single period is plotted) and the
                # inter-period formulation has been applied --> Recalculate SOC!
                if self.is_clustered and self.single_period is None \
                        and var_soc_inter is not None:
                    soc = np.array([p_soc + var_soc_inter[i]
                                    for i, p_soc in enumerate(var_soc)])
                else:
                    soc = var_soc  # just use the original SOC results

                # Plot the state if charge variable (SOC):
                idx = self._get_index(additional_time_step=True)
                for i, p_var in enumerate(soc):
                    ax.plot(idx[i], p_var, label=(name if i == 0 else None),
                            zorder=10, color=self.line_colors[0],
                            linewidth=props['line_lw'])

                # Add horizontal line at y=0
                ax.axhline(0, color='black', lw=0.8)

            # ------------------------------------------------------------------
            #    Source / Sink:   Add commodity rates as step plots (if applied)
            # ------------------------------------------------------------------
            if self.model_class in ['Source', 'Sink']:
                count = 0  # init counter (to have different colors for lines)
                for rate in ['commodity_rate_min', 'commodity_rate_max',
                             'commodity_rate_fix', 'commodity_cost_time_series',
                             'commodity_revenues_time_series']:
                    name, para = self._get_and_convert_variable(rate)
                    if para is not None:
                        idx = self._get_index(
                            additional_time_step=False).flatten()
                        # Extend the data by appending last value at the end
                        # again --> better representation in the step function!
                        idx_ext = np.append(idx, idx[-1] + self.dt_plot)
                        para_ext = np.append(para.flatten(), para.flatten()[-1])
                        # Plot step function
                        ax.step(idx_ext, para_ext * self.scale_plot,
                                where='post', label=name,
                                zorder=10, color=self.line_colors[count],
                                linewidth=props['line_lw'])
                        count += 1  # increase counter by 1

            # ***********************************************************
            #    General Layouts and Finishing
            # ***********************************************************
            # Plot vertical lines to separate individual typical periods
            # that are connected to represent the full scale time series.
            if self.is_clustered and self.single_period is None:
                for p in range(1, self.nbr_of_periods):
                    x = p * self.nbr_of_ts_per_period * self.dt_plot
                    ax.axvline(x, color='black', lw=props['period_lw'],
                               linestyle='--', zorder=100)

        # Catch Exception if problem occurs and print a message in the title
        except Exception as e:
            ax.set_title('PLOTTING FAILED!', size=40, color='red', ha='center')
            print('*** Exception detected while trying to plot:', e)

        ax.tick_params(axis='x', labelrotation=props['xticks_rotation'])
        ax.set_xlabel(props['xlabel'])
        ax.set_ylabel(props['ylabel'])
        ax.legend(ncol=props['lgd_ncol'], loc=props['lgd_pos'],
                  framealpha=0.8, edgecolor='black').set_zorder(100)
        if props['grid']:
            ax.grid(which='major', linestyle='--', zorder=0)
        fig.tight_layout(pad=0.0, w_pad=0.2)
        if show_plot:
            plt.show()
        if save_plot:
            if props['save_png']:
                fig.savefig(file_name+'.png', bbox_inches="tight",
                            pad_inches=props['pad_inches'], dpi=props['dpi'])
            if props['save_pdf']:
                fig.savefig(file_name+'.pdf', bbox_inches="tight",
                            pad_inches=props['pad_inches'])
            if props['save_pgf']:
                fig.savefig(file_name+'.pgf', bbox_inches="tight",
                            pad_inches=props['pad_inches'])
        plt.close()

    # --------------------------------------------------------------------------
    def _get_and_convert_variable(self, var_name):

        # if the variable is not in the component dict keys (e.g.
        # 'basic_variable', ...) , just use the provided 'var_name' and
        # check if it is in the variables or parameters or the arc dictionary.
        if var_name in self.data['components'][self.comp].keys():
            name = self.data['components'][self.comp][var_name]
        else:
            name = var_name

        data = self._get_values(name)
        # if name does not exist in the variables or parameters or is unused
        if data is None:
            return None, None

        # If the index of the data is not a tuple (e.g. for SOC_INTER) simply
        # convert the values to a list and return it.
        if not isinstance(next(iter(data)), tuple):
            new_data = list(data.values())
            return name, new_data

        # Index of the data is a tuple:
        # Remove all unnecessary data from the index if a single period is asked
        if self.single_period is not None:
            new_data = [[val for key, val in data.items()
                        if key[0] == self.single_period]]

        # Check if the data is clustered and form new data in right order with
        # length of the original (not clustered) time series data.
        elif self.is_clustered:
            # The number of time steps depends on the data. Usually it will be
            # equal to self.nbr_of_ts_per_period but e.g. the SOC has an
            # additional time step at the end of each period --> calculate it!
            ts_per_period = int(len(data) / self.nbr_of_typ_periods)
            new_data = [[data[(p, ts)] for ts in range(ts_per_period)]
                        for p in self.periods_order]

        # If not clustered: Convert values to a list and return data directly.
        else:
            new_data = [list(data.values())]

        return name, np.array(new_data)  # Return data as numpy array

    # --------------------------------------------------------------------------
    def _get_values(self, name):
        # Try to find the variable or parameter in the component dictionary
        comp_dict = self.data['components'][self.comp]
        if name in comp_dict['variables'].keys():
            return ast.literal_eval(comp_dict['variables'][name])
        elif name in comp_dict['parameters'].keys():
            return ast.literal_eval(comp_dict['parameters'][name])
        # Alternatively try to find the (arc) name in dict 'arc_variables':
        elif name in self.data['arc_variables'].keys():
            return ast.literal_eval(self.data['arc_variables'][name])
        else:
            # warn('Could not find data for "{}"'.format(name))
            return None

    # --------------------------------------------------------------------------
    def _get_index(self, additional_time_step=False):
        idx = []
        add_ts = 1 if additional_time_step else 0
        if self.single_period is not None:
            # [[0, 1]] or [[0, 1, 2]]
            end = self.nbr_of_ts_per_period * self.dt_plot + add_ts
            idx.append([i for i in range(0, end, self.dt_plot)])
        elif self.is_clustered:
            # [[0, 1], [2, 3]] or [[0, 1, 2], [2, 3, 4]]
            for p in range(self.nbr_of_periods):
                start = self.nbr_of_ts_per_period * p * self.dt_plot
                end = self.nbr_of_ts_per_period * (p+1) * self.dt_plot + add_ts
                idx.append([i for i in range(start, end, self.dt_plot)])
        else:
            # [[0, 1, 2, 3, 4, 5, 6, 7]] or [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
            end = self.nbr_of_ts * self.dt_plot + add_ts
            idx.append([i for i in range(0, end, self.dt_plot)])
        return np.array(idx)
