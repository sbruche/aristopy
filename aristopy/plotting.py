import json
import ast
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
from aristopy import utils

# The results from the optimization are exported to dictionaries and stored as
# strings in the json-file to easily handle multidimensional indices (e.g.
# tuples). To evaluate the Python strings use the function "literal_eval" from
# the python built in library "ast". (the strings can only consist of:
# strings, bytes, numbers, tuples, lists, dicts, sets, booleans, and None)
# https://stackoverflow.com/questions/4547274/convert-a-python-dict-to-a-string-and-back


class Plotter:
    def __init__(self, json_file):

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

        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        self.line_colors = ['black', 'red', 'blue', 'green', 'orange' 'brown']
        # 'tab20' contains 10 discrete bar_colors (first from 0 to 0.999, ...)
        self.bar_colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
        # Global properties dictionary:
        self.props = {'fig_width': 10, 'fig_height': 6,
                      'bar_width': 1, 'bar_lw': 0.5, 'line_lw': 2,
                      'xlabel': 'Hours of the year [h]', 'ylabel': '',
                      'grid': False, 'lgd_ncol': 1, 'lgd_pos': 'best',
                      'save_pgf': False, 'save_pdf': False,
                      'save_png': True, 'dpi': 200, 'pad_inches': None}

    # --------------------------------------------------------------------------
    def plot_operation(self, component_name, plot_single_period_with_index=None,
                       level_of_detail=2, show_plot=False, save_plot=True,
                       file_name='operation_plot', **kwargs):

        # Check the user input:
        utils.check_plot_operation_input(
            self.data, component_name, plot_single_period_with_index,
            level_of_detail, show_plot, save_plot, file_name)

        self.single_period = plot_single_period_with_index
        self.level_of_detail = level_of_detail
        self.comp = component_name
        self.model_class = self.data['components'][self.comp]['model_class']

        # Get the global plotting properties of the Plotter class (defaults) and
        # overwrite with local kwargs if specified and found in the global dict.
        props = self.props
        for key, val in kwargs.items():
            if key in props.keys():
                props[key] = val
            else:
                warn('Keyword argument "{}" is unknown and ignored'.format(key))

        # ===========================================================
        #    P L O T T I N G
        # ===========================================================
        fig, ax = plt.subplots(figsize=(props['fig_width'],
                                        props['fig_height']))
        try:
            # ***********************************************************
            #    Plotting: Sink, Source, Conversion
            # ***********************************************************
            if self.model_class in ['Sink', 'Source', 'Conversion']:

                # Get and plot BASIC variable as bar
                name, var = self._get_and_convert_variable('basic_variable')
                if var is not None:
                    idx = self._get_index(additional_time_step=False).flatten()
                    ax.bar(idx, var.flatten(), props['bar_width'], align='edge',
                           label=name, zorder=5, color=self.bar_colors[0],
                           edgecolor='black', linewidth=props['bar_lw'])

                # Todo: Add the possibility to plot more details
                #  (level_of_detail=2) -> where is input coming from or where
                #  does output go to?!
                if level_of_detail == 2:
                    # Find out if basic variable is on an inlet or an outlet
                    # ... SKIP for now ...
                    # Should I organize my information handling for the
                    # ports and vars in a better way?
                    # See idea on Desktop.
                    if self.model_class == 'Sink':  # should be an inlet
                        pass

                '''
                # Order the data in "stacked" according to their sum --> easier
                # to read if the series with frequent occurrence is at bottom.
                # Create lists in original order:
                names, data = list(stacked.keys()), list(stacked.values())
                # Calculate new order (from large to small):
                order = np.array([abs(sum(v)) for v in data]).argsort()[::-1]
                # Set new order for data and names
                data_ordered = np.vstack([data[i] for i in order])
                names_ordered = [names[i] for i in order]
                
                # Plot stacked data:
                for i, val in enumerate(data_ordered):
                    ax.bar(plot_idx, val, props['bar_width'],
                           bottom=data_ordered[:i].sum(axis=0), align='edge',
                           label=names_ordered[i], zorder=5,
                           edgecolor='black', linewidth=props['bar_lw'])
                '''
                # Plot OPERATION RATES as step function (if applied):
                count = 0
                for rate in ['operation_rate_min', 'operation_rate_max',
                             'operation_rate_fix']:
                    name, para = self._get_and_convert_variable(rate)
                    if para is not None:
                        idx = self._get_index(
                            additional_time_step=False).flatten()
                        # Extend the data by appending last value at the end
                        # again --> better representation in the step function!
                        idx_ext = np.append(idx, idx[-1] + self.dt)
                        para_ext = np.append(para.flatten(), para.flatten()[-1])

                        ax.step(idx_ext, para_ext, where='post', label=name,
                                zorder=10, color=self.line_colors[count],
                                linewidth=props['line_lw'])
                        count += 1

            # ***********************************************************
            #    Storage Plotting
            # ***********************************************************
            elif self.model_class == 'Storage':

                # Get and plot CHARGE variable as bar
                name, var = self._get_and_convert_variable('charge_variable')
                if var is not None:
                    idx = self._get_index(additional_time_step=False).flatten()
                    ax.bar(idx, var.flatten(), props['bar_width'], align='edge',
                           label=name, zorder=5, color=self.bar_colors[0],
                           edgecolor='black', linewidth=props['bar_lw'])

                # Get and plot DISCHARGE variable as bar (multiplied with -1)
                name, var = self._get_and_convert_variable('discharge_variable')
                if var is not None:
                    idx = self._get_index(additional_time_step=False).flatten()
                    ax.bar(idx, var.flatten() * -1, props['bar_width'],
                           align='edge', label=name, zorder=5,
                           color=self.bar_colors[1], edgecolor='black',
                           linewidth=props['bar_lw'])

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
                if var is not None:
                    idx = self._get_index(additional_time_step=True)
                    for i, p_var in enumerate(soc):
                        ax.plot(idx[i], p_var, label=(name if i == 0 else None),
                                zorder=10, color=self.line_colors[0],
                                linewidth=props['line_lw'])

                # Add horizontal line at y=0
                ax.axhline(0, color='black', lw=0.8)

            # ***********************************************************
            #    Bus Plotting
            # ***********************************************************
            elif self.model_class == 'Bus':
                # Collect and convert data:
                # Todo: Add function for the Bus class!
                raise NotImplementedError

        # ***********************************************************
        #    General Layouts and Finishing
        # ***********************************************************
            # Plot vertical lines to separate individual typical periods
            # that are connected to represent the full scale time series.
            if self.is_clustered and self.single_period is None:
                for p in range(1, self.nbr_of_periods):
                    x = p * self.nbr_of_ts_per_period * self.dt
                    ax.axvline(x, color='black', lw=1.5,
                               linestyle='--', zorder=100)

        # Catch Exception if problem occurs and print a message in the title
        except Exception as e:
            ax.set_title('PLOTTING FAILED!', size=40, color='red', ha='center')
            print('*** Exception detected while trying to plot:', e)

        ax.set_xlabel(props['xlabel'])
        ax.set_ylabel(props['ylabel'])
        ax.legend(ncol=props['lgd_ncol'], loc=props['grid'],
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
        # check if it is in the variables or parameters dictionary.
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
        else:
            # warn('Could not find "{}" in "{}"'.format(name, self.comp))
            return None

    # --------------------------------------------------------------------------
    def _get_index(self, additional_time_step=False):
        idx = []
        add_ts = 1 if additional_time_step else 0
        if self.single_period is not None:
            # [[0, 1]] or [[0, 1, 2]]
            end = self.nbr_of_ts_per_period * self.dt + add_ts
            idx.append([i for i in range(0, end, self.dt)])
        elif self.is_clustered:
            # [[0, 1], [2, 3]] or [[0, 1, 2], [2, 3, 4]]
            for p in range(self.nbr_of_periods):
                start = self.nbr_of_ts_per_period * p * self.dt
                end = self.nbr_of_ts_per_period * (p + 1) * self.dt + add_ts
                idx.append([i for i in range(start, end, self.dt)])
        else:
            # [[0, 1, 2, 3, 4, 5, 6, 7]] or [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
            end = self.nbr_of_ts * self.dt + add_ts
            idx.append([i for i in range(0, end, self.dt)])
        return np.array(idx)
