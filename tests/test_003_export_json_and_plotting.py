import os
import pytest
import aristopy as ar

log = ar.Logger(default_log_level='WARNING').get_logger(__name__)

def test_export_and_plot():
    # Set a solver
    if hasattr(pytest, 'AVAILABLE_SOLVERS') and \
            len(pytest.AVAILABLE_SOLVERS) > 0:
        solver = pytest.AVAILABLE_SOLVERS[0]  # use first one from the list
    else:
        # if file is run as standalone test, pytest will not have the attribute
        # AVAILABLE_SOLVERS (it's set in test_000_...) --> try to use solver CBC
        solver = 'gurobi'
    log.info('The following solver is used in this test run: %s' % solver)

    es = ar.EnergySystemModel(number_of_time_steps=8, hours_per_time_step=1,
                              interest_rate=0, economic_lifetime=1)

    gas_src = ar.Source(ensys=es, name='gas_src', basic_commodity='F',
                        commodity_cost=20)

    boiler = ar.Conversion(ensys=es, name='boiler', basic_commodity='Q',
                           inlet_connections={'F': 'gas_src'},
                           outlet_connections={'Q': 'bus'},
                           capex_per_capacity=60e3, capacity_max=6,
                           user_expressions=['Q_OUT == F_IN'])

    chp = ar.Conversion(ensys=es, name='chp', basic_commodity='Q',
                        inlet_connections={'F': 'gas_src'},
                        outlet_connections={'Q': 'bus', 'P': 'elec_snk'},
                        capex_per_capacity=600e3, has_operation_binary_var=True,
                        start_up_cost=10e3, min_load_rel=0.5,
                        capacity=4, user_expressions=['Q_OUT == 0.5 * F_IN',
                                                      'P_OUT == 0.4 * F_IN'])

    storage = ar.Storage(ensys=es, name='storage', basic_commodity='Q',
                         inlet_connections='bus', outlet_connections='bus',
                         capex_per_capacity=1000, capacity_max=50,
                         opex_discharging=1e-9, soc_initial=0.6,
                         self_discharge=0.05,
                         precise_inter_period_modeling=True)

    bus = ar.Bus(es, 'bus', 'Q', outlet_connections=['heat_snk'], losses=0)

    elec_snk = ar.Sink(ensys=es, name='elec_snk', basic_commodity='P',
                       commodity_revenues=42)

    heat_snk = ar.Sink(ensys=es, name='heat_snk', basic_commodity='Q',
                       commodity_rate_fix='demand',
                       time_series_data={'demand': [4, 4, 6, 6, 10, 10, 4, 4]})

    # Helper function to write to or get from subdirectory 'temp'
    def temp_file(file_name):
        return os.path.join(os.path.dirname(__file__), 'temp', file_name)

    # Run the model as it is and assert the obj. fct. value
    es.optimize(declares_optimization_problem=True, solver=solver, tee=False,
                results_file=temp_file('results.json'))

    assert es.pyM.Obj() == pytest.approx(-3.16985142475922e+06)

    plotter = ar.Plotter(json_file=temp_file('results.json'))

    plotter.plot_operation('bus', 'Q', level_of_detail=2, period_lw=0.5,
                           file_name=temp_file('bus'),
                           ylabel='Thermal energy transmission [MW/timestep]')

    plotter.plot_operation('storage', 'Q', level_of_detail=2, period_lw=0.5,
                           file_name=temp_file('storage'),
                           ylabel='Charging and discharging [MW] and SOC [MWh]')

    plotter.plot_operation('gas_src', 'F', level_of_detail=1, period_lw=0.5,
                           plot_single_period_with_index=0,
                           file_name=temp_file('fuel_single'))

    plotter.plot_objective(bar_width=0.8, bar_lw=0, show_plot=False,
                           file_name=temp_file('objective'))

    plotter.quick_plot('elec_snk', 'P_IN', save_plot=True,
                       file_name=temp_file('quick_plot'))


if __name__ == '__main__':
    test_export_and_plot()
