import os
import pytest
import aristopy as ar

log = ar.Logger(default_log_level='WARNING').get_logger(__name__)


def test_relax_integrality():
    # Set a solver
    if hasattr(pytest, 'AVAILABLE_SOLVERS') and \
            len(pytest.AVAILABLE_SOLVERS) > 0:
        solver = pytest.AVAILABLE_SOLVERS[0]  # use first one from the list
    else:
        # if file is run as standalone test, pytest will not have the attribute
        # AVAILABLE_SOLVERS (it's set in test_000_...) --> try to use solver CBC
        solver = 'cbc'
    log.info('The following solver is used in this test run: %s' % solver)

    # Helper function to write to or get from subdirectory 'temp'
    def temp_file(file_name):
        return os.path.join(os.path.dirname(__file__), 'temp', file_name)
    # --------------------------------------------------------------------------
    # Simple model with power plant (with minimal-part-load 50% and max. cap.),
    # electricity load and external electricity source.
    es = ar.EnergySystemModel(number_of_time_steps=3, hours_per_time_step=1,
                              interest_rate=0.05, economic_lifetime=20)

    gas_source = ar.Source(ensys=es, name='gas_source', commodity_cost=20,
                           outlet=ar.Flow('FUEL'))

    elec_source = ar.Source(ensys=es, name='elec_source', commodity_cost=1000,
                            outlet=ar.Flow('ELEC', 'elec_sink'))

    power_plant = ar.Conversion(
        es, 'power_plant', 'ELEC',
        inlet=ar.Flow('FUEL', 'gas_source'),
        outlet=ar.Flow('ELEC', 'elec_sink'),
        has_existence_binary_var=True, has_operation_binary_var=True,
        capacity=25, capex_per_capacity=1e6, capex_if_exist=3e6,
        opex_operation=10, min_load_rel=0.5,
        user_expressions='ELEC == 0.5 * FUEL + 1.5 * BI_OP')

    elec_sink = ar.Sink(
        es, 'elec_sink', inlet=ar.Flow('ELEC'),
        commodity_rate_fix=ar.Series('elec_demand', [11, 22, 33]),
        commodity_revenues=ar.Series('elec_rev', [-10, 100, 50]))

    es.optimize(solver='scip', tee=True, results_file=None)
    assert es.pyM.Obj() == pytest.approx(-6.64455051961791e+08)

    es.relax_integrality()
    es.optimize(solver='scip', tee=True, results_file=None)
    assert es.pyM.Obj() == pytest.approx(-2.82261791827754e+08)

    es.reset_component_variables(which_instances=['power_plant'])
    es.optimize(solver='scip', tee=True,
                results_file=temp_file('004_results.json'))
    assert es.pyM.Obj() == pytest.approx(-6.64455051961791e+08)

    plotter = ar.Plotter(json_file=temp_file('004_results.json'))
    plotter.plot_operation('elec_sink', 'ELEC', level_of_detail=2,
                           period_lw=0.5,
                           plot_single_period_with_index=0,
                           file_name=temp_file('004_elec_sink'))
