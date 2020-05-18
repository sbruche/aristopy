import pytest
import aristopy as ar

log = ar.Logger(default_log_level='WARNING').get_logger(__name__)


def test_storage_example():
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

    chp = ar.Conversion(ensys=es, name='chp', basic_commodity='Q',
                        inlet_connections={'F': 'gas_src'},
                        outlet_connections={'Q': 'bus', 'P': 'elec_snk'},
                        capex_per_capacity=600e3, has_operation_binary_var=True,
                        start_up_cost=10e3, min_load_rel=0.5,
                        capacity=6, user_expressions=['Q_OUT == 0.5 * F_IN',
                                                      'P_OUT == 0.4 * F_IN'])

    storage = ar.Storage(ensys=es, name='storage', basic_commodity='Q',
                         inlet_connections='bus', outlet_connections='bus',
                         capex_per_capacity=1000, capacity_max=50,
                         opex_discharging=1e-9, soc_initial=0.5,
                         self_discharge=0.0)

    bus = ar.Bus(es, 'bus', 'Q', outlet_connections=['heat_snk'], losses=0)

    elec_snk = ar.Sink(ensys=es, name='elec_snk', basic_commodity='P',
                       commodity_revenues=42)

    heat_snk = ar.Sink(ensys=es, name='heat_snk', basic_commodity='Q',
                       commodity_rate_fix='demand',
                       time_series_data={'demand': [4, 4, 6, 6, 10, 10, 4, 4]})

    # Run the model as it is and assert the obj. fct. value
    es.optimize(declares_optimization_problem=True,
                solver=solver, tee=False, results_file=None)

    assert es.pyM.Obj() == pytest.approx(-3.94438400000876e+06)

    # Perform time series aggregation and run the model again.
    # The overall result should be the same.
    es.cluster(number_of_typical_periods=3, number_of_time_steps_per_period=2)
    es.optimize(declares_optimization_problem=True,
                time_series_aggregation=True,
                solver=solver, tee=False, results_file=None)

    assert es.pyM.Obj() == pytest.approx(-3.94438400000876e+06)

    # Problem should be infeasible for clustered data, without using the
    # inter-period-formulation.
    storage.use_inter_period_formulation = False
    es.optimize(declares_optimization_problem=True,
                time_series_aggregation=True,
                solver=solver, tee=False, results_file=None)

    assert es.run_info['termination_condition'] in [
        'infeasible', 'unbounded', 'infeasibleOrUnbounded']


if __name__ == '__main__':
    test_storage_example()
