import pytest
import aristopy as ar

log = ar.Logger(default_log_level='WARNING').get_logger(__name__)


def test_run_minimal_example():
    # Set a solver
    if hasattr(pytest, 'AVAILABLE_SOLVERS') and \
            len(pytest.AVAILABLE_SOLVERS) > 0:
        solver = pytest.AVAILABLE_SOLVERS[0]  # use first one from the list
    else:
        # if file is run as standalone test, pytest will not have the attribute
        # AVAILABLE_SOLVERS (it's set in test_000_...) --> try to use solver CBC
        solver = 'cbc'
    log.info('The following solver is used in this test run: %s' % solver)

    # Create basic energy system model instance
    es = ar.EnergySystemModel(number_of_time_steps=3, hours_per_time_step=1,
                              interest_rate=0.05, economic_lifetime=20)
    # Add source, conversions and sinks
    gas_source = ar.Source(ensys=es, name='gas_source', commodity_cost=20,
                           outlet=ar.Flow('Fuel'))
    gas_boiler = ar.Conversion(ensys=es, name='gas_boiler',
                               basic_variable='Heat',
                               inlet=ar.Flow('Fuel', 'gas_source'),
                               outlet=ar.Flow('Heat', 'heat_sink'),
                               capacity_max=150, capex_per_capacity=60e3,
                               user_expressions='Heat == 0.9 * Fuel')
    chp_unit = ar.Conversion(ensys=es, name='chp_unit', basic_variable='Elec',
                             inlet=ar.Flow('Fuel', 'gas_source'),
                             outlet=[ar.Flow('Heat', 'heat_sink'),
                                     ar.Flow('Elec', 'elec_sink')],
                             capacity_max=100, capex_per_capacity=600e3,
                             user_expressions=['Heat == 0.5 * Fuel',
                                               'Elec == 0.4 * Fuel'])
    heat_sink = ar.Sink(ensys=es, name='heat_sink', inlet=ar.Flow('Heat'),
                        commodity_rate_fix='heat_demand',
                        time_series_data={'heat_demand': [100, 200, 150]})
    elec_sink = ar.Sink(ensys=es, name='elec_sink', inlet=ar.Flow('Elec'),
                        commodity_revenues=30)
    es.optimize(solver=solver, tee=False, results_file=None)

    assert es.pyM.Obj() == pytest.approx(-3.4914796e+08)


if __name__ == '__main__':
    test_run_minimal_example()
