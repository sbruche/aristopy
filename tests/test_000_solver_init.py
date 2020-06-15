import pytest
import aristopy as ar

# Load aristopy's Logger class
log = ar.Logger(default_log_level='WARNING').get_logger(__name__)

def test_find_available_solvers():

    # Create basic energy system instance
    es = ar.EnergySystem()

    # Run the optimization and check availability of different solvers
    available_solvers = []
    for s in ['gurobi', 'cplex', 'cbc', 'glpk']:
        try:
            es.optimize(solver=s, tee=False, results_file=None)
            available_solvers.append(s)
        except Exception:
            pass

    log.info('Available solvers are: %s' % available_solvers)

    # Add the available solvers to the namespace of pytest for this run
    pytest.AVAILABLE_SOLVERS = available_solvers

    assert len(available_solvers) > 0, \
        'Could not find a solver. Make sure you have a solver installed and ' \
        'the path to the solver is set up correctly.'


if __name__ == '__main__':
    test_find_available_solvers()
