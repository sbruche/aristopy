#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#    S O U R C E   A N D   S I N K
# ==============================================================================
"""
* File name: sourceSink.py
* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)

Sources and sinks are responsible for the transportation of commodities
across the system boundary into and out of the EnergySystem instance.
The Sink class inherits from the Source class. Both have the same input
parameters with only one exception: The Sink has an "inlet" instead of
an "outlet" attribute.
"""
import pyomo.environ as pyomo
from aristopy.component import Component
from aristopy import utils


class Source(Component):
    def __init__(self, ensys, name, outlet, basic_variable='outlet_variable',
                 has_existence_binary_var=False, has_operation_binary_var=False,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 commodity_cost=0, commodity_revenues=0,
                 **kwargs  # only a backdoor for 'inlet' keyword of Sink class
                 ):

        """
        Initialize an instance of the Source class.

        *See the documentation of the Component class for a description of all
        keyword arguments and inherited methods.*

        :param commodity_rate_min: Scalar value or time series that provides a
            minimal value (lower bound) for the basic variable (typically, Sink
            inlet commodity, or Source outlet commodity) for every time step.
            |br| *Default: None*
        :type commodity_rate_min: int, or float, or aristopy Series, or None

        :param commodity_rate_max: Scalar value or time series that provides a
            maximal value (upper bound) for the basic variable (typically, Sink
            inlet commodity, or Source outlet commodity) for every time step.
            |br| *Default: None*
        :type commodity_rate_max: int, or float, or aristopy Series, or None

        :param commodity_rate_fix: Scalar value or time series that provides a
            fixed value for the basic variable (typically, Sink inlet commodity,
            or Source outlet commodity) for every time step.
            |br| *Default: None*
        :type commodity_rate_fix: int, or float, or aristopy Series, or None

        :param commodity_cost: Incurred costs for the use / expenditure of the
            basic variable. Keyword argument takes scalar values or time series
            data (Note: scalar values provide the same functionality like
            keyword argument 'opex_operation').
            |br| *Default: 0*
        :type commodity_cost: int, or float, or aristopy Series

        :param commodity_revenues: Accruing revenues associated with the
            allocation of the basic variable. Keyword argument takes scalar
            values or time series data.
            |br| *Default: 0*
        :type commodity_revenues: int, or float, or aristopy Series
        """

        inlet = None  # init (used for Source) => is overwritten for Sink class

        # Source: needs outlet Flow (not None!) and kwargs should not be used
        if self.__class__ == Source:
            if outlet is None:
                raise utils.io_error_message('Source', name, 'outlet')
            if len(kwargs.keys()) != 0:
                raise ValueError('Found unexpected keyword arguments for Source'
                                 ' component: %s' % list(kwargs.keys()))
        # Sinks: needs inlet Flow (not None!). __init__ doesn't allow kwargs
        if self.__class__ == Sink:
            inlet = kwargs['inlet']
            if inlet is None:
                raise utils.io_error_message('Sink', name, 'inlet')

        Component.__init__(self, ensys=ensys, name=name,
                           inlet=inlet, outlet=outlet,
                           basic_variable=basic_variable,
                           has_existence_binary_var=has_existence_binary_var,
                           has_operation_binary_var=has_operation_binary_var,
                           time_series_data=time_series_data,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity,
                           capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           opex_operation=opex_operation
                           )

        # Check and set additional input arguments
        self.commodity_cost, self.commodity_cost_time_series = \
            utils.check_and_set_cost_and_revenues(self, commodity_cost)
        self.commodity_revenues, self.commodity_revenues_time_series = \
            utils.check_and_set_cost_and_revenues(self, commodity_revenues)

        # Check and set data for commodity rates (scalar or time series or None)
        self.op_rate_min, self.op_rate_max, self.op_rate_fix = \
            utils.check_and_set_commodity_rates(
                self, commodity_rate_min, commodity_rate_max,
                commodity_rate_fix)

        # Last step: Add the component to the EnergySystem instance
        self.add_to_energy_system(ensys, name)

    def __repr__(self):
        return '<Source: "%s">' % self.name

    # ==========================================================================
    #    C O N V E N T I O N A L   C O N S T R A I N T   D E C L A R A T I O N
    # ==========================================================================
    def declare_component_constraints(self, ensys, model):
        """
        Method to declare all component constraints.

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        # Time-independent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()

        # Time-dependent constraints:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_bi_var_ex_and_op_relation(model)
        self.con_operation_limit(model)
        self.con_couple_op_binary_and_basic_var(model)
        self.con_commodity_rate_min(model)
        self.con_commodity_rate_max(model)
        self.con_commodity_rate_fix(model)

    # **************************************************************************
    #    Time-dependent constraints
    # **************************************************************************
    def con_operation_limit(self, model):
        """
        The basic variable of a component is limited by its nominal capacity.
        This usually means, the operation (main commodity) of a sink / source
        (MWh) is limited by its nominal power (MW) multiplied with the number of
        hours per time step. E.g.: |br|
        ``Q[p, t] <= CAP * dt``

        *Method is not intended for public access!*
        """
        # Only required if component has a capacity variable
        if self.has_capacity_var:
            cap = self.variables[utils.CAP]['pyomo']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.variables[self.basic_variable]['has_time_set']
            dt = self.ensys.hours_per_time_step

            def con_operation_limit(m, p, t):
                if has_time_set:
                    return basic_var[p, t] <= cap * dt
                else:
                    # Exceptional case: Selection of a scalar basic variable
                    return basic_var <= cap

            setattr(self.block, 'con_operation_limit', pyomo.Constraint(
                model.time_set, rule=con_operation_limit))

    def con_commodity_rate_min(self, model):
        """
        The basic variable of a component needs to have a minimal value of
        "commodity_rate_min" in every time step. (Without correction with
        "hours_per_time_step" because it should already be included in the time
        series). E.g.: |br|
        ``Q[p, t] >= op_rate_min[p, t]``

        *Method is not intended for public access!*
        """
        # Only required if component has a time series for "commodity_rate_min".
        if self.op_rate_min is not None:
            op_min = self.parameters[self.op_rate_min]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_min]['has_time_set']

            def con_commodity_rate_min(m, p, t):
                if has_time_set:
                    return basic_var[p, t] >= op_min[p, t]
                else:
                    # if 'commodity_rate_min' is provided as scalar value
                    return basic_var[p, t] >= op_min
            setattr(self.block, 'con_commodity_rate_min', pyomo.Constraint(
                model.time_set, rule=con_commodity_rate_min))

    def con_commodity_rate_max(self, model):
        """
        The basic variable of a component can have a maximal value of
        "commodity_rate_max" in every time step. (Without correction with
        "hours_per_time_step" because it should already be included in the time
        series). E.g.: |br|
        ``Q[p, t] <= op_rate_max[p, t]``

        *Method is not intended for public access!*
        """
        # Only required if component has a time series for "commodity_rate_max".
        if self.op_rate_max is not None:
            op_max = self.parameters[self.op_rate_max]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_max]['has_time_set']

            def con_commodity_rate_max(m, p, t):
                if has_time_set:
                    return basic_var[p, t] <= op_max[p, t]
                else:
                    # if 'commodity_rate_max' is provided as scalar value
                    return basic_var[p, t] <= op_max
            setattr(self.block, 'con_commodity_rate_max', pyomo.Constraint(
                model.time_set, rule=con_commodity_rate_max))

    def con_commodity_rate_fix(self, model):
        """
        The basic variable of a component needs to have a value of
        "commodity_rate_fix" in every time step. (Without correction with
        "hours_per_time_step" because it should already be included in the time
        series). E.g.: |br|
        ``Q[p, t] == op_rate_fix[p, t]``

        *Method is not intended for public access!*
        """
        # Only required if component has a time series for "commodity_rate_fix"
        if self.op_rate_fix is not None:
            op_fix = self.parameters[self.op_rate_fix]['values']
            basic_var = self.variables[self.basic_variable]['pyomo']
            has_time_set = self.parameters[self.op_rate_fix]['has_time_set']

            def con_commodity_rate_fix(m, p, t):
                if has_time_set:
                    return basic_var[p, t] == op_fix[p, t]
                else:
                    # if 'commodity_rate_fix' is provided as scalar value
                    return basic_var[p, t] == op_fix
            setattr(self.block, 'con_commodity_rate_fix', pyomo.Constraint(
                model.time_set, rule=con_commodity_rate_fix))

    # ==========================================================================
    #    O B J E C T I V E   F U N C T I O N   C O N T R I B U T I O N
    # ==========================================================================
    def get_objective_function_contribution(self, ensys, model):
        """
        Calculate the objective function contributions of the component and add
        the values to the component dictionary "comp_obj_dict".

        *Method is not intended for public access!*

        :param ensys: Instance of the EnergySystem class
        :param model: Pyomo ConcreteModel of the EnergySystem instance
        """
        # Call method in "Component" class and calculate CAPEX and OPEX
        super().get_objective_function_contribution(ensys, model)

        # Get the basic variable:
        basic_var = self.variables[self.basic_variable]['pyomo']

        # COMMODITY COST AND REVENUES :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Time-independent cost of a commodity (scalar cost value)
        if self.commodity_cost > 0:
            self.comp_obj_dict['commodity_cost'] = \
                -1 * ensys.pvf * self.commodity_cost * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in model.time_set) / ensys.number_of_years

        # Time-dependent cost of a commodity (time series cost values)
        if self.commodity_cost_time_series is not None:
            cost_ts = self.parameters[self.commodity_cost_time_series]['values']
            self.comp_obj_dict['commodity_cost'] = -1 * ensys.pvf * sum(
                cost_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in model.time_set) / ensys.number_of_years

        # Time-independent revenues of a commodity (scalar revenue value)
        if self.commodity_revenues > 0:
            self.comp_obj_dict['commodity_revenues'] = \
                ensys.pvf * self.commodity_revenues * sum(
                    basic_var[p, t] * ensys.period_occurrences[p]
                    for p, t in model.time_set) / ensys.number_of_years

        # Time-dependent revenues of a commodity (time series revenue values)
        if self.commodity_revenues_time_series is not None:
            rev_ts = self.parameters[
                self.commodity_revenues_time_series]['values']
            self.comp_obj_dict['commodity_revenues'] = ensys.pvf * sum(
                rev_ts[p, t] * basic_var[p, t] * ensys.period_occurrences[p]
                for p, t in model.time_set) / ensys.number_of_years

        return sum(self.comp_obj_dict.values())

    # ==========================================================================
    #    S E R I A L I Z E
    # ==========================================================================
    def serialize(self):
        """
        This method collects all relevant input data and optimization results
        from the Component instance, and returns them in an ordered dictionary.

        :return: OrderedDict
        """
        comp_dict = super().serialize()
        comp_dict['commodity_rate_min'] = self.op_rate_min
        comp_dict['commodity_rate_max'] = self.op_rate_max
        comp_dict['commodity_rate_fix'] = self.op_rate_fix
        comp_dict['commodity_cost_time_series'] = \
            self.commodity_cost_time_series
        comp_dict['commodity_revenues_time_series'] = \
            self.commodity_revenues_time_series
        return comp_dict


class Sink(Source):
    def __init__(self, ensys, name, inlet, basic_variable='inlet_variable',
                 has_existence_binary_var=None, has_operation_binary_var=None,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 commodity_rate_min=None, commodity_rate_max=None,
                 commodity_rate_fix=None,
                 commodity_cost=0, commodity_revenues=0
                 ):
        """
        Initialize an instance of the Sink class.

        The Sink class inherits from the Source class. Both have the same input
        parameters with only one exception: The Sink has an "inlet" instead of
        an "outlet" attribute.

        *See the documentation of the Component class and the Source class for
        a description of all keyword arguments and inherited methods.*
        """

        Source.__init__(self, ensys=ensys, name=name,
                        outlet=None, inlet=inlet,  # introduced by **kwargs!
                        basic_variable=basic_variable,
                        has_existence_binary_var=has_existence_binary_var,
                        has_operation_binary_var=has_operation_binary_var,
                        time_series_data=time_series_data,
                        scalar_params=scalar_params,
                        additional_vars=additional_vars,
                        user_expressions=user_expressions,
                        capacity=capacity, capacity_min=capacity_min,
                        capacity_max=capacity_max,
                        capex_per_capacity=capex_per_capacity,
                        capex_if_exist=capex_if_exist,
                        opex_per_capacity=opex_per_capacity,
                        opex_if_exist=opex_if_exist,
                        opex_operation=opex_operation,
                        commodity_rate_min=commodity_rate_min,
                        commodity_rate_max=commodity_rate_max,
                        commodity_rate_fix=commodity_rate_fix,
                        commodity_cost=commodity_cost,
                        commodity_revenues=commodity_revenues)

    def __repr__(self):
        return '<Sink: "%s">' % self.name
