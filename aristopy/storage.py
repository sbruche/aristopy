#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**The Storage class**

* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
from aristopy import utils
from aristopy.component import Component


class Storage(Component):
    """
    A storage component can collect a commodity at the inlet at one time step
    and make it available at the outlet at another time step.
    Thus, it is a component to provide flexibility.
    """
    def __init__(self, ensys, name, inlet, outlet,
                 basic_variable='inlet_variable',
                 has_existence_binary_var=False,
                 time_series_data=None, scalar_params=None,
                 additional_vars=None, user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0, opex_operation=0,
                 charge_rate=1, discharge_rate=1, self_discharge=0,
                 charge_efficiency=1, discharge_efficiency=1,
                 soc_min=0, soc_max=1, soc_initial=None,
                 use_inter_period_formulation=True,
                 precise_inter_period_modeling=False
                 ):
        """
        Initialize an instance of the Storage class.

        *See the documentation of the Component class for a description of all
        keyword arguments and inherited methods.*

        :param charge_rate: Ratio between the maximum charging power or flow and
            the storage capacity. It indicates the reciprocal value of the
            time for a full storage charging process from empty to full (e.g.,
            'charge_rate'=1/6 => 6 hours needed to load an empty storage fully).
            |br| *Default: 1*
        :type charge_rate: float or int (>=0)

        :param discharge_rate: Ratio between the maximum discharging power or
            flow and the storage capacity. It indicates the reciprocal value of
            the time for a full storage discharging process from full to empty
            (e.g., 'discharge_rate'=1/6 => 6 hours needed for emptying a fully
            loaded storage).
            |br| *Default: 1*
        :type discharge_rate: float or int (>=0)

        :param self_discharge: Share of the storage content that is dissipated
            and can not be used (e.g., heat losses to the environment for a
            heat storage). The value is specified in "percent per hour" [%/h].
            |br| *Default: 0*
        :type self_discharge: float or int (0<=value<=1)

        :param charge_efficiency: Efficiency value for the charging process.
            It indicates the ratio between stored and entering commodities.
            E.g., 'charge_efficiency'=0.9 => for 1 MWh entering the storage in
            one time step 0.9 MWh are stored, and 0.1 MWh are lost.
            |br| *Default: 1*
        :type charge_efficiency: float or int (0<=value<=1)

        :param discharge_efficiency: Efficiency of the discharging process.
            It indicates the ratio between the usable commodity at the outlet
            and the reduction in the stored commodity. E.g.,
            'discharge_efficiency'=0.9 => if SOC is reduced by 1 MWh in one time
            step, 0.9 MWh are available at the outlet, and 0.1 MWh are lost.
            |br| *Default: 1*
        :type discharge_efficiency: float or int (0<=value<=1)

        :param soc_min: Relative value to provide a lower bound for the usable
            storage capacity. E.g., with a storage capacity of 5 MWh and a value
            for 'soc_min' given with 0.2, the SOC cannot fall below 1 MWh.
            |br| *Default: 0*
        :type soc_min: float or int (0<=value<=1)

        :param soc_max: Relative value to provide an upper bound for the usable
            storage capacity. E.g., with a storage capacity of 5 MWh and a value
            for 'soc_max' given with 0.8, the SOC cannot exceed 4 MWh.
            |br| *Default: 1*
        :type soc_max: float or int (0<=value<=1)

        :param soc_initial: Provides a value for the relative state of charge in
            the first time step of the optimization problem (e.g., 0.5 => 50%).
            The initial SOC value is applied to all periods if the model has
            multiple periods (calculation with clustered data), and the keyword
            argument 'use_inter_period_formulation' is set to False.
            |br| *Default: None*
        :type soc_initial: float or int (0<=value<=1), or None

        :param use_inter_period_formulation: States whether a model formulation
            should be applied that connects the states of charge of a storage
            component for otherwise independent periods (only used if time
            series aggregation is applied). Additional variables and constraints
            are created if the keyword argument is set to True. This formulation
            enables the (energy) transport between periods (especially relevant
            for long-term storages) and likewise increases the model complexity.
            |br| *Default: True*
        :type use_inter_period_formulation: bool

        :param precise_inter_period_modeling: States whether the inter-period
            formulation should be implemented in a simplified (False) or precise
            (True) way. The type of formulation influences how the constraints
            are modeled that enforce the SOC's bounds. If the storage has only a
            low self-discharge value, it is recommended to choose the simplified
            version (False). This version introduces some additional variables,
            but requires a significantly smaller number of constraints
            (`Ref: DOI 10.1016/j.apenergy.2018.01.023
            <https://doi.org/10.1016/j.apenergy.2018.01.023>`_).
            |br| *Default: False*
        :type precise_inter_period_modeling: bool
        """

        # Prevent None at inlet & outlet! (Flows are checked in Component init)
        if inlet is None:
            raise utils.io_error_message('Storage', name, 'inlet')
        if outlet is None:
            raise utils.io_error_message('Storage', name, 'outlet')

        # Set an upper bound for the storage capacity if nothing is specified to
        # make sure that storage components always have a capacity variable!
        if not capacity and not capacity_min and not capacity_max:
            capacity_max = 1e9

        Component.__init__(self, ensys=ensys, name=name,
                           inlet=inlet, outlet=outlet,
                           basic_variable=basic_variable,
                           has_existence_binary_var=has_existence_binary_var,
                           time_series_data=time_series_data,
                           scalar_params=scalar_params,
                           additional_vars=additional_vars,
                           user_expressions=user_expressions,
                           capacity=capacity,
                           capacity_min=capacity_min,
                           capacity_max=capacity_max,
                           capacity_per_module=capacity_per_module,
                           maximal_module_number=maximal_module_number,
                           capex_per_capacity=capex_per_capacity,
                           capex_if_exist=capex_if_exist,
                           opex_per_capacity=opex_per_capacity,
                           opex_if_exist=opex_if_exist,
                           opex_operation=opex_operation
                           )

        # Check and set additional input arguments
        self.charge_rate = utils.check_and_set_positive_number(
            charge_rate, 'charge_rate')
        self.discharge_rate = utils.check_and_set_positive_number(
            discharge_rate, 'discharge_rate')
        self.self_discharge = utils.check_and_set_range_zero_one(
            self_discharge, 'self_discharge')
        self.charge_efficiency = utils.check_and_set_range_zero_one(
            charge_efficiency, 'charge_efficiency')
        self.discharge_efficiency = utils.check_and_set_range_zero_one(
            discharge_efficiency, 'discharge_efficiency')
        self.soc_min = utils.check_and_set_range_zero_one(soc_min, 'soc_min')
        self.soc_max = utils.check_and_set_range_zero_one(soc_max, 'soc_max')
        self.soc_initial = utils.check_and_set_range_zero_one(
            soc_initial, 'soc_initial') if soc_initial is not None else None

        self.use_inter_period_formulation = utils.check_and_set_bool(
            use_inter_period_formulation, 'use_inter_period_formulation')
        self.precise_inter_period_modeling = utils.check_and_set_bool(
            precise_inter_period_modeling, 'precise_inter_period_modeling')

        # Store the names of the charging and discharging variables
        self.charge_variable = self.inlet[0].var_name
        self.discharge_variable = self.outlet[0].var_name

        # Create a state of charge (SOC) variable and if the inter-period
        # formulation is selected create an additional inter-period SOC variable
        # and SOC_MIN and SOC_MAX variables if precise modeling is requested.
        if not self.use_inter_period_formulation:
            self.add_var(utils.SOC, has_time_set=False,
                         alternative_set='intra_period_time_set')  # NonNegReal
        # use inter-period formulation:
        else:
            self.add_var(utils.SOC, domain='Reals', has_time_set=False,
                         alternative_set='intra_period_time_set')  # Real
            self.add_var(utils.SOC_INTER, has_time_set=False,
                         alternative_set='inter_period_time_set')
            if not self.precise_inter_period_modeling:
                self.add_var(utils.SOC_MAX, has_time_set=False,
                             alternative_set='typical_periods_set',
                             domain='Reals')
                self.add_var(utils.SOC_MIN, has_time_set=False,
                             alternative_set='typical_periods_set',
                             domain='Reals')

        # Last step: Add the component to the EnergySystem instance
        self.add_to_energy_system(ensys, name)

    def __repr__(self):
        return '<Storage: "%s">' % self.name

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
        self.con_cap_modular()
        self.con_modular_sym_break()
        self.con_couple_existence_and_modular()

        # Time-dependent constraints :
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.con_operation_limit(model)
        self.con_soc_balance(ensys, model)
        self.con_charge_rate(ensys, model)
        self.con_discharge_rate(ensys, model)
        self.con_cyclic_condition(ensys, model)
        self.con_soc_initial(ensys, model)
        self.con_soc_intra_period_start(ensys, model)  # only if inter-period f.
        self.con_soc_inter_period_balance(ensys, model)  # dito
        self.con_soc_bounds_without_inter_period_formulation(ensys, model)
        self.con_soc_bounds_with_inter_period_formulation_simple(ensys, model)
        self.con_soc_bounds_with_inter_period_formulation_precise(ensys, model)

    # **************************************************************************
    #    Time-dependent constraints
    # **************************************************************************
    def con_operation_limit(self, model):
        """
        The state of charge (SOC) of a storage component is limit by its nominal
        capacity in all time steps. E.g.: |br|
        ``SOC[p, t] <= CAP``

        *Method is not intended for public access!*
        """
        cap = self.variables[utils.CAP]['pyomo']
        soc = self.variables[utils.SOC]['pyomo']

        def con_operation_limit(m, p, t):
            return soc[p, t] <= cap
        setattr(self.block, 'con_operation_limit', pyomo.Constraint(
            model.time_set, rule=con_operation_limit))

    def con_soc_balance(self, ensys, model):
        """
        Constraint that connects the state of charge (SOC) of each time step
        with the charging and discharging events. The change in the SOC between
        two points in time has to match the values of charging and discharging
        and the self-discharge of the storage (explicit Euler formulation).
        Note: The SOC is not necessarily a value between 0 and 1 here. E.g.,
        |br| ``SOC[p, t+1] <= SOC[p, t]*(1-self_dischar)**dt
        + Q_IN[p, t]*eta_char - Q_OUT[p, t]/eta_dischar``

        *Method is not intended for public access!*
        """
        charge = self.variables[self.charge_variable]['pyomo']
        discharge = self.variables[self.discharge_variable]['pyomo']
        soc = self.variables[utils.SOC]['pyomo']
        dt = ensys.hours_per_time_step

        def con_soc_balance(m, p, t):
            return soc[p, t+1] == soc[p, t] * (1-self.self_discharge)**dt \
                   + charge[p, t] * self.charge_efficiency \
                   - discharge[p, t] / self.discharge_efficiency
        setattr(self.block, 'con_soc_balance', pyomo.Constraint(
            model.time_set, rule=con_soc_balance))

    def con_charge_rate(self, ensys, model):
        """
        Constraint to limit the value of the charge variable by applying a
        maximal charge rate. E.g.: |br|
        ``Q_IN[p, t] <= CAP * charge_rate * dt``

        *Method is not intended for public access!*
        """
        cap = self.variables[utils.CAP]['pyomo']
        charge = self.variables[self.charge_variable]['pyomo']
        dt = ensys.hours_per_time_step

        def con_charge_rate(m, p, t):
            return charge[p, t] <= cap * dt * self.charge_rate
        setattr(self.block, 'con_charge_rate', pyomo.Constraint(
            model.time_set, rule=con_charge_rate))

    def con_discharge_rate(self, ensys, model):
        """
        Constraint to limit the value of the discharge variable by applying a
        maximal discharge rate. E.g.: |br|
        ``Q_OUT[p, t] <= CAP * discharge_rate * dt``

        *Method is not intended for public access!*
        """
        cap = self.variables[utils.CAP]['pyomo']
        discharge = self.variables[self.discharge_variable]['pyomo']
        dt = ensys.hours_per_time_step

        def con_discharge_rate(m, p, t):
            return discharge[p, t] <= cap * dt * self.discharge_rate
        setattr(self.block, 'con_discharge_rate', pyomo.Constraint(
            model.time_set, rule=con_discharge_rate))

    def con_cyclic_condition(self, ensys, model):
        """
        Constraint to enforce that the SOC in the last time step of a period
        (after charging and discharging events) equals the SOC at the beginning
        of the same period.
        In case the inter-period formulation is activated, this constraint
        demands that the cycle condition is also fulfilled for the full time
        scale problem => SOC in global first time step (e.g., SOC[t=1]) equals
        SOC in global last time step (e.g., SOC[t=8760]).

        *Method is not intended for public access!*
        """
        if self.use_inter_period_formulation and ensys.is_data_clustered:
            soc_inter = self.variables[utils.SOC_INTER]['pyomo']

            def con_cyclic_condition_inter(m):
                last_idx = model.inter_period_time_set.last()
                return soc_inter[0] == soc_inter[last_idx]
            setattr(self.block, 'con_cyclic_condition_inter', pyomo.Constraint(
                rule=con_cyclic_condition_inter))
        else:
            # Use the formulation without inter-period time steps. This version
            # is computationally less challenging. All periods represent
            # independent entities. Energy cannot be transferred between periods
            # Only one "typical period" exists if data is not clustered [0]
            soc = self.variables[utils.SOC]['pyomo']

            def con_cyclic_condition(m, p):
                last_t_idx = model.intra_period_time_set.last()[1]
                return soc[p, 0] == soc[p, last_t_idx]
            setattr(self.block, 'con_cyclic_condition', pyomo.Constraint(
                model.typical_periods_set, rule=con_cyclic_condition))

    def con_soc_initial(self, ensys, model):
        """
        Constraint that sets a value for the relative state of charge in the
        first time step of the optimization problem. The same initial SOC value
        is applied to all periods if the model has multiple periods. Otherwise,
        the initial value is only specified for very beginning of the time set.
        E.g.: |br|
        ``SOC[p, 0] == CAP * soc_initial`` or |br|
        ``SOC_INTER[0] == CAP * soc_initial``

        *Method is not intended for public access!*
        """
        if self.soc_initial is not None:
            cap = self.variables[utils.CAP]['pyomo']

            if self.use_inter_period_formulation and ensys.is_data_clustered:
                soc_inter = self.variables[utils.SOC_INTER]['pyomo']

                def con_soc_inter_initial(m):
                    return soc_inter[0] == cap * self.soc_initial
                setattr(self.block, 'con_soc_inter_initial', pyomo.Constraint(
                    rule=con_soc_inter_initial))
            else:
                # Use the formulation without inter-period time steps. This
                # is computationally less challenging. All periods represent
                # independent entities. Energy can't be transferred between them
                # Only one "typical period" exists if data is not clustered [0]
                soc = self.variables[utils.SOC]['pyomo']

                def con_soc_initial(m, p):
                    return soc[p, 0] == cap * self.soc_initial
                setattr(self.block, 'con_soc_initial', pyomo.Constraint(
                    model.typical_periods_set, rule=con_soc_initial))

    def con_soc_intra_period_start(self, ensys, model):
        """
        The state of charge consists of two parts (Intra and Inter), if the
        intra-period model formulation ('use_inter_period_formulation'=True) is
        selected. This constraint sets the intra-period part (SOC) to zero at
        the beginning of each period. E.g.: |br|
        ``SOC[p, 0] == 0``

        *Method is not intended for public access!*
        """
        if self.use_inter_period_formulation and ensys.is_data_clustered:
            soc = self.variables[utils.SOC]['pyomo']

            def con_soc_intra_period_start(m, p):
                return soc[p, 0] == 0
            setattr(self.block, 'con_soc_intra_period_start', pyomo.Constraint(
                model.typical_periods_set, rule=con_soc_intra_period_start))

    def con_soc_inter_period_balance(self, ensys, model):
        """
        Constraint to calculate the inter-period state of charge (SOC_INTER) of
        the next period from the previous one and the value of the SOC (Intra)
        in the last time step of the related typical period. The constraint
        also accounts for self-discharge losses associated with the variable
        SOC_INTER (=> this can result in steps in the overall SOC profile at the
        boundary of 2 periods). E.g.: |br|
        ``SOC_INTER[p+1] == SOC_INTER[p]*(1-self_dischar)**(
        time_steps_per_period*dt) + SOC[typ_p, last_t]``

        *Method is not intended for public access!*
        """
        if self.use_inter_period_formulation and ensys.is_data_clustered:
            soc = self.variables[utils.SOC]['pyomo']
            soc_inter = self.variables[utils.SOC_INTER]['pyomo']
            dt = ensys.hours_per_time_step

            def con_soc_inter_period_balance(m, p):
                typ_period = ensys.periods_order[p]
                last_ts_idx = model.intra_period_time_set.last()[1]
                return soc_inter[p + 1] == soc_inter[p] * (
                        1 - self.self_discharge)**(
                        ensys.number_of_time_steps_per_period * dt) + soc[
                    typ_period, last_ts_idx]

            setattr(self.block, 'con_soc_inter_period_balance',
                    pyomo.Constraint(ensys.periods,
                                     rule=con_soc_inter_period_balance))

    def con_soc_bounds_without_inter_period_formulation(self, ensys, model):
        """
        Two constraints that specify bounds for the state of charge variable
        (minimal and maximal values) in case the inter-period formulation is NOT
        selected, or the data is NOT clustered!  E.g.: |br|
        ``SOC[p, t] >= CAP * soc_min`` and |br|
        ``SOC[p, t] <= CAP * soc_max``

        *Method is not intended for public access!*
        """
        if not self.use_inter_period_formulation or not ensys.is_data_clustered:
            soc = self.variables[utils.SOC]['pyomo']
            cap = self.variables[utils.CAP]['pyomo']

            # Always enforce a minimal value for the SOC (CAP*soc_min) with a
            # constraint, because it is possible that the domain of the SOC
            # variable is 'Reals' (if the storage component is constructed with
            # 'use_inter_period_formulation'=True, but the time series
            # aggregation was not performed before optimization).
            def con_minimal_soc(m, p, t):
                return soc[p, t] >= cap * self.soc_min
            setattr(self.block, 'con_minimal_soc', pyomo.Constraint(
                model.intra_period_time_set, rule=con_minimal_soc))

            # Only built the constraint if the value for soc_max is less than 1.
            # Otherwise it is already considered by "con_operation_limit".
            if self.soc_max < 1:
                def con_maximal_soc(m, p, t):
                    return soc[p, t] <= cap * self.soc_max
                setattr(self.block, 'con_maximal_soc', pyomo.Constraint(
                    model.intra_period_time_set, rule=con_maximal_soc))

    def con_soc_bounds_with_inter_period_formulation_simple(self, ensys, model):
        """
        Four constraints that define the bounds for the state of charge variable
        (minimal and maximal values) in a simplified way in case the inter-
        period formulation is requested and the data is clustered. The error is
        relatively small in comparison to the precise method if the specified
        value for the 'self_discharge' is not too high. However, this version
        requires a reasonable smaller number of constraints but also some
        additional variables. |br| See Eq. B1 and B2 in the Appendix of
        `Ref: DOI 10.1016/j.apenergy.2018.01.023
        <https://doi.org/10.1016/j.apenergy.2018.01.023>`_.

        *Method is not intended for public access!*
        """
        if self.use_inter_period_formulation and ensys.is_data_clustered \
                and not self.precise_inter_period_modeling:

            # Get variables:
            soc = self.variables[utils.SOC]['pyomo']
            cap = self.variables[utils.CAP]['pyomo']
            soc_max = self.variables[utils.SOC_MAX]['pyomo']
            soc_min = self.variables[utils.SOC_MIN]['pyomo']
            soc_inter = self.variables[utils.SOC_INTER]['pyomo']
            dt = ensys.hours_per_time_step

            # The variable "SOC_MAX" of a typical period is larger than all
            # occurring intra-period SOCs in the same period (except the last
            # one since it is already used in "SOC_INTER" of the next period)
            def con_soc_max_intra(m, p, t):
                return soc[p, t] <= soc_max[p]
            setattr(self.block, 'con_soc_max_intra', pyomo.Constraint(
                model.time_set, rule=con_soc_max_intra))

            # The variable "SOC_MIN" of a typical period is smaller than all
            # occurring intra-period SOCs in the same period (except the last
            # one since it is already used in "SOC_INTER" of the next period)
            def con_soc_min_intra(m, p, t):
                return soc[p, t] >= soc_min[p]
            setattr(self.block, 'con_soc_min_intra', pyomo.Constraint(
                model.time_set, rule=con_soc_min_intra))

            # The inter-period SOC at the beginning of a period plus the maximum
            # intra-period SOC of the associated typical period is less than
            # the available capacity times the maximal usable share (soc_max).
            def con_soc_max_bound_simple(m, p):
                typ_period = ensys.periods_order[p]
                return soc_inter[p] + soc_max[typ_period] <= cap * self.soc_max
            setattr(model, 'con_soc_max_bound_simple', pyomo.Constraint(
                ensys.periods, rule=con_soc_max_bound_simple))

            # The inter-period SOC at the beginning of a period minus the
            # maximal self discharge during that period plus the minimum
            # intra-period SOC of the associated typical period is larger than
            # the available capacity times the minimal required SOC (soc_min).
            def con_soc_min_bound_simple(m, p):
                typ_period = ensys.periods_order[p]
                return soc_inter[p] * (1 - self.self_discharge)**(
                        ensys.number_of_time_steps_per_period * dt) + soc_min[
                    typ_period] >= cap * self.soc_min
            setattr(model, 'con_soc_min_bound_simple', pyomo.Constraint(
                ensys.periods, rule=con_soc_min_bound_simple))

    def con_soc_bounds_with_inter_period_formulation_precise(self, ensys,
                                                             model):
        """
        Two constraints define the bounds for the state of charge variable
        (minimal and maximal values) in a precise way if the inter-period
        formulation is requested, and the data is clustered.
        This version requires two constraints for each time step of the full-
        scale problem and can be computationally expensive. Users might want to
        consider applying the simplified constraint formulation (keyword
        argument 'precise_inter_period_modeling'=False).
        |br| See Eq. 20 in `Ref: DOI 10.1016/j.apenergy.2018.01.023
        <https://doi.org/10.1016/j.apenergy.2018.01.023>`_.

        *Method is not intended for public access!*
        """
        if self.use_inter_period_formulation and ensys.is_data_clustered \
                and self.precise_inter_period_modeling:

            # Get variables:
            soc = self.variables[utils.SOC]['pyomo']
            cap = self.variables[utils.CAP]['pyomo']
            soc_inter = self.variables[utils.SOC_INTER]['pyomo']
            dt = ensys.hours_per_time_step

            # The inter-period SOC at the beginning of a period minus the self
            # discharge plus the intra-period SOC in each hour of that period
            # is less than the available capacity times the max. usable share.
            def con_soc_max_bound_precise(m, p, t):
                typ_period = ensys.periods_order[p]
                return soc_inter[p] * (1 - self.self_discharge)**(t * dt) + soc[
                    typ_period, t] <= cap * self.soc_max
            setattr(model, 'con_soc_max_bound_precise', pyomo.Constraint(
                ensys.periods, range(ensys.number_of_time_steps_per_period),
                rule=con_soc_max_bound_precise))

            # The inter-period SOC at the beginning of a period minus the self
            # discharge plus the intra-period SOC in each hour of that period
            # is larger than the available capacity times the min. required SOC.
            def con_soc_min_bound_precise(m, p, t):
                typ_period = ensys.periods_order[p]
                return soc_inter[p] * (1 - self.self_discharge)**(t * dt) + soc[
                    typ_period, t] >= cap * self.soc_min
            setattr(model, 'con_soc_min_bound_precise', pyomo.Constraint(
                ensys.periods, range(ensys.number_of_time_steps_per_period),
                rule=con_soc_min_bound_precise))

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
        comp_dict['charge_variable'] = self.charge_variable
        comp_dict['discharge_variable'] = self.discharge_variable
        comp_dict['soc_variable'] = utils.SOC
        comp_dict['soc_inter_variable'] = utils.SOC_INTER
        return comp_dict
