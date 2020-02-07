#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
** The Storage class **

* Last edited: 2020-01-01
* Created by: Stefan Bruche (TU Berlin)
"""
import pyomo.environ as pyomo
import pyomo.network as network
from aristopy.component import Component
from aristopy import utils


class Storage(Component):
    # Storage components store commodities and transfer them between time steps.
    def __init__(self, ensys, name, basic_variable, inlets=None, outlets=None,
                 existence_binary_var=None,
                 time_series_data_dict=None, time_series_weight_dict=None,
                 scalar_params_dict=None, additional_vars=None,
                 user_expressions=None,
                 capacity=None, capacity_min=None, capacity_max=None,
                 capacity_per_module=None, maximal_module_number=None,
                 capex_per_capacity=0, capex_if_exist=0,
                 opex_per_capacity=0, opex_if_exist=0,
                 opex_charging=0, opex_discharging=0,
                 charge_rate=1, discharge_rate=1, self_discharge=0,
                 charge_efficiency=1, discharge_efficiency=1,
                 soc_min=0, soc_max=1, soc_initial=None,
                 use_inter_period_formulation=False):  # TODO
        """
        Initialize a storage component.

        :param ensys:
        :param name:
        :param basic_variable:
        :param inlets:
        :param outlets:
        :param existence_binary_var:
        :param time_series_data_dict:
        :param time_series_weight_dict:
        :param scalar_params_dict:
        :param additional_vars:
        :param user_expressions:
        :param capacity:
        :param capacity_min:
        :param capacity_max:
        :param capacity_per_module:
        :param maximal_module_number:
        :param capex_per_capacity:
        :param capex_if_exist:
        :param opex_per_capacity:
        :param opex_if_exist:
        :param opex_charging:
        :param opex_discharging:
        :param charge_rate:
        :param discharge_rate:
        :param self_discharge:
        :param soc_min:
        :param soc_max:
        """

        Component.__init__(self, ensys, name, basic_variable,
                           inlets=inlets, outlets=outlets,
                           existence_binary_var=existence_binary_var,
                           time_series_data_dict=time_series_data_dict,
                           time_series_weight_dict=time_series_weight_dict,
                           scalar_params_dict=scalar_params_dict,
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
                           opex_if_exist=opex_if_exist
                           )

        self.modeling_class = 'Stor'

        # Check and set storage specific input arguments
        self.charge_rate = utils.set_if_positive(charge_rate)
        self.discharge_rate = utils.set_if_positive(discharge_rate)
        self.self_discharge = utils.set_if_between_zero_and_one(self_discharge)
        self.charge_efficiency = utils.set_if_between_zero_and_one(
            charge_efficiency)
        self.discharge_efficiency = utils.set_if_between_zero_and_one(
            discharge_efficiency)
        self.soc_min = utils.set_if_between_zero_and_one(soc_min)
        self.soc_max = utils.set_if_between_zero_and_one(soc_max)
        self.soc_initial = utils.set_if_between_zero_and_one(soc_initial) \
            if soc_initial is not None else None
        self.opex_charging = utils.set_if_positive(opex_charging)
        self.opex_discharging = utils.set_if_positive(opex_discharging)

        utils.is_boolean(use_inter_period_formulation)  # check input
        self.use_inter_period_formulation = use_inter_period_formulation

        # Declare create two variables. One for loading and one for unloading.
        self.charge_variable = self.basic_variable + '_CHARGE'
        self.discharge_variable = self.basic_variable + '_DISCHARGE'
        self._add_var(self.charge_variable)
        self._add_var(self.discharge_variable)

        # Create a SOC (state of charge) variable (set: 'inter_time_steps_set')
        self.soc_variable = self.basic_variable + '_SOC'
        self._add_var(self.soc_variable, has_time_set=False,
                      alternative_set='inter_time_steps_set')  # TODO: Hier wird eine NonNegativeReals Variable gebaut!

        # Last step: Add the component to the energy system model instance
        self.add_to_energy_system_model(ensys, name)

    def __repr__(self):
        return '<Storage: "%s">' % self.name

    def declare_component_ports(self):
        """
        Create all ports from dict 'ports_and_vars' and add variables to ports.
        """
        # Create ports and assign variables to ports
        for port_name, var_name in self.ports_and_vars.items():
            # Declare port
            setattr(self.pyB, port_name, network.Port())
            # Add charge (inlet) and discharge (outlet) variables
            port = getattr(self.pyB, port_name)
            if port_name.startswith('inlet_'):
                port.add(getattr(self.pyB, self.charge_variable),
                         var_name, port.Extensive)
            elif port_name.startswith('outlet_'):
                port.add(getattr(self.pyB, self.discharge_variable),
                         var_name, port.Extensive)

    def declare_component_constraints(self, ensys, pyM):
        """
        Declare time independent and dependent constraints.

        :param ensys: EnergySystemModel instance representing the energy system
            in which the component should be added.
        :type ensys: EnergySystemModel class instance

        :param pyM: Pyomo ConcreteModel which stores the mathematical
            formulation of the energy system model.
        :type pyM: Pyomo ConcreteModel
        """
        # Time independent constraints:
        # -----------------------------
        self.con_couple_bi_ex_and_cap()
        self.con_cap_min()
        self.con_cap_modular()
        self.con_modular_sym_break()
        self.con_couple_existence_and_modular()

        # Time dependent constraints:
        # ---------------------------
        self.con_operation_limit(pyM)
        self.con_soc_balance(ensys, pyM)
        self.con_first_time_step_soc(ensys, pyM)
        self.con_cyclic_condition(ensys, pyM)
        self.con_charge_rate(ensys, pyM)
        self.con_discharge_rate(ensys, pyM)
        self.con_minimal_soc(ensys, pyM)
        self.con_maximal_soc(ensys, pyM)

    def get_objective_function_contribution(self, ensys, pyM):
        """ Get contribution to the objective function. """

        # Alias of the components' objective function dictionary
        obj = self.comp_obj_dict
        # Get general required variables:
        charge = self.variables[self.charge_variable]['pyomo']
        discharge = self.variables[self.discharge_variable]['pyomo']

        # ---------------
        #   C A P E X
        # ---------------
        # CAPEX depending on capacity
        if self.capex_per_capacity > 0:
            cap = self.variables[self.capacity_variable]['pyomo']
            obj['capex_capacity'] = -1 * self.capex_per_capacity * cap

        # CAPEX depending on existence of component
        if self.capex_if_exist > 0:
            bi_ex = self.variables[self.bi_ex]['pyomo']
            obj['capex_exist'] = -1 * self.capex_if_exist * bi_ex
        # ---------------
        #   O P E X
        # ---------------
        # OPEX depending on capacity
        if self.opex_per_capacity > 0:
            cap = self.variables[self.capacity_variable]['pyomo']
            obj['opex_capacity'] = -1 * ensys.pvf * self.opex_per_capacity * cap

        # OPEX depending on existence of storage unit
        if self.opex_if_exist > 0:
            bi_ex = self.variables[self.bi_ex]['pyomo']
            obj['opex_exist'] = -1 * ensys.pvf * self.opex_if_exist * bi_ex

        # OPEX for charging and discharging the storage
        if self.opex_charging > 0 or self.opex_discharging > 0:
            obj['opex_operation'] = -1 * ensys.pvf * sum(
                (self.opex_charging * charge[p, t] + self.opex_discharging
                 * discharge[p, t]) * ensys.period_occurrences[p]
                for p, t in pyM.time_set) / ensys.number_of_years

        return sum(obj.values())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #    A D D I T I O N A L   T I M E   D E P E N D E N T   C O N S .
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def con_operation_limit(self, pyM):
        """
        The operation of a storage component (state of charge) is limit by its
        nominal capacity E.g.: |br|
        ``Q_SOC[p, t] <= Q_CAP``
        """
        # Only required if component has a capacity variable
        if self.capacity_variable is not None:
            # Get variables:
            cap = self.variables[self.capacity_variable]['pyomo']
            soc = self.variables[self.soc_variable]['pyomo']

            def con_operation_limit(m, p, t):
                return soc[p, t] <= cap

            setattr(self.pyB, 'con_operation_limit', pyomo.Constraint(
                pyM.time_set, rule=con_operation_limit))

    def con_soc_balance(self, ensys, pyM):
        """
        Constraint that connects the state of charge (SOC) with the charge and
        discharge operation: the change in the state of charge between two
        points in time has to match the values of charging and discharging
        and the self-discharge of the storage (explicit Euler formulation).
        Note that the SOC is not necessarily a value between 0 and 1 here.
        """
        # Get variables:
        charge = self.variables[self.charge_variable]['pyomo']
        discharge = self.variables[self.discharge_variable]['pyomo']
        soc = self.variables[self.soc_variable]['pyomo']
        dt = ensys.hours_per_time_step

        def con_soc_balance(m, p, t):
            return soc[p, t+1] == soc[p, t] * (1-self.self_discharge)**dt \
                   + charge[p, t] * self.charge_efficiency \
                   - discharge[p, t] / self.discharge_efficiency

        setattr(self.pyB, 'con_soc_balance', pyomo.Constraint(
            pyM.time_set, rule=con_soc_balance))

    def con_cyclic_condition(self, ensys, pyM):
        """
        State of charge storage in last time step (after last charging and
        discharging events) equals SOC in first time step.
        TODO: Extend explanation!
        """
        # Get variables:
        soc = self.variables[self.soc_variable]['pyomo']

        if self.use_inter_period_formulation:
            pass  # TODO: Fill it!  ---> connect the interPeriodSOCs

        else:
            # Use the formulation without inter-period time steps. This version
            # is computationally less challenging. All periods represent
            # independent entities. Energy cannot be transferred between periods
            def con_cyclic_condition(m, p):
                soc_last_ts = pyM.inter_time_steps_set.last()[1]
                return soc[p, 0] == soc[p, soc_last_ts]

            if ensys.is_data_clustered:
                # Set cyclic condition for every typical period
                setattr(self.pyB, 'con_cyclic_condition', pyomo.Constraint(
                    ensys.typical_periods, rule=con_cyclic_condition))
            else:
                # Set cyclic condition for the only period with number "0"
                setattr(self.pyB, 'con_cyclic_condition', pyomo.Constraint(
                    [0], rule=con_cyclic_condition))

    def con_first_time_step_soc(self, ensys, pyM):
        """
        A value for the relative state of charge in the first time step of each
        period can be specified here. (same value for all periods)
        """
        if self.capacity_variable is not None and self.soc_initial is not None:
            # Get variables:
            soc = self.variables[self.soc_variable]['pyomo']
            cap = self.variables[self.capacity_variable]['pyomo']

            if self.use_inter_period_formulation:
                pass  # TODO: Fill it!  ---> set first interPeriodSOC

            else:
                # Use the formulation without inter-period time steps. This
                # is computationally less challenging. All periods represent
                # independent entities. Energy can't be transferred between them
                def con_first_time_step_soc(m, p):
                    return soc[p, 0] == cap * self.soc_initial

                if ensys.is_data_clustered:
                    # Set cyclic condition for every typical period
                    setattr(self.pyB, 'con_first_time_step_soc',
                            pyomo.Constraint(ensys.typical_periods,
                                             rule=con_first_time_step_soc))
                else:
                    # Set cyclic condition for the only period with number "0"
                    setattr(self.pyB, 'con_first_time_step_soc',
                            pyomo.Constraint([0], rule=con_first_time_step_soc))

    def con_charge_rate(self, ensys, pyM):
        """
        XXX
        """
        if self.capacity_variable is not None:
            charge = self.variables[self.charge_variable]['pyomo']
            cap = self.variables[self.capacity_variable]['pyomo']
            dt = ensys.hours_per_time_step

            def con_charge_rate(m, p, t):
                return charge[p, t] <= cap * dt * self.charge_rate

            setattr(self.pyB, 'con_charge_rate', pyomo.Constraint(
                pyM.time_set, rule=con_charge_rate))

    def con_discharge_rate(self, ensys, pyM):
        """
        XXX
        """
        if self.capacity_variable is not None:
            discharge = self.variables[self.discharge_variable]['pyomo']
            cap = self.variables[self.capacity_variable]['pyomo']
            dt = ensys.hours_per_time_step

            def con_discharge_rate(m, p, t):
                return discharge[p, t] <= cap * dt * self.discharge_rate

            setattr(self.pyB, 'con_discharge_rate', pyomo.Constraint(
                pyM.time_set, rule=con_discharge_rate))

    def con_minimal_soc(self, ensys, pyM):
        """
        XXX
        """
        if self.capacity_variable is not None:  # todo:  and self.soc_min > 0: ?
            # Get variables:
            soc = self.variables[self.soc_variable]['pyomo']
            cap = self.variables[self.capacity_variable]['pyomo']

            if self.use_inter_period_formulation:
                pass  # TODO: Fill it!  ---> precise or simplified version

            else:
                # Use the formulation without inter-period time steps.
                def con_minimal_soc(m, p, t):
                    return soc[p, t] >= cap * self.soc_min

                setattr(self.pyB, 'con_minimal_soc',
                        pyomo.Constraint(pyM.inter_time_steps_set,
                                         rule=con_minimal_soc))

    def con_maximal_soc(self, ensys, pyM):
        """
        XXX
        """
        if self.capacity_variable is not None:  # todo:  and self.soc_max < 1: ?
            # Get variables:
            soc = self.variables[self.soc_variable]['pyomo']
            cap = self.variables[self.capacity_variable]['pyomo']

            if self.use_inter_period_formulation:
                pass  # TODO: Fill it!  ---> precise or simplified version

            else:
                # Use the formulation without inter-period time steps.
                def con_maximal_soc(m, p, t):
                    return soc[p, t] <= cap * self.soc_max

                setattr(self.pyB, 'con_maximal_soc',
                        pyomo.Constraint(pyM.inter_time_steps_set,
                                         rule=con_maximal_soc))
