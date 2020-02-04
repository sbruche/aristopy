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
                 soc_min=0, soc_max=1):  # TODO
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
        self.opex_charging = utils.set_if_positive(opex_charging)
        self.opex_discharging = utils.set_if_positive(opex_discharging)

        # Declare create two variables. One for loading and one for unloading.
        self.charge_variable = self.basic_variable + '_CHARGE'
        self.discharge_variable = self.basic_variable + '_DISCHARGE'
        self._add_var(self.charge_variable)
        self._add_var(self.discharge_variable)

        # Create a SOC (state of charge) variable.
        self.soc_variable = self.basic_variable + '_SOC'
        self._add_var(self.soc_variable)

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
       # self.con_first_time_step(ensys, pyM)  #  TODO: remove it?
        self.con_last_time_step(ensys, pyM)
        self.con_charge_rate(ensys, pyM)
        self.con_discharge_rate(ensys, pyM)

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
                for p, t in pyM.time_set) / ensys.number_of_years  # * ensys.hours_per_time_step

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
            if t != ensys.time_steps_per_period[0]:  # not for first time step
                return soc[p, t] == soc[p, t-1] * (1-self.self_discharge)**dt \
                       + charge[p, t-1] - discharge[p, t-1]
            else:
                return pyomo.Constraint.Skip

        setattr(self.pyB, 'con_soc_balance', pyomo.Constraint(
            pyM.time_set, rule=con_soc_balance))

    # def con_first_time_step(self, ensys, pyM):
    #     # TODO: Remove it!
    #     """
    #     XXX --> not needed ? (half capacity in first time step of period)
    #     --> just relevant for reciding horizon optimization?!
    #     """
    #     if self.capacity_variable is not None:
    #         # Get variables:
    #         soc = self.variables[self.soc_variable]['pyomo']
    #         cap = self.variables[self.capacity_variable]['pyomo']
    #
    #         def con_first_time_step(m, p, t):
    #             if t == ensys.time_steps_per_period[0]:  # first time step
    #                 return soc[p, t] == 0.5 * cap
    #             else:
    #                 return pyomo.Constraint.Skip
    #
    #         setattr(self.pyB, 'con_first_time_step', pyomo.Constraint(
    #             pyM.time_set, rule=con_first_time_step))

    def con_last_time_step(self, ensys, pyM):
        """
        Boundary Condition: Level in storage in last time step (plus loading,
        minus unloading and loss) at least as full as in first time step.
        # Todo: Decide if at least or exactly?
        """
        # Get variables:
        charge = self.variables[self.charge_variable]['pyomo']
        discharge = self.variables[self.discharge_variable]['pyomo']
        soc = self.variables[self.soc_variable]['pyomo']
        dt = ensys.hours_per_time_step

        def con_last_time_step(m, p, t):
            if t == ensys.time_steps_per_period[-1]:  # last time step of period
                return soc[p, t] * (1 - self.self_discharge)**dt \
                       + charge[p, t] - discharge[p, t] >= soc[p, 0]
            else:
                return pyomo.Constraint.Skip

        setattr(self.pyB, 'con_last_time_step', pyomo.Constraint(
            pyM.time_set, rule=con_last_time_step))

    # TODO: Add constraint that prevents overloading in last time step
    #  (more than capacity).

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
