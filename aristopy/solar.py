#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#    S O L A R
# ==============================================================================
"""
* File name: solar.py
* Last edited: 2020-06-14
* Created by: Stefan Bruche (TU Berlin)

The solar classes SolarData, SolarThermalCollector, and PVSystem provide
functionality to model feed-in time series data for solar components (thermal
and electrical) at a certain location and with specific tilt and azimuth angles.

Please note: The solar classes require the availability of the Python module
*pvlib*. The module is not provided with the standard installation of aristopy.
If you want to use the solar classes, consider installing the module in your
current environment, e.g.: ::

>> pip install pvlib

For further information and an installation guide, users are referred to the
`pvlib documentation <https://pvlib-python.readthedocs.io/en/stable/>`_.
"""
import pandas as pd
try:
    import pvlib
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False


class SolarData:
    def __init__(self, ghi, dhi, latitude, longitude, altitude=0):
        """
        Class to provide solar input data for PV or solar thermal calculations.

        The main output is accessed via function "get_plane_of_array_irradiance"
        for the POA data with specified surface tilt and azimuth values and
        "get_irradiance_dataframe" for a pandas DataFrame consisting of GHI, DHI
        and DNI values at the specified location and time index.

        :param ghi: Pandas series (with datetime index and time zone) for global
            horizontal irradiation data at the respective location.
        :param dhi: Pandas series (with datetime index and time zone) for
            diffuse horizontal irradiation data at the respective location.
        :param latitude: Latitude value (float, int) for respective location.
        :param longitude: Longitude value (float, int) for respective location.
        :param altitude: Altitude value (float, int) for respective location.
        """
        if not HAS_PVLIB:
            raise ImportError(
                'Module "pvlib" not found. Please consider installing it in '
                'your current environment via "pip install pvlib" if you '
                'want to use the solar class "%s".' % self.__class__.__name__)

        self.solar_position = None  # init

        self.ghi = ghi
        self.dhi = dhi

        self._tz = 'UTC'  # default, overwritten by GHI input time series

        self.location = (latitude, longitude, altitude)

    @property
    def ghi(self):
        return self._ghi

    @ghi.setter
    def ghi(self, ghi):
        if isinstance(ghi, pd.Series):
            self._ghi = ghi
            self._data_index = ghi.index
            self._tz = ghi.index.tz
            # Update the solar position for the location and the time index
            # Not during first initialization --> location is inited later
            if hasattr(self, 'location'):
                self.solar_position = self.location.get_solarposition(
                    times=self._data_index)
        else:
            raise TypeError('Input for GHI needs to be a pandas Series')

    @property
    def data_index(self):
        return self._data_index

    @property
    def dhi(self):
        return self._dhi

    @dhi.setter
    def dhi(self, dhi):
        if isinstance(dhi, pd.Series):
            if not dhi.index.equals(self._data_index):
                raise ValueError('GHI and DHI need to have the same data index')
            elif dhi.index.tz != self._tz:
                raise ValueError('GHI and DHI need to have the same time zone')
            else:
                self._dhi = dhi
        else:
            raise TypeError('Input for DHI needs to be a pandas Series')

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        if isinstance(location, pvlib.location.Location):
            self._location = location
        elif isinstance(location, tuple) and len(location) == 3:
            self._location = pvlib.location.Location(
                latitude=location[0], longitude=location[1],
                altitude=location[2], tz=self._tz)
        # Update the solar position for the location and the time index
        self.solar_position = self._location.get_solarposition(
            times=self.data_index)

    def calculate_dni(self):
        # Calc. direct normal irradiation (DNI) from GHI, DHI and solar position
        # DNI may be unreasonably high or neg. for zenith angles close to 90°
        # (sunrise/sunset transitions). Function sets them to NaN => corr. to 0
        return pvlib.irradiance.dni(ghi=self.ghi, dhi=self.dhi,
                                    zenith=self.solar_position[
                                        'apparent_zenith']).fillna(0)

    def get_plane_of_array_irradiance(self, surface_tilt, surface_azimuth):
        """
        Calculate and return the plane of array irradiance (POA).

        :param surface_tilt: tilt of the PV modules (0=horizontal, 90=vertical)
        :param surface_azimuth: module azimuth angle (180=facing south)
        :return: pandas DataFrame with POA ('poa_global', ...)
        """
        return pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
            dni=self.calculate_dni(), ghi=self.ghi, dhi=self.dhi,
            solar_zenith=self.solar_position['apparent_zenith'],
            solar_azimuth=self.solar_position['azimuth'])

    def get_irradiance_dataframe(self):
        """
        Create and return a pandas DataFrame consisting of global (GHI) and
        diffuse horizontal (DHI) and direct normal irradiation (DNI).

        :return: pandas DataFrame with column names 'ghi', 'dhi', 'dni'
        """
        df = pd.DataFrame(index=self.ghi.index)
        df['ghi'] = self.ghi
        df['dhi'] = self.dhi
        df['dni'] = self.calculate_dni()
        return df


class SolarThermalCollector:
    def __init__(self, **kwargs):
        """
        Required input arguments (either while creating the class object or
        while calling function 'get_collector_heat_output'.

        * 'optical_efficiency': Opt. eff of the collector (float, int)
        * 'thermal_loss_parameter_1': Th. loss of the collector (float, int)
        * 'thermal_loss_parameter_2': Th. loss of the collector (float, int)
          |br| => See equation in: V.Quaschning, 'Regenerative Energiesysteme',
          10th edition, Hanser, 2019, p.131ff.
        * 'irradiance_data': Irradiance (POA) on collector array (pd.Series)
        * 't_ambient': Ambient temperature (float, int, pd.Series)
        * 't_collector_in': Collector inlet temperature (float, int, pd.Series)
        * 't_collector_out': Collector outlet temp. (float, int, pd.Series)
        """
        if not HAS_PVLIB:
            raise ImportError(
                'Module "pvlib" not found. Please consider installing it in '
                'your current environment via "pip install pvlib" if you '
                'want to use the solar class "%s".' % self.__class__.__name__)

        # initialize private helper variables with None
        self._all_private_vars = [
            '_optical_efficiency', '_thermal_loss_parameter_1',
            '_thermal_loss_parameter_2', '_t_ambient',
            '_t_collector_in', '_t_collector_out', '_irradiance_data']

        for var in self._all_private_vars:
            setattr(self, var, None)

        # Initialize the target values
        self.t_delta = None
        self.collector_efficiency = None
        self.collector_heat = None

        # Set the keyword arguments
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def t_collector_in(self):
        return self._t_collector_in

    @t_collector_in.setter
    def t_collector_in(self, value):
        if isinstance(value, (float, int, pd.Series)):  # multiple type options
            self._t_collector_in = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected numeric value or pandas Series.')

    @property
    def t_collector_out(self):
        return self._t_collector_out

    @t_collector_out.setter
    def t_collector_out(self, value):
        if isinstance(value, (float, int, pd.Series)):  # multiple type options
            self._t_collector_out = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected numeric value or pandas Series.')

    @property
    def irradiance_data(self):
        return self._irradiance_data

    @irradiance_data.setter
    def irradiance_data(self, value):
        if isinstance(value, pd.Series):
            self._irradiance_data = value
            self._trigger_calculation()
        else:
            raise ValueError('Irradiance data required as pandas Series')

    @property
    def optical_efficiency(self):
        return self._optical_efficiency

    @optical_efficiency.setter
    def optical_efficiency(self, value):
        if isinstance(value, (int, float)) and 0 < value <= 1:
            self._optical_efficiency = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected a value between 0 and 1 for opt. eff.')

    @property
    def thermal_loss_parameter_1(self):
        return self._thermal_loss_parameter_1

    @thermal_loss_parameter_1.setter
    def thermal_loss_parameter_1(self, value):
        if isinstance(value, (float, int)):
            self._thermal_loss_parameter_1 = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected a numeric value for the th. loss param.')

    @property
    def thermal_loss_parameter_2(self):
        return self._thermal_loss_parameter_2

    @thermal_loss_parameter_2.setter
    def thermal_loss_parameter_2(self, value):
        if isinstance(value, (float, int)):
            self._thermal_loss_parameter_2 = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected a numeric value for the th. loss param.')

    @property
    def t_ambient(self):
        return self._t_ambient

    @t_ambient.setter
    def t_ambient(self, value):
        if isinstance(value, (float, int, pd.Series)):  # multiple type options
            self._t_ambient = value
            self._trigger_calculation()
        else:
            raise ValueError('Expected numeric value or pandas Series.')

    def _has_all_inputs(self):
        # flag to check if collector is ready for calculation of heat output
        has_all_inputs = True
        for var in self._all_private_vars:
            if getattr(self, var) is None:
                has_all_inputs = False
        return has_all_inputs

    def _matching_indices(self):
        # Get all indices and store them in temporary list:
        indices = [getattr(self, var).index for var in self._all_private_vars
                   if isinstance(getattr(self, var), pd.Series)]
        # Check that all indices are the same:
        if len(indices) > 1:
            for idx in indices[1:]:  # starting with second element
                if not idx.equals(indices[0]):  # not equal first element?
                    return False
        # list of indices has only one element or no return of False
        return True

    def _trigger_calculation(self):
        # calculation of output is triggered whenever a parameter is changed
        # but only if all required input values are available and indices match.
        if self._has_all_inputs():
            if not self._matching_indices():
                raise ValueError('The provided data indices do not match!')
            else:
                t_mean = (self.t_collector_in + self.t_collector_out) / 2
                self.t_delta = t_mean - self.t_ambient
                self.collector_efficiency = \
                    self.optical_efficiency \
                    - self.thermal_loss_parameter_1 * self.t_delta \
                    / self.irradiance_data \
                    - self.thermal_loss_parameter_2 * self.t_delta**2 \
                    / self.irradiance_data
                # Replace negative values with zeros
                self.collector_efficiency[self.collector_efficiency < 0] = 0

                self.collector_heat = \
                    self.irradiance_data * self.collector_efficiency

    def get_collector_heat_output(self, **kwargs):
        """
        Required input arguments (either while creating the class object or
        while calling function 'get_collector_heat_output'.

        * 'optical_efficiency': Opt. eff of the collector (float, int)
        * 'thermal_loss_parameter_1': Th. loss of the collector (float, int)
        * 'thermal_loss_parameter_2': Th. loss of the collector (float, int)
          |br| => See equation in: V.Quaschning, 'Regenerative Energiesysteme',
          10th edition, Hanser, 2019, p.131ff.
        * 'irradiance_data': Irradiance (POA) on collector array (pd.Series)
        * 't_ambient': Ambient temperature (float, int, pd.Series)
        * 't_collector_in': Collector inlet temperature (float, int, pd.Series)
        * 't_collector_out': Collector outlet temp. (float, int, pd.Series)

        :return: pandas Series of provided collector heat output
        """
        # Set the keyword arguments.
        # The calculation is performed while setting input arguments.
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # Print a hint which required variables are still missing (if any)
        for var in self._all_private_vars:
            if getattr(self, var) is None:
                print('Missing input parameter "%s" detected' % var[1:])

        # Export the result
        return self.collector_heat


class PVSystem:
    def __init__(self, module, inverter):
        """
        PVSystem class holds a type of PV module and PV inverter.

        The main class function is "get_feedin", to get the power feedin for a
        PV plant with specified module, inverter, tilt and azimuth angle,
        location and weather data.

        :param module: Name of the module as in PVLib Database. to see the full
            database type e.g. "pvlib.pvsystem.retrieve_sam(name='cecmod')"
        :param inverter: Name of the inverter as in PVLib Database. to see the
            database type e.g. "pvlib.pvsystem.retrieve_sam(name='cecinverter')"
        """
        if not HAS_PVLIB:
            raise ImportError(
                'Module "pvlib" not found. Please consider installing it in '
                'your current environment via "pip install pvlib" if you '
                'want to use the solar class "%s".' % self.__class__.__name__)

        self.system = pvlib.pvsystem.PVSystem()  # create empty PVSystem

        self.module = module
        self.inverter = inverter

        # Initialization of default values and (private) attributes
        self.mode = 'ac'
        self._location = None
        self.mc = None  # ModelChain

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module):
        if module in pvlib.pvsystem.retrieve_sam(name='cecmod'):
            self._module_parameters = pvlib.pvsystem.retrieve_sam(
                name='cecmod')[module]
        elif module in pvlib.pvsystem.retrieve_sam(name='sandiamod'):
            self._module_parameters = pvlib.pvsystem.retrieve_sam(
                name='sandiamod')[module]
        else:
            raise ValueError('Module %s not found in the database.' % module)
        # If not raised, do ...
        self._module = module
        self.system.module_parameters = self._module_parameters

    @property
    def module_parameters(self):
        return self._module_parameters

    @property
    def inverter(self):
        return self._inverter

    @inverter.setter
    def inverter(self, inverter):
        if inverter in pvlib.pvsystem.retrieve_sam(name='cecinverter'):
            self._inverter_parameters = pvlib.pvsystem.retrieve_sam(
                name='cecinverter')[inverter]
        elif inverter in pvlib.pvsystem.retrieve_sam(name='sandiainverter'):
            self._inverter_parameters = pvlib.pvsystem.retrieve_sam(
                name='sandiainverter')[inverter]
        elif inverter in pvlib.pvsystem.retrieve_sam(name='adrinverter'):
            self._inverter_parameters = pvlib.pvsystem.retrieve_sam(
                name='adrinverter')[inverter]
        else:
            raise ValueError('Inverter %s not found in the database' % inverter)
        # If not raised, do ...
        self._inverter = inverter
        self.system.inverter_parameters = self._inverter_parameters

    @property
    def inverter_parameters(self):
        return self._inverter_parameters

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode.lower() == 'ac' or mode.lower() == 'dc':
            self._mode = mode.lower()
        else:
            raise ValueError("Mode must either be 'ac' or 'dc'")

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        if isinstance(location, pvlib.location.Location):
            self._location = location
        else:
            raise ValueError('Please provide a valid PVLib location instance!')

    @property
    def area(self):
        """
        Get area of the PV system in :math:`m^2`

        :return: PV System area
        """
        if 'Area' in self.system.module_parameters.index:
            area_per_module = self.system.module_parameters.Area
        elif 'A_c' in self.system.module_parameters.index:
            area_per_module = self.system.module_parameters.A_c
        else:
            return None
        return (area_per_module
                * self.system.strings_per_inverter
                * self.system.modules_per_string)

    @property
    def peak_power(self):
        """
        PV system peak power [W] can be limited by the inverter or the modules
        (minimum). If DC mode is selected the inverter is not considered.

        :return: Peak power of the PV System
        """
        # I and V at the MPP have different names depending on the database
        mod_par = self.system.module_parameters
        i_mpo = mod_par.Impo if hasattr(mod_par, 'Impo') else mod_par.I_mp_ref
        v_mpo = mod_par.Vmpo if hasattr(mod_par, 'Vmpo') else mod_par.V_mp_ref

        if self.mode == "ac":
            return min(
                i_mpo * v_mpo
                * self.system.strings_per_inverter
                * self.system.modules_per_string,
                self.system.inverter_parameters.Paco)
        else:  # self.mode == "dc":
            return (i_mpo * v_mpo
                    * self.system.strings_per_inverter
                    * self.system.modules_per_string)

    def set_location(self, latitude, longitude, altitude):
        self.location = pvlib.location.Location(
            latitude=latitude, longitude=longitude, altitude=altitude)

    def get_feedin(self, weather, surface_tilt, surface_azimuth,
                   scaling=None, mode='ac', **kwargs):
        """
        :param weather: requires pandas DataFrame with at least two out of three
            column names: 'ghi', 'dhi', 'dni'. Additionally users can specify
            column names 'wind_speed' and 'temp_air' (used in calc. of losses)
        :param surface_tilt: tilt of the PV modules (0=horizontal, 90=vertical)
        :param surface_azimuth: module azimuth angle (180=facing south)
        :param scaling:
            a) None=no feed-in scaling [W],
            b) 'area'=scale feed-in to area [W/m2],
            c) 'peak_power'=scale feed-in to nominal power [-]
        :param mode:
            a) 'ac': return AC feed-in (including inverter),
            b) 'dc': return DC feed-in (excluding inverter)
        :param kwargs: Examples for kwargs are 'albedo', 'modules_per_string',
            'strings_per_inverter', 'temperature_model_parameters', ...

        :return: pandas DataFrame with POA ('poa_global', ...)
        """
        if 'location' in kwargs.keys():
            self.location = kwargs['location']
        self.mode = mode

        self.system.surface_tilt = surface_tilt
        self.system.surface_azimuth = surface_azimuth

        # Construct a ModelChain instance
        try:
            self.mc = pvlib.modelchain.ModelChain(self.system, self.location)
        except ValueError:
            # If parameters can not be found in module_parameters, try:
            self.mc = pvlib.modelchain.ModelChain(self.system, self.location,
                                                  aoi_model='no_loss',
                                                  spectral_model='no_loss')
        # Set additional keyword arguments to PV System and ModelChain
        for key, val in kwargs.items():
            if key in dir(self.system):
                setattr(self.system, key, val)
            elif key in dir(self.mc):
                setattr(self.mc, key, val)

        # Check that ghi, dhi, dni are available or at least 2 out of 3
        has_ghi = 1 if 'ghi' in [i.lower() for i in weather] else 0
        has_dhi = 1 if 'dhi' in [i.lower() for i in weather] else 0
        has_dni = 1 if 'dni' in [i.lower() for i in weather] else 0
        if has_ghi + has_dhi + has_dni < 2:
            raise ValueError('Need at least 2 out of 3 from GHI, DHI, DNI '
                             'provided in weather Dataframe.')
        elif has_ghi + has_dhi + has_dni == 2:
            self.mc.complete_irradiance(weather=weather)  # Might need pytables?

        # run the model chain
        self.mc.run_model(weather=weather)

        # Get feed-in power in AC or DC, scale it to area or peak_power if
        # required and export result:
        feedin = self.mc.ac if self.mode == 'ac' else self.mc.dc.p_mp
        # Replace negative values with zeros
        feedin[feedin < 0] = 0

        if scaling is None:
            return feedin
        elif isinstance(scaling, str) and scaling.lower() == 'area':
            return feedin / float(self.area)
        elif isinstance(scaling, str) and scaling.lower() == 'peak_power':
            return feedin / float(self.peak_power)
        else:
            raise ValueError('Valid input for parameters "scaling" are None, '
                             '"area" or "peak_power"')


if __name__ == '__main__':

    # Set time index (incl. time zone) and sample data for global (GHI) and
    # diffuse horizontal irradiation (DHI) [W/m²]:
    idx = pd.date_range(start='2018-01-01 12:00', periods=12, freq='M', tz='UTC')
    ghi = pd.Series([0, 50, 150, 400, 600, 900, 1000, 600, 400, 150, 50, 0], idx)
    dhi = pd.Series([0, 30, 100, 200, 300, 400, 0, 100, 50, 10, 0, 0], idx)

    # SolarData:
    # ----------
    solar = SolarData(ghi=ghi, dhi=dhi, latitude=52.3822,
                      longitude=13.0622, altitude=81)

    # Solar thermal collector:
    # ------------------------
    poa = solar.get_plane_of_array_irradiance(
        surface_tilt=45, surface_azimuth=180)
    solar_coll = SolarThermalCollector(
        optical_efficiency=0.73, thermal_loss_parameter_1=1.7,
        thermal_loss_parameter_2=0.016, irradiance_data=poa['poa_global'],
        t_ambient=20, t_collector_in=20, t_collector_out=40)
    heat_out = solar_coll.get_collector_heat_output()

    # PV-System:
    # ----------
    # If available: append wind and temperature data to the DataFrame here
    weather = solar.get_irradiance_dataframe()
    # Create a PV System (consisting of a module and an inverter)
    pv_sys = PVSystem(module='Canadian_Solar_CS5P_220M___2009_',
                      inverter='ABB__MICRO_0_25_I_OUTD_US_208__208V_')
    # Calculate the feed-in of the PV system for specified weather conditions at
    # a site and a collector position.
    temp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass']
    feedin = pv_sys.get_feedin(weather=weather, location=solar.location,
                               surface_tilt=25, surface_azimuth=180,
                               temperature_model_parameters=temp)
