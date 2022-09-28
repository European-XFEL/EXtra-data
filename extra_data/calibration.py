
"""Interfaces to calibration constant data."""

from datetime import datetime, date, time, timezone
from enum import Enum, IntFlag
from functools import lru_cache
from os import getenv
from pathlib import Path
from weakref import WeakKeyDictionary
import re
import socket

import numpy as np
import h5py

from . import SourceData

try:
    from calibration_client import CalibrationClient
    from calibration_client.modules import Detector, PhysicalDetectorUnit, \
        Parameter, Calibration, CalibrationConstantVersion
except ImportError as e:
    raise RuntimeError(
        f'`calibration_client` not available, please install to enable '
        f'CalibrationData interface: {e}')


__all__ = [
    'BadPixels',
    'CalCatError',
    'CalibrationData',
    'AGIPD_CalibrationData',
    'LPD_CalibrationData',
    'DSSC_CalibrationData',
    'JUNGFRAU_CalibrationData',
    'PNCCD_CalibrationData',
    'EPIX100_CalibrationData',
]


class BadPixels(IntFlag):
    """The European XFEL Bad Pixel Encoding."""

    OFFSET_OUT_OF_THRESHOLD = 1 << 0
    NOISE_OUT_OF_THRESHOLD = 1 << 1
    OFFSET_NOISE_EVAL_ERROR = 1 << 2
    NO_DARK_DATA = 1 << 3
    CI_GAIN_OF_OF_THRESHOLD = 1 << 4
    CI_LINEAR_DEVIATION = 1 << 5
    CI_EVAL_ERROR = 1 << 6
    FF_GAIN_EVAL_ERROR = 1 << 7
    FF_GAIN_DEVIATION = 1 << 8
    FF_NO_ENTRIES = 1 << 9
    CI2_EVAL_ERROR = 1 << 10
    VALUE_IS_NAN = 1 << 11
    VALUE_OUT_OF_RANGE = 1 << 12
    GAIN_THRESHOLDING_ERROR = 1 << 13
    DATA_STD_IS_ZERO = 1 << 14
    ASIC_STD_BELOW_NOISE = 1 << 15
    INTERPOLATED = 1 << 16
    NOISY_ADC = 1 << 17
    OVERSCAN = 1 << 18
    NON_SENSITIVE = 1 << 19
    NON_LIN_RESPONSE_REGION = 1 << 20
    WRONG_GAIN_VALUE = 1 << 21
    NON_STANDARD_SIZE = 1 << 22


class CCVMetadata(dict):
    """Dictionary for CCV metadata.

    Identical to a regular dict, but with a custom pandas-based
    string representation to be easier to read.
    """

    def __str__(self):
        """Pretty-print CCV metadata using pandas."""

        import pandas as pd

        res = {pdu_idx: {calibration: ccv_data['ccv_name']
                         for calibration, ccv_data in pdu_data.items()}
               for pdu_idx, pdu_data in self.items()}

        return str(pd.DataFrame.from_dict(res, orient='index'))


class CalCatError(Exception):
    """CalCat API error."""

    def __init__(self, response):
        super().__init__(response['info'])


class ClientWrapper(type):
    """Metaclass to wrap each calibration_client exactly once."""

    _clients = WeakKeyDictionary()

    def __call__(cls, client):
        instance = cls._clients.get(client, None)

        if instance is None:
            instance = cls._clients[client] = type.__call__(cls, client)

        return instance


class CalCatApi(metaclass=ClientWrapper):
    """Internal calibration_client wrapper."""

    get_detector_keys = ['id', 'name', 'identifier', 'karabo_name',
                         'karabo_id_control', 'description']
    get_pdu_keys = ['id', 'physical_name', 'karabo_da', 'virtual_device_name',
                    'detector_type_id', 'detector_id', 'description']

    def __init__(self, client):
        self.client = client

    @classmethod
    def format_time(cls, dt):
        """Parse different ways to specify time to CalCat."""

        if isinstance(dt, datetime):
            return dt.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%Z')
        elif isinstance(dt, date):
            return cls.format_time(datetime.combine(dt, time()))

        return dt

    def format_cond(self, condition):
        """Encode operating condition to CalCat API format.

        Args:
            caldata (CalibrationData): Calibration data instance used to
                interface with database.

        Returns:
            (dict) Operating condition for use in CalCat API.
        """

        return {'parameters_conditions_attributes': [
            {'parameter_id': self.parameter_id(k), 'value': str(v)}
            for k, v in condition.items()]}

    @lru_cache()
    def detector(self, detector_name):
        """Detector metadata."""

        resp_detector = Detector.get_by_identifier(
            self.client, detector_name)

        if not resp_detector['success']:
            raise CalCatError(resp_detector)

        return {k: resp_detector['data'][k] for k in self.get_detector_keys}

    @lru_cache()
    def physical_detector_units(self, detector_id, snapshot_at):
        """Physical detector unit metadata."""

        resp_pdus = PhysicalDetectorUnit.get_all_by_detector(
            self.client, detector_id, self.format_time(snapshot_at))

        if not resp_pdus['success']:
            raise CalCatError(resp_pdus)

        return {int(pdu['karabo_da'][-2:]): {
                    k: pdu[k] for k in self.get_pdu_keys}
                for pdu in resp_pdus['data']}

    @lru_cache()
    def calibration_id(self, calibration_name):
        """ID for a calibration in CalCat."""

        resp_calibration = Calibration.get_by_name(
            self.client, calibration_name)

        if not resp_calibration['success']:
            raise CalCatError(resp_calibration)

        return resp_calibration['data']['id']

    @lru_cache()
    def parameter_id(self, param_name):
        """ID for an operating condition parameter in CalCat."""

        resp_parameter = Parameter.get_by_name(self.client, param_name)

        if not resp_parameter['success']:
            raise CalCatError(resp_parameter)

        return resp_parameter['data']['id']

    def closest_ccv_by_time_by_condition(
        self, detector_name, calibrations, condition,
        modules=None, event_at=None, snapshot_at=None, metadata=None
    ):
        """Query bulk CCV metadata from CalCat.

        This method uses the /get_closest_version_by_detector API
        to query matching CCVs for PDUs connected to a detector instance
        in one go. In particular, it automatically includes the PDU as
        an operating condition parameter to allow for a single global
        condition rather than PDU-specific ones.

        Args:
            detector_name (str): Detector instance name.
            calibrations (Iterable of str): Calibrations to query
                metadata for.
            condition (dict): Mapping of parameter name to value.
            modules (Collection of int or None): List of module numbers
                or None for all (default).
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid or None for now (default).
            snapshot_at (datetime, date, str or None): Time of database
                state to look at or None for now (default).
            metadata (dict or None): Mapping to fill for results or
                None for a new dictionary (default).

        Returns:
            (dict) Nested mapping of module number to calibrations to
                CCV metadata. Identical to passed metadata argument if
                passed.
        """

        event_at = self.format_time(event_at)
        snapshot_at = self.format_time(snapshot_at)

        # Map aggregator to module number.
        da_to_modno = {
            data['karabo_da']: modno for modno, data in
            self.physical_detector_units(
                self.detector(detector_name)['id'], snapshot_at).items()
            if not modules or modno in modules}

        if metadata is None:
            metadata = CCVMetadata()

        if not calibrations:
            # Make sure there are at least empty dictionaries for each
            # module.
            for modno in da_to_modno.values():
                metadata.setdefault(modno, dict())
            return metadata

        # Map calibration ID to calibratio name.
        cal_id_map = {self.calibration_id(calibration): calibration
                      for calibration in calibrations}
        calibration_ids = list(cal_id_map.keys())

        # The API call supports a single module or all modules, as the
        # performance increase is only minor in between. Hence, all
        # modules are queried if more than one is selected and filtered
        # afterwards, if necessary.
        karabo_da = next(iter(da_to_modno)) if len(da_to_modno) == 1 else '',

        resp_versions = CalibrationConstantVersion.get_closest_by_time_by_detector_conditions(  # noqa
            self.client, detector_name, calibration_ids,
            self.format_cond(condition),
            karabo_da=karabo_da,
            event_at=event_at, snapshot_at=snapshot_at)

        if not resp_versions['success']:
            raise CalCatError(resp_versions)

        for ccv in resp_versions['data']:
            try:
                modno = da_to_modno[ccv['physical_detector_unit']['karabo_da']]
            except KeyError:
                # Not included in our modules
                continue

            cc = ccv['calibration_constant']
            metadata.setdefault(
                modno, dict())[cal_id_map[cc['calibration_id']]] = dict(
                    cc_id=cc['id'],
                    cc_name=cc['name'],
                    condition_id=cc['condition_id'],
                    ccv_id=ccv['id'],
                    ccv_name=ccv['name'],
                    path=Path(ccv['path_to_file']) / ccv['file_name'],
                    dataset=ccv['data_set_name'],
                    begin_validity_at=ccv['begin_validity_at'],
                    end_validity_at=ccv['end_validity_at'],
                    raw_data_location=ccv['raw_data_location'],
                    start_idx=ccv['start_idx'],
                    end_idx=ccv['end_idx']
                )

        return metadata


class CalibrationData:
    """Calibration constants data for detectors.

    European XFEL uses a web app and database to store records about the
    characterization of detectors and the data necessary to their
    correction and analysis, collectively called CalCat. The default
    installation is available at https://in.xfel.eu/calibration.

    A detector is identified by a name (e.g. SPB_DET_AGIPD1M-1) and
    consists of one or more detector modules. The modules are a virtual
    concept and may be identified by their number (e.g. 3), the Karabo
    data aggregator in EuXFEL's DAQ system they're connected to
    (e.g. AGIPD05) or a virtual device name describing their relative
    location (e.g. Q3M2).

    A detector module is mapped to an actual physical detector unit
    (PDU), which may be changed in case of a physical replacement. When
    characterization data is inserted into the database, it is attached
    to the PDU currently mapped to a module and not the virtual module
    itself.

    Characterization data is organized by its type just called
    calibration (e.g. Offset or SlopesFF) and the operating condition it
    was taken in, which is a mapping of parameter keys to their values
    (e.g. Sensor bias voltage or integration time). Any unique
    combination of calibration (type) and operating condition is a
    calibration constant (CC). Any individual measurement of a CC is
    called a calibration constant version (CCV). There may be many CCVs
    for any given CC.

    Note that while a connection to CalCat is possible from anywhere,
    the actual calibration data referred to is only available on the
    European XFEL infrastructure.
    """

    calibrations = set()
    default_client = None

    def __init__(self, detector_name, modules=None, client=None, event_at=None,
                 snapshot_at=None):
        """Initialize a new CalibrationData object.

        Args:
            detector_name (str): Name of detector in CalCat.
            modules (Iterable of int, optional): Module numbers to
                query for or None for all available (default).
            client (CalibrationClient, optional): Client for CalCat
                communication, global one by default.
            event_at (datetime, date, str or None): Default time at which the
                CCVs should have been valid, now if omitted
            snapshot_at (datetime, date, str or None): Default time of
                database state to look at, now if omitted.
            **condition_params: Operating condition parameters defined
                on an instance level.
        """

        self.detector_name = detector_name
        self.modules = modules
        self.event_at = event_at
        self.snapshot_at = snapshot_at

        if client is None:
            client = self.__class__.default_client or \
                self.__class__.new_anonymous_client()

        self._api = CalCatApi(client)

    @staticmethod
    def new_anonymous_client():
        return CalibrationData.new_client(None, None, None, use_oauth2=False,
                                          base_url='http://max-exfl017:9876')

    @staticmethod
    def new_client(client_id, client_secret, user_email, installation='',
                   base_url='https://in.xfel.eu/{}calibration', **kwargs):
        """Create a new calibration-client object.

        The client object is saved as a class property and is
        automatically to any future CalibrationData objects created, if
        no other client is passed explicitly.

        Arguments:
            client_id (str): Client ID.
            client_secret (str): Client secret.
            user_email (str): LDAP user email.
            installation (str, optional): Prefix for CalCat
                installation, production system by default.
            base_url (str, optional): URL template for CalCat
                installation, public European XFEL by default.
            Any further keyword arguments are passed on to
            CalibrationClient.__init__().

        Returns:
            (CalibrationClient) CalCat client.
        """

        base_url = base_url.format(f'{installation}_' if installation else '')

        # Note this is not a classmethod and we're modifying
        # CalibrationData directly to use the same object across all
        # detector-specific implementations.
        CalibrationData.default_client = CalibrationClient(
            client_id=client_id,
            client_secret=client_secret,
            user_email=user_email,
            base_api_url=f'{base_url}/api/',
            token_url=f'{base_url}/oauth/token',
            refresh_url=f'{base_url}/oauth/token',
            auth_url=f'{base_url}/oauth/authorize',
            scope='',
            **kwargs
        )
        return CalibrationData.default_client

    @property
    def caldb_root(self):
        """Root directory for calibration constant data.

        Returns:
            (Path or None) Location of caldb store or
                None if not available.
        """

        if not hasattr(CalibrationData, '_caldb_root'):
            if getenv('SASE'):
                # ONC
                CalibrationData._caldb_root = Path('/common/cal/caldb_store')
            elif re.match(r'^max-(.+)\.desy\.de$', socket.getfqdn()):
                # Maxwell
                CalibrationData._caldb_root = Path(
                    '/gpfs/exfel/d/cal/caldb_store')
            else:
                # Probably unavailable
                CalibrationData._caldb_root = None

        return CalibrationData._caldb_root

    @property
    def client(self):
        return self._api.client

    @property
    def detector(self):
        return self._api.detector(self.detector_name)

    @property
    def physical_detector_units(self):
        return self._api.physical_detector_units(
            self.detector['id'], self.snapshot_at)

    @property
    def condition(self):
        return self._build_condition(self.parameters)

    def replace(self, **new_kwargs):
        """Create a new CalibrationData object with altered values."""

        keys = {
            'detector_name', 'modules', 'client', 'event_at', 'snapshot_at'
        } | {
            self._simplify_parameter_name(name)for name in self.parameters
        }

        kwargs = {key: getattr(self, key) for key in keys}
        kwargs.update(new_kwargs)

        return self.__class__(**kwargs)

    def metadata(self, calibrations=None, event_at=None, snapshot_at=None):
        """Query CCV metadata for calibrations, conditions and time.

        Args:
            calibrations (Iterable of str, optional): Calibrations to
                query metadata for, may be None to retrieve all.
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid, now or default value passed at
                initialization time if omitted.
            snapshot_at (datetime, date, str or None): Time of database
                state to look at, now or default value passed at
                initialization time if omitted.

        Returns:
            (CCVMetadata) CCV metadata result.
        """

        metadata = CCVMetadata()
        self._api.closest_ccv_by_time_by_condition(
            self.detector_name, calibrations or self.calibrations,
            self.condition, self.modules,
            event_at or self.event_at, snapshot_at or self.snapshot_at,
            metadata)

        return metadata

    def ndarray(self, module, calibration, metadata=None):
        """Load CCV data as ndarray.

        Args:
            module (int): Module number
            calibration (str): Calibration constant.
            metadata (CCVMetadata, optional): CCV metadata to load
                constant data for, may be None to query metadata.

        Returns:
            (ndarray): CCV data
        """

        if self.caldb_root is None:
            raise RuntimeError('calibration database store unavailable')

        if self.modules and module not in self.modules:
            raise ValueError('module not part of this calibration data')

        if metadata is None:
            metadata = self.metadata([calibration])

        row = metadata[module][calibration]

        with h5py.File(self.caldb_root / row['path'], 'r') as f:
            return np.asarray(f[row['dataset'] + '/data'])

    def ndarray_map(self, calibrations=None, metadata=None):
        """Load all CCV data in a nested map of ndarrays.

        Args:
            calibrations (Iterable of str, optional): Calibration constants
                or None for all available (default).
            metadata (CCVMetadata, optional): CCV metadata to load constant
                for or None to query metadata automatically (default).

        Returns:
            (dict of dict of ndarray): CCV data by module number and
                calibration constant name.
        """

        if self.caldb_root is None:
            raise RuntimeError('calibration database store unavailable')

        if metadata is None:
            metadata = self.metadata(calibrations)

        return {modno: {name: self.ndarray(modno, name, metadata)
                        for name in data.keys()}
                for modno, data in metadata.items()}

    def _build_condition(self, parameters):
        cond = dict()

        for db_name in parameters:
            value = getattr(self, self._simplify_parameter_name(db_name), None)

            if value is not None:
                cond[db_name] = value

        return cond

    @classmethod
    def _from_multimod_detector_data(cls, component_cls, data, detector,
                                     modules, client):
        if isinstance(detector, component_cls):
            detector_name = detector.detector_name
        elif detector is None:
            detector_name = component_cls._find_detector_name(data)
        elif isinstance(detector, str):
            detector_name = detector
        else:
            raise ValueError(f'detector may be an object of type '
                             f'{type(cls)}, a string or None')

        source_to_modno = dict(component_cls._source_matches(
            data, detector_name))
        detector_sources = [data[source] for source in source_to_modno.keys()]

        if modules is None:
            modules = sorted(source_to_modno.values())

        creation_date = cls._determine_data_creation_date(data)

        # Create new CalibrationData object.
        caldata = cls(detector_name, modules, client,
                       creation_date, creation_date)

        caldata.memory_cells = component_cls._get_memory_cell_count(
            detector_sources[0])
        caldata.pixels_x = component_cls.module_shape[1]
        caldata.pixels_y = component_cls.module_shape[0]

        return caldata, detector_sources

    @staticmethod
    def _simplify_parameter_name(name):
        """Convert parameter names to valid Python symbols."""

        return name.lower().replace(' ', '_')

    @staticmethod
    def _determine_data_creation_date(data):
        """Determine data creation date."""

        assert data.files, 'data contains no files'

        try:
            creation_date = data.files[0].metadata()['creationDate']
        except KeyError:
            from warnings import warn
            warn('Last file modification time used as creation date for old '
                 'DAQ file format may be unreliable')

            return datetime.fromtimestamp(
                Path(data.files[0].filename).lstat().st_mtime)
        else:
            if not data.is_single_run:
                from warnings import warn
                warn('Sample file used to determine creation date for multi '
                     'run data')

            return creation_date


class SplitConditionCalibrationData(CalibrationData):
    """Calibration data with dark and illuminated conditions.

    Some detectors of this kind distinguish between two different
    operating conditions depending on whether photons illuminate the
    detector or not, correspondingly called the illuminated and dark
    conditions. Typically the illuminated condition is a superset of the
    dark condition.

    Not all implementations for semiconductor detectors inherit from
    this type, but only those that make this distinction such as AGIPD
    and LPD.
    """

    dark_calibrations = set()
    illuminated_calibrations = set()
    dark_parameters = list()
    illuminated_parameters = list()

    @property
    def calibrations(self):
        """Compatibility with CalibrationData."""

        return self.dark_calibrations | self.illuminated_calibrations

    @property
    def parameters(self):
        """Compatibility with CalibrationData."""

        return self.dark_parameters + self.illuminated_parameters

    @property
    def condition(self):
        """Compatibility with CalibrationData."""

        cond = dict()
        cond.update(self.dark_condition)
        cond.update(self.illuminated_condition)

        return cond

    @property
    def dark_condition(self):
        return self._build_condition(self.dark_parameters)

    @property
    def illuminated_condition(self):
        return self._build_condition(self.illuminated_parameters)

    def metadata(self, calibrations=None, event_at=None, snapshot_at=None):
        """Query CCV metadata for calibrations, conditions and time.

        Args:
            calibrations (Iterable of str, optional): Calibrations to
                query metadata for, may be None to retrieve all.
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid, now or default value passed at
                initialization time if omitted.
            snapshot_at (datetime, date, str or None): Time of database
                state to look at, now or default value passed at
                initialization time if omitted.

        Returns:
            (CCVMetadata) CCV metadata result.
        """

        if calibrations is None:
            calibrations = (
                self.dark_calibrations | self.illuminated_calibrations)

        metadata = CCVMetadata()

        dark_calibrations = self.dark_calibrations & set(calibrations)
        if dark_calibrations:
            self._api.closest_ccv_by_time_by_condition(
                self.detector_name, dark_calibrations,
                self.dark_condition, self.modules,
                event_at or self.event_at, snapshot_at or self.snapshot_at,
                metadata)

        illum_calibrations = self.illuminated_calibrations & set(calibrations)
        if illum_calibrations:
            self._api.closest_ccv_by_time_by_condition(
                self.detector_name, illum_calibrations,
                self.illuminated_condition, self.modules,
                event_at or self.event_at, snapshot_at or self.snapshot_at,
                metadata)

        return metadata


class AGIPD_CalibrationData(SplitConditionCalibrationData):
    """Calibration data for the AGIPD detector."""

    dark_calibrations = {'Offset', 'Noise', 'ThresholdsDark', 'BadPixelsDark',
                         'BadPixelsPC', 'SlopesPC'}
    illuminated_calibrations = {'BadPixelsFF', 'SlopesFF'}

    dark_parameters = ['Sensor Bias Voltage', 'Pixels X', 'Pixels Y',
                       'Memory cells', 'Acquisition rate', 'Gain setting',
                       'Gain mode', 'Integration time']
    illuminated_parameters = dark_parameters + ['Source energy']

    def __init__(self, detector_name, modules=None, client=None,
                 event_at=None, snapshot_at=None,
                 sensor_bias_voltage=300.0, memory_cells=202, pixels_x=512,
                 pixels_y=128, acquisition_rate=4.5, gain_setting=None,
                 gain_mode=None, integration_time=12, source_energy=9.2):
        super().__init__(detector_name, modules, client, event_at, snapshot_at)

        self.sensor_bias_voltage = sensor_bias_voltage
        self.memory_cells = memory_cells
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.acquisition_rate = acquisition_rate
        self.gain_setting = gain_setting
        self.gain_mode = gain_mode
        self.integration_time = integration_time
        self.source_energy = source_energy

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None,
                  control_source=None):
        """Initialize a new AGIPD_CalibrationData object based on data.

        The calibration constants validity time `event_at` is chosen
        based on the run creation date, while the database time
        `snapshot_at` is chosen to be now.

        Args:
            data (extra_data.DataCollection): Data to create
                AGIPD_CalibrationData object for.
            detector (extra_data.components.AGIPD1M or str, optional):
                Detector component object or name, may be omitted if
                only a single detector instance is present.
            modules (Iterable of ints, optional): Module numbers to
                include, may be omitted for all modules found in data
                and/or detector object.
            client (CalibrationClient, optional): Client for CalCat
                communication, may be omitted to use the global object.
            control_source (str, optional): Data source for control
                data, may be omitted to query from CalCat.

        Returns:
            (AGIPD_CalibrationData) Initialized object.
        """

        from .components import AGIPD1M
        self, _ = cls._from_multimod_detector_data(
            AGIPD1M, data, detector, modules, client)

        self.pixels_x, self.pixels_y = self.pixels_y, self.pixels_x

        if control_source is None:
            control_source = '{domain}/MDL/FPGA_COMP'.format(
                domain=self.detector['karabo_id_control'])

        ctrl_data = data[control_source]

        self.acquisition_rate = \
            ctrl_data['bunchStructure.repetitionRate'].as_single_value()
        self.gain_setting = ctrl_data['gain'].as_single_value()
        self.gain_mode = ctrl_data['gainModeIndex'].as_single_value()
        self.integration_time = ctrl_data['integrationTime'].as_single_value()

        return self

    def _build_condition(self, parameters):
        cond = super()._build_condition(parameters)

        # Fix-up some database quirks.
        if int(cond.get('Gain mode', -1)) == 0:
            del cond['Gain mode']


        if int(cond.get('Integration time', -1)) == 12:
            del cond['Integration time']

        return cond

class LPD_CalibrationData(SplitConditionCalibrationData):
    """Calibration data for the LPD detector."""

    dark_calibrations = {'Offset', 'Noise', 'BadPixelsDark'}
    illuminated_calibrations = {'RelativeGain', 'GainAmpMap', 'FFMap',
                                'BadPixelsFF'}

    dark_parameters = ['Sensor Bias Voltage', 'Memory cells', 'Pixels X',
                       'Pixels Y', 'Feedback capacitor']
    illuminated_parameters = dark_parameters + ['Source Energy', 'category']

    def __init__(self, detector_name,
                 modules=None, client=None, event_at=None, snapshot_at=None,
                 sensor_bias_voltage=250.0, memory_cells=512,
                 pixels_x=256, pixels_y=256, feedback_capacitor=5.0,
                 source_energy=9.2, category=1):
        super().__init__(detector_name, modules, client, event_at, snapshot_at)

        self.sensor_bias_voltage = sensor_bias_voltage
        self.memory_cells = memory_cells
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.feedback_capacitor = feedback_capacitor
        self.source_energy = source_energy
        self.category = category

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None,
                  control_source=None):
        """Initialize a new LPD_CalibrationData object based on data.

        The calibration constants validity time `event_at` is chosen
        based on the run creation date, while the database time
        `snapshot_at` is chosen to be now.

        Args:
            data (extra_data.DataCollection): Data to create
                LPD_CalibrationData object for.
            detector (extra_data.components.LPD1M or str, optional):
                Detector component object or name, may be omitted if
                only a single detector instance is present.
            modules (Iterable of ints, optional): Module numbers to
                include, may be omitted for all modules found in data
                and/or detector object.
            client (CalibrationClient, optional): Client for CalCat
                communication, may be omitted to use the global object.
            **condition_params: Operating condition parameters defined
                on an instance level.

        Returns:
            (LPD_CalibrationData) Initialized object.
        """

        from .components import LPD1M
        self, _ = cls._from_multimod_detector_data(
            LPD1M, data, detector, modules, client)

        # TODO: LPD control device?

        return self


class DSSC_CalibrationData(CalibrationData):
    """Calibration data for the DSSC detetor."""

    calibrations = {'Offset', 'Noise'}
    parameters = ['Sensor Bias Voltage', 'Memory cells', 'Pixels X',
                  'Pixels Y', 'Pulse id checksum', 'Acquisition rate',
                  'Target gain', 'Encoded gain']

    def __init__(self, detector_name, modules=None, client=None,
                 event_at=None, snapshot_at=None,
                 sensor_bias_voltage=100.0, memory_cells=400, pixels_x=512,
                 pixels_y=128, pulse_id_checksum=None, acquisition_rate=None,
                 target_gain=None, encoded_gain=None):
        super().__init__(detector_name, modules, client, event_at, snapshot_at)

        self.sensor_bias_voltage = sensor_bias_voltage
        self.memory_cells = memory_cells
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.pulse_id_checksum = pulse_id_checksum
        self.acquisition_rate = acquisition_rate
        self.target_gain = target_gain
        self.encoded_gain = encoded_gain

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None,
                  control_source=None):
        """Initialize a new DSSC_CalibrationData object based on data.

        The calibration constants validity time `event_at` is chosen
        based on the run creation date, while the database time
        `snapshot_at` is chosen to be now.

        Args:
            data (extra_data.DataCollection): Data to create
                DSSC_CalibrationData object for.
            detector (extra_data.components.DSSC1M or str, optional):
                Detector component object or name, may be omitted if
                only a single detector instance is present.
            modules (Iterable of ints, optional): Module numbers to
                include, may be omitted for all modules found in data
                and/or detector object.
            client (CalibrationClient, optional): Client for CalCat
                communication, may be omitted to use the global object.
            control_source (str, optional): Data source for control
                data, may be omitted to query from CalCat. May
                contain a format field for the quadrant number.

        Returns:
            (DSSC_CalibrationData) Initialized object.
        """

        from .components import DSSC1M
        self, det_data = cls._from_multimod_detector_data(
            DSSC1M, data, detector, modules, client)

        if control_source is None:
            control_source = '{domain}/FPGA/PPT_Q{{}}'.format(
                domain=self.detector['karabo_id_control'])

        control_sources = data.all_sources \
            & {control_source.format(i) for i in range(1, 5)}

        if not control_sources:
            raise ValueError(f'no quadrant control source found in data for '
                             f'pattern {control_source}')

        # ctrl_data = data[next(iter(control_sources))]

        # Several of these parameters are not reproducible so far.
        # self.pulse_id_checksum = self._get_pulse_id_checksum(det_data[0])
        # self.acquisition_rate = self._get_acquisition_rate(data, ctrl_data)
        # self.encoded_gain = self._get_encoded_gain(data, ctrl_data)
        # self.target_gain = self._get_target_gain(data, ctrl_data)

        return self

    @staticmethod
    def _get_pulse_id_checksum(ctrl_data):
        pulse_id_data = ctrl_data['image.pulseId']
        nonempty_trains = pulse_id_data.data_counts(labelled=False).nonzero()

        if not nonempty_trains:
            raise ValueError('all trains in data are empty')

        pulse_ids = pulse_id_data[nonempty_trains[0]].ndarray().squeeze()

        from hashlib import blake2b
        from struct import unpack

        return unpack('d', blake2b(pulse_ids.data, digest_size=8).digest())[0]

    @staticmethod
    def _get_acquisition_rate(data, ctrl_data):
        cycle_length = data.get_run_value(
            ctrl_data.source, 'sequencer.cycleLength')
        return 4.5 * (22.0 / cycle_length)

    @staticmethod
    def _get_target_gain(data, ctrl_data):
        filename = Path(
            data.get_run_value(ctrl_data.source, 'epcRegisterFilePath')).stem
        m = re.match(r'.*_TG(?P<TG>\d+.?\d+)', filename)

        if not m:
            raise ValueError('malformed epcRegisterFilePath')

        return float(m.group('TG'))

    @staticmethod
    def _get_encoded_gain(data, ctrl_data):
        # The description of the paramters one can find in:
        # https://docs.xfel.eu/share/page/site/dssc/documentlibrary
        # Documents> DSSC - Documentation> DSSC - ASIC Documentation.
        csaFbCap = int(ctrl_data['gain.csaFbCap'].as_single_value())
        fcfEnCap = int(ctrl_data['gain.fcfEnCap'].as_single_value())
        csaResistor = int(ctrl_data['gain.csaResistor'].as_single_value())
        trimmed = int(data.get_run_value(
            ctrl_data.source, 'gain.irampFineTrm') == 'Various')

        return (csaFbCap << 0) + (fcfEnCap << 8) + (csaResistor << 16) \
            + (trimmed << 32)


class JUNGFRAU_CalibrationData(CalibrationData):
    """Calibration data for the JUNGFRAU detector."""

    calibrations = {'Offset10Hz', 'Noise10Hz', 'BadPixelsDark10Hz',
                    'RelativeGain10Hz', 'BadPixelsFF10Hz'}
    parameters = ['Sensor Bias Voltage', 'Memory Cells', 'Pixels X',
                  'Pixels Y', 'Integration Time', 'Sensor temperature',
                  'Gain Setting']

    class GainSetting(Enum):
        dynamicgain = 0
        dynamichg0 = 1

    def __init__(self, detector_name, modules=None, client=None,
                 event_at=None, snapshot_at=None,
                 sensor_bias_voltage=100.0, memory_cells=16, pixels_x=1024,
                 pixels_y=512, integration_time=10.0, sensor_temperature=291,
                 gain_setting=0):
        super().__init__(detector_name, modules, client, event_at, snapshot_at)

        self.sensor_bias_voltage = sensor_bias_voltage
        self.memory_cells = memory_cells
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.integration_time = integration_time
        self.sensor_temperature = sensor_temperature
        self.gain_setting = gain_setting

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None,
                  control_source=None):
        from extra_data.components import JUNGFRAU
        self, _ = cls._from_multimod_detector_data(
            JUNGFRAU, data, detector, modules, client)

        # Constants are saved for 16 memory cells, even though the data
        # contains only a single one.
        self.memory_cells = 16

        if control_source is None:
            control_source = '{domain}/DET/CONTROL'.format(
                domain=self.detector['karabo_id_control'])

        ctrl_data = data[control_source]

        # Voltage is a vector for some reason with only the first
        # element containing data.
        self.sensor_bias_voltage = \
            ctrl_data['vHighVoltage.'].as_single_value()[0]

        # Time is saved in us, but prefix is stripped in CalCat.
        self.integration_time = \
            ctrl_data['exposureTime'].as_single_value() * 1e6

        # Default case (dynamicgain)
        self.gain_setting = cls.GainSetting[
            data.get_run_value(control_source, 'settings')].value

        return self


class PNCCD_CalibrationData(CalibrationData):
    calibrations = {'OffsetCCD', 'BadPixelsDarkCCD', 'NoiseCCD',
                    'RelativeGainCCD', 'CTECCD'}
    parameters = ['Sensor Bias Voltage', 'Memory cells', 'Pixels X',
                  'Pixels Y', 'Integration Time', 'Sensor Temperature',
                  'Gain Setting']

    def __init__(self, detector_name, modules=None, client=None,
                 event_at=None, snapshot_at=None, memory_cells=1,
                 sensor_bias_voltage=270.0, pixels_x=1024, pixels_y=1024,
                 integration_time=70, sensor_temperature=243.347,
                 gain_setting=64):
        # Ignore modules for this detector.
        super().__init__(detector_name, None, client, event_at, snapshot_at)

        self.sensor_bias_voltage = sensor_bias_voltage
        self.memory_cells = 1  # Ignore memory_cells for this detector
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.integration_time = integration_time
        self.sensor_temperature = sensor_temperature
        self.gain_setting = gain_setting

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None):
        if not isinstance(detector, SourceData):
            # Convert any non-SourceData detector argument
            if detector is None:
                sources = {source for source in data.instrument_sources
                           if source.endswith('PNCCD_FMT-0:output')}
            elif isinstance(detector, str):
                sources = {source for source in data.instrument_sources
                           if source.startswith(detector) and
                                source.endswith('PNCCD_FMT-0:output')}
            else:
                raise ValueError('detector may be SourceData, str or None')

            if len(sources) != 1:
                raise ValueError('No or multiple candidate sources for '
                                 'pnCCD detector found')

            detector = data[next(iter(sources))]

        # detector is always a SourceData object at this point.
        detector_name = detector.source[:detector.source.find('/')]
        det_data = detector

        creation_date = cls._determine_data_creation_date(data)

        sensor_bias_voltage = abs(
            data[f'{detector_name}/MDL/DAQ_MPOD', 'u0voltage']
            .as_single_value(atol=1))

        gain_setting = \
            data[f'{detector_name}/MDL/DAQ_GAIN', 'pNCCDGain'] \
            .as_single_value()

        # Notebook seem to use the top sensor temperature as reference,
        # bottom sensor is found in inputB.
        sensor_temperature = \
            data[f'{detector_name}/CTRL/TCTRL', 'inputA.krdg'] \
            .as_single_value(atol=1)

        # Integration time is not found in control data.

        return cls(detector_name, modules, client, creation_date,
                   creation_date, sensor_bias_voltage=sensor_bias_voltage,
                   gain_setting=gain_setting,
                   sensor_temperature=sensor_temperature)


class EPIX100_CalibrationData(CalibrationData):
    pass
