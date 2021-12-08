
"""Interfaces to calibration constant data."""

from datetime import datetime, date, time, timezone
from enum import IntFlag
from functools import lru_cache
from os import getenv
from pathlib import Path
import re
import socket

import numpy as np
import h5py

from . import SourceNameError, PropertyNameError, MultiRunError, NoDataError

from calibration_client import CalibrationClient
from calibration_client.modules import Detector, PhysicalDetectorUnit, \
    Parameter, Calibration, CalibrationConstantVersion


__all__ = [
    'BadPixels',
    'CalCatError',
    'AGIPD_CalibrationData',
    'LPD_CalibrationData',
    'DSSC_CalibrationData',
    'JUNGFRAU_CalibrationData',
    'PNCCD_CalibrationData',
    'EPIX_CalibrationData',
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


class OperatingCondition(dict):
    """Dictionary for operating condition parameters.

    A subclass of the regular Python dict to handle the semantics of
    operating conditions for calibration constants in the form of
    multiple layers to specify values and required/optional values.

    The multiple layers allow to overwrite default or earlier specified
    values at later times when actually using an operating condition.
    The actual parameters contained in the concretized operating
    conditions however (and thus in the object as dict itself) are only
    those actually added via set_required and set_optional, with
    addtional keys contained in one or more layers ignored.

    Any given parameter of an operating condition may either be required
    or optional. A required key is always present and accordingly must
    have a default value. An optional key will be present if it is
    specified in any of the layers or has a default value. If however
    its value is set to None explicitly in any layer, the default value
    is ignored.

    A parameter name has two representations, the canonical database
    form (e.g. Sensor Bias Voltage) and a key form, which must be a
    valid name for a Python symbol. By default, the key form is the all
    lower-case version of the database name with spaces replaced by
    underscores (e.g. sensor_bias_voltage). The key form is expected in
    the parameter layers.

    Example:

    .. code-block:: python

       cond = OperatingCondition(
           dict(key_a=1, key_b=2),
           dict(key_a=3, key_c=None)
       )

       cond.set_required('Key A', -1)
       cond.set_optional('Key B')
       cond.set_optional('Key C', 5)

       assert cond == {'Key A': 1, 'Key B': 2}
    """

    def __init__(self, *param_layers):
        """Initialize a new operating condition.

        Args:
            *param_layers (list of dict): Parameters layers with values
                with the lowest layer (passed first) taking precedence
                for any given parameter, which must be referred to by
                their key name.
        """

        self._param_layers = param_layers

    def set_required(self, db_name, default, key_name=None):
        """Set a required parameter.

        Args:
            db_name (str): Parameter name in database form.
            default (Any): Default value for this parameter if not found
                in any parameter layer.
            key_name (str, optional): Parameter name in key form, if
                omitted obtained from database form.

        Returns:
            None
        """

        assert default is not None, \
            'Required condition parameter must have a default'
        self.set_optional(db_name, default, key_name)

    def set_optional(self, db_name, default=None, key_name=None):
        """Set an optional parameter.

        Args:
            db_name (str): Parameter name in database form.
            default (Any, optional): Default value for this parameter if
                not found in any parameter layer. If omitted, the
                parameter is not added at all in this case.
            key_name (str, optional): Parameter name in key form, if
                omitted obtained from database form.

        Returns:
            None
        """

        if key_name is None:
            key_name = db_name.lower().replace(' ', '_')

        for layer in self._param_layers:
            if key_name in layer:
                if layer[key_name] is not None:
                    self[db_name] = layer[key_name]
                return
        else:
            if default is not None:
                self[db_name] = default

    def api_encode(self, caldata):
        """Encode operating condition to CalCat API format.

        Args:
            caldata (CalibrationData): Calibration data instance used to
                interface with database.

        Returns:
            (dict) Operating condition for use in CalCat API.
        """

        return {'parameters_conditions_attributes': [
            {'parameter_id': caldata.parameter_id(k), 'value': str(v)}
            for k, v in self.items()
        ]}


class CCVMetadata(dict):
    """Dictionary for CCV metadata."""

    def set_pdu_ccv(self, modno, calibration_name, entry):
        """Set a PDU CCV entry.

        Convenience method to handle insertion of keys if missing.
        """

        pdu_metadata = self.get(modno, dict())
        pdu_metadata[calibration_name] = entry
        self[modno] = pdu_metadata

    def __str__(self):
        """Pretty-print CCV metadata using pandas."""

        import pandas as pd

        res = {pdu_idx: {cal_name: ccv_data['ccv_name']
                         for cal_name, ccv_data in pdu_data.items()}
               for pdu_idx, pdu_data in self.items()}

        return str(pd.DataFrame.from_dict(res, orient='index'))


class CalCatError(RuntimeError):
    """CalCat API error."""

    def __init__(self, response):
        super().__init__(response['info'])


class CalibrationData:
    """Calibration data for detectors.

    European XFEL uses a web app and database to store records about the
    characterization of detectors and the data necessary to their
    correction, collectively called CalCat. The default installation is
    available at https://in.xfel.eu/calibration.

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

    get_detector_keys = ['id', 'name', 'identifier', 'karabo_name',
                         'karabo_id_control', 'description']
    get_pdu_keys = ['id', 'physical_name', 'karabo_da', 'virtual_device_name',
                    'detector_type_id', 'detector_id', 'description']

    calibrations = set()

    client = None

    def __init__(self, detector_name, modules=None, client=None,
                 **condition_params):
        """Initialize a new CalibrationData object.

        Args:
            detector_name (str): Name of detector in CalCat.
            modules (Iterable of int, optional): Module numbers to
                query for or None for all available (default).
            client (CalibrationClient, optional): Client for CalCat
                communication, global one by default.
            **condition_params: Operating condition parameters defined
                on an instance level.
        """

        if client is None:
            client = self.__class__.client

            if client is None:
                raise ValueError(f'need client passed by argument or prior '
                                 f'client set-up via {self.__class__.__name__}'
                                 f'.new_client')

        self.client = client
        self.detector_name = detector_name
        self.modules = modules

        self._condition_params = condition_params
        self._detector = None
        self._pdus = None

    @classmethod
    def new_client(cls, client_id, client_secret, user_email, installation='',
                   base_url='https://in.xfel.eu/{}calibration'):
        """Create a new calibration client object.

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

        Returns:
            (CalibrationClient) CalCat client.
        """

        base_url = base_url.format(f'{installation}_' if installation else '')
        cls.client = CalibrationClient(
            client_id=client_id,
            client_secret=client_secret,
            user_email=user_email,
            base_api_url=f'{base_url}/api/',
            token_url=f'{base_url}/oauth/token',
            refresh_url=f'{base_url}/oauth/token',
            auth_url=f'{base_url}/oauth/authorize',
            scope='',
        )
        return cls.client

    @property
    def caldb_root(self):
        """Root directory for calibration constant data.

        Returns:
            (Path or None) Location of caldb store or
                None if not available.
        """

        if not hasattr(self, '_caldb_root'):
            if getenv('SASE'):
                # ONC
                self._caldb_root = Path('/common/cal/caldb_store')
            elif re.match(r'^max-(.+)\.desy\.de$', socket.getfqdn()):
                # Maxwell
                self._caldb_root = Path('/gpfs/exfel/d/cal/caldb_store')
            else:
                # Probably unavailable
                self._caldb_root = None

        return self._caldb_root

    @property
    def detector(self):
        """Detector metadata."""

        if self._detector is None:
            resp_detector = Detector.get_by_identifier(
                self.client, self.detector_name)

            if not resp_detector['success']:
                raise CalCatError(resp_detector)

            self._detector = {k: resp_detector['data'][k]
                              for k in self.get_detector_keys}

        return self._detector

    @property
    def detector_id(self):
        """Detector ID in CalCat."""

        return self.detector['id']

    @property
    def physical_detector_units(self):
        """Physical detector unit metadata."""

        if self._pdus is None:
            resp_pdus = PhysicalDetectorUnit.get_all_by_detector(
                self.client, self.detector_id, '')

            if not resp_pdus['success']:
                raise CalCatError(resp_pdus)

            def is_selected_da(karabo_da):
                # Either all modules are selected or it must be in the
                # list of modules.
                return not self.modules or int(karabo_da[-2:]) in self.modules

            self._pdus = [{k: pdu[k] for k in self.get_pdu_keys}
                          for pdu in resp_pdus['data']
                          if is_selected_da(pdu['karabo_da'])]

        return self._pdus

    @property
    def pdu_by_aggregator(self):
        """PDU metadata by currently mapped data aggregator."""

        return self._map_pdu_by('karabo_da')

    @property
    def pdu_by_module_number(self):
        """PDU metadata by currently mapped module number."""

        return {int(pdu['karabo_da'][-2:]): pdu for pdu
                in self.physical_detector_units}

    @property
    def pdu_by_module_name(self):
        """PDU metadata by currently mapped virtual module name."""

        return self._map_pdu_by('virtual_device_name')

    @lru_cache()
    def calibration_id(self, calibration_name):
        """ID for a calibration in CalCat."""

        resp_calibration = Calibration.get_by_name(self.client,
                                                   calibration_name)

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

    def condition(self, **condition_params):
        """Concretize operating conditions.

        Args:
            **condition_params: Operating condition parameters defined
                on the level of the returned conditions.

        Returns:
            (OperatingCondition) Concretized operating conditions.
        """

        raise NotImplementedError(f'{self.__class__.__name__}.condition')

    def metadata(self, calibrations=None, condition=None,
                 event_at=None, snapshot_at=None, **condition_params):
        """Query CCV metadata for calibrations, conditions and time.

        Args:
            calibrations (Iterable of str, optional): Calibrations to
                query metadata for, may be None to retrieve all.
            condition (OperatingCondition, optional): Operating
                condition to query CCVs for, may be None to use
                .condition() with any additional keywords passed.
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid, now of omitted.
            snapshot_at (datetime, date, str or None): Time of database
                state to look at.
            **condition_params: Additional operating condition
                parameters defined only for this query, ignored if
                condition is passed.

        Returns:
            (CCVMetadata) CCV metadata result.
        """

        metadata = CCVMetadata()
        self._retrieve_constant_version(
            metadata,
            calibrations or self.calibrations,
            condition or self.condition(**condition_params),
            event_at, snapshot_at)

        return metadata

    def ndarray(self, *args, metadata=None, **kwargs):
        """Load CCV data as ndarray.

        The CCV metadata may either be passed directly through the
        metadata keyword argument or queried through the same call
        signature as .metadata().

        Args:
            metadata (CCVMetadata, optional): CCV metadata to load
                constant data for, may be None to query metadata.
            *args, **kwargs: If metadata is omitted, any additional
            positional and keyword arguments are passed on to
            .metadata().

        Returns:
            (dict of dict): Nested dictionary of PDU index and
                calibration name to CCV data.
        """

        if self.caldb_root is None:
            raise RuntimeError('calibration database store unavailable')

        if metadata is None:
            metadata = self.metadata(*args, **kwargs)

        data = dict()

        for mod_idx, calibrations in metadata.items():
            mod_data = dict()

            for name, row in calibrations.items():
                h5path = self.caldb_root / row['data_path']
                h5dset = row['dataset_name'] + '/data'

                with h5py.File(h5path, 'r') as f:
                    mod_data[name] = np.asarray(f[h5dset])

            data[mod_idx] = mod_data

        return data

    def _map_pdu_by(self, key):
        return {pdu[key]: pdu for pdu in self.physical_detector_units}

    @classmethod
    def _api_format_time(cls, dt):
        """Parse different ways to specify time to CalCat."""

        if isinstance(dt, datetime):
            return dt.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%z')
        elif isinstance(dt, date):
            return cls._api_format_time(datetime.combine(dt, time()))

        return dt

    def _query_constant_version(self, metadata, calibrations, condition,
                                event_at, snapshot_at):
        """Query CCV metadata from CalCat.

        This method should be used directly but rather .metadata().

        Args:
            metadata (CCVMetadata): Nested mapping of module index to
                calibration name to metdata to store the results in.
            calibrations (Iterable of str): Calibrations to query
                metadata for.
            condition (OperatingCondition): Operating condition to
                query CCVs for.
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid, now of omitted.
            snapshot_at (datetime, date, str or None): Time of database
                state to look at.

        Returns:
            None
        """

        da_to_modno = {data['karabo_da']: modno for modno, data in
                       self.pdu_by_module_number.items()}

        if not calibrations:
            # Make sure there are at least empty dictionaries for each
            # module.
            for modno in da_to_modno.values():
                metadata[modno] = metadata.get(modno, dict())
            return

        cal_id_to_name = {
            self.calibration_id(calibration_name): calibration_name
            for calibration_name in calibrations}
        calibration_ids = list(cal_id_to_name.keys())

        resp_versions = CalibrationConstantVersion.get_closest_by_time_by_detector_conditions(  # noqa
            self.client,
            self.detector_name,
            calibration_ids,
            condition.api_encode(self),
            karabo_da=next(iter(da_to_modno)) if len(da_to_modno) == 1 else '',
            event_at=self._api_format_time(event_at),
            snapshot_at=self._api_format_time(snapshot_at))

        if not resp_versions['success']:
            raise CalCatError(resp_versions)

        for ccv in resp_versions['data']:
            try:
                modno = da_to_modno[ccv['physical_detector_unit']['karabo_da']]
            except KeyError:
                # Not included in our modules
                continue

            cc = ccv['calibration_constant']

            # Manually build the result dict with expressive keys.
            entry = dict(
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
                end_idx=ccv['end_idx'])

            metadata.set_pdu_ccv(
                modno, cal_id_to_name[cc['calibration_id']], entry)


class DualConditionCalibrationData(CalibrationData):
    """Calibration data with dark and illuminated distinction.

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

    def condition(self, **condition_params):
        """Concretize operating condition.

        This method is not available in this subclass and any
        implementations inheriting from it. Use the corresponding
        methods .dark_condition() and .illuminated_condition() instead.

        Raises:
            NotImplementedError: Not available for this type
        """

        raise NotImplementedError(
            'detector with distinct {cls}.dark_condition and '
            '{cls}.illuminated_condition methods'.format(
                cls=self.__class__.__name__))

    def dark_condition(self, **condition_params):
        """Concretize dark operating condition.

        Args:
            **condition_params: Operating condition parameters defined
                on the level of the returned conditions.

        Returns:
            (OperatingCondition) Concretized operating conditions.
        """

        raise NotImplementedError(f'{self.__class__.__name__}.dark_condition')

    def illuminated_condition(self, **condition_params):
        """Concretize illuminated operating condition.

        Args:
            **condition_params: Operating condition parameters defined
                on the level of the returned conditions.

        Returns:
            (OperatingCondition) Concretized operating conditions.
        """

        raise NotImplementedError(
            f'{self.__class__.__name__}.illuminated_condition')

    def metadata(self, calibrations=None, condition=None,
                 event_at=None, snapshot_at=None, **condition_params):
        """Query CCV metadata for calibrations, conditions and time.

        Args:
            calibrations (Iterable of str, optional): Calibrations to
                query metadata for, may be None to retrieve all.
            condition (Sequence of OperatingCondition, optional):
                Dark and illuminated operating condition to query CCVs
                for, may be None to use .condition() with any additional
                keywords passed.
            event_at (datetime, date, str or None): Time at which the
                CCVs should have been valid, now of omitted.
            snapshot_at (datetime, date, str or None): Time of database
                state to look at.
            **condition_params: Additional operating condition
                parameters defined only for this query, ignored if
                condition is passed.

        Returns:
            (CCVMetadata) CCV metadata result.
        """

        try:
            dark_condition = condition[0]
            illum_condition = condition[1]
        except (TypeError, IndexError, AssertionError):
            if condition is not None:
                # If the condition was actually specified, inform the
                # caller that this method's interface is different.
                raise ValueError('condition must be 2-len sequence of dark '
                                 'and illuminated condition')

            dark_condition = self.dark_condition(**condition_params)
            illum_condition = self.illuminated_condition(**condition_params)

        if calibrations is None:
            calibrations = (
                self.dark_calibrations | self.illuminated_calibrations)

        metadata = CCVMetadata()

        dark_calibrations = self.dark_calibrations & set(calibrations)
        if dark_calibrations:
            self._retrieve_constant_version(
                metadata, dark_calibrations, dark_condition,
                event_at, snapshot_at)

        illum_calibrations = self.illuminated_calibrations & set(calibrations)
        if illum_calibrations:
            self._retrieve_constant_version(
                metadata, illum_calibrations, illum_condition,
                event_at, snapshot_at)

        return metadata


class AGIPD_CalibrationData(DualConditionCalibrationData):
    """Calibration data for the AGIPD detector.

    Dark operating condition:
        Required parameters:
            * Sensor Bias Voltage: 300.0
            * Pixels X: 512
            * Pixels Y: 128

        Optional parameters with default values:
            * Memory cells: 202
            * Acquisition rate: 4.5

        Optional parameters:
            * Gain setting
            * Gain mode
            * Integration time

    Illuminated operating condition:
        Required parameter:
            * Source energy: 9.2

        and all of the above for the dark condition.
    """

    dark_calibrations = {'Offset', 'Noise', 'ThresholdsDark', 'BadPixelsDark',
                         'BadPixelsPC', 'SlopesPC'}
    illuminated_calibrations = {'BadPixelsFF', 'SlopesFF'}

    def __init__(self, detector_name, modules=None, client=None,
                 **condition_params):
        super().__init__(detector_name, modules=modules, client=client,
                         **condition_params)

    @classmethod
    def from_data(cls, data, detector=None, modules=None, client=None,
                  ctrl_device_id=None, **condition_params):
        """Initialize a new AGIPD_CalibrationData object based on data.

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
            ctrl_device_id (str, optional): Karabo device ID for the
                control device, may be omitted to query from CalCat.
            **condition_params: Operating condition parameters defined
                on an instance level.

        Returns:
            (AGIPD_CalibrationData) CalibrationData
        """

        from .components import AGIPD1M

        if detector is None or isinstance(detector, str):
            detector = AGIPD1M(data, detector, modules)
        elif not isinstance(detector, AGIPD1M):
            raise ValueError('detector may be an object of type AGIPD1M, '
                             'a string or None')

        caldata = cls(detector.detector_name,
                      sorted(detector.modno_to_source.keys()),
                      client=client, memory_cells=detector.frames_per_train,
                      pixels_x=detector.module_shape[0],
                      pixels_y=detector.module_shape[1])

        if ctrl_device_id is None:
            ctrl_device_id = caldata.detector['karabo_id_control'] + \
                'MDL/FPGA_COMP'

        try:
            ctrl_src = data[ctrl_device_id]
        except KeyError:  # Should be SourceNameError
            raise ValueError(f'control source {ctrl_device_id} not found in '
                             f'DataCollection')

        auto_params = dict()

        try:
            auto_params['gain_setting'] = ctrl_src['gain'] \
                .as_single_value()
        except KeyError:  # Should be PropertyNameError
            pass

        try:
            auto_params['gain_mode'] = ctrl_src['gainModeIndex'] \
                .as_single_value()
        except KeyError:  # Should be PropertyNameError
            pass

        try:
            auto_params['integration_time'] = ctrl_src['integrationTime'] \
                .as_single_value()
        except KeyError:  # Should be PropertyNameError
            pass

        caldata._condition_params.update(auto_params)
        caldata._condition_params.update(condition_params)

        return caldata

    def dark_condition(self, **condition_params):
        cond = OperatingCondition(condition_params, self._condition_params)
        cond.set_required('Sensor Bias Voltage', 300.0)
        cond.set_required('Pixels X', 512)
        cond.set_required('Pixels Y', 128)

        # These parameters are optional, as some very old constants
        # don't have it. Still, they get default values to work when
        # used with recent data.
        cond.set_optional('Memory cells', 202)
        cond.set_optional('Acquisition rate', 4.5)

        # These conditions each got added later on.
        cond.set_optional('Gain setting')
        cond.set_optional('Gain mode')
        cond.set_optional('Integration time')

        if cond.get('Integration time', None) == 12:
            # Remove integration time parameter if its value is
            # identical to its legacy default of 12 to keep
            # compatibility with older constants.
            del cond['Integration time']

        return cond

    def illuminated_condition(self, **condition_params):
        cond = self.dark_condition(**condition_params)
        cond.set_required('Source energy', 9.2)
        return cond


class LPD_CalibrationData(DualConditionCalibrationData):
    """Calibration data for the LPD detector."""

    dark_calibrations = {'Offset', 'Noise', 'BadPixelsDark'}
    illuminated_calibrations = {'RelativeGain', 'GainAmpMap', 'FFMap',
                                'BadPixelsFF'}

    def dark_condition(self, **condition_params):
        cond = OperatingCondition(condition_params, self._condition_params)
        cond.set_required('Sensor Bias Voltage', 250.0)
        cond.set_required('Memory cells', 512)
        cond.set_required('Pixels X', 256)
        cond.set_required('Pixels Y', 256)
        cond.set_required('Feedback capacitor', 5.0)
        return cond

    def illuminated_condition(self, **condition_params):
        cond = self.dark_condition(**condition_params)
        cond.set_required('Source Energy', 9.2)
        cond.set_required('category', 1)
        return cond


class DSSC_CalibrationData(CalibrationData):
    """Calibration data for the DSSC detetor."""

    calibrations = {'Offset', 'Noise'}

    def condition(self, **condition_params):
        cond = OperatingCondition(condition_params, self._condition_params)
        cond.set_required('Sensor Bias Voltage', 100.0)
        cond.set_required('Memory cells', 400)
        cond.set_required('Pixels X', 512)
        cond.set_required('Pixels Y', 128)
        cond.set_optional('Pulse id checksum')
        cond.set_optional('Acquisition rate')
        cond.set_optional('Encoded gain')
        return cond

    @classmethod
    def from_data(cls, data, modules=None, client=None, ctrl_device_id=None,
                  **condition_params):
        from .components import DSSC1M
        det = DSSC1M(data, modules)

        caldata = cls(det.detector_name, sorted(det.modno_to_source.keys()),
                      client=client, memory_cells=det.frames_per_train,
                      pixels_x=det.module_shape[0],
                      pixels_y=det.module_shape[1])

        if ctrl_device_id is None:
            ctrl_device_id = caldata.detector['karabo_id_control'] + \
                '/MDL/FPGA_COMP'

        # Assume for now that all quadrants are configured identically.

        try:
            ctrl_src = data[ctrl_device_id]
        except SourceNameError:
            raise ValueError(f'control source {ctrl_device_id} not found in '
                             f'DataCollection')

        auto_params = dict()

        try:
            auto_params['gain_setting'] = ctrl_src['gain'].single_value()
        except PropertyNameError:
            pass

        try:
            auto_params['gain_mode'] = ctrl_src['gainModeIndex'].single_value()
        except PropertyNameError:
            pass

        try:
            cycle_length = data.get_run_value(ctrl_device_id,
                                              'sequencer.cycleLength')
            auto_params['acquisition_rate'] = 4.5 * (22.0 / cycle_length)
        except PropertyNameError:
            pass

        caldata._condition_params.update(auto_params)
        caldata._condition_params.update(condition_params)

        return caldata


class JUNGFRAU_CalibrationData(CalibrationData):
    """Calibration data for the JUNGFRAU detector."""

    calibrations = {'Offset10Hz', 'Noise10Hz', 'BadPixelsDark10Hz',
                    'RelativeGain10Hz', 'BadPixelsFF10Hz'}

    @classmethod
    def from_data(cls, data, modules=None, client=None, ctrl_device_id=None,
                  **condition_params):
        from extra_data.components import JUNGFRAU
        det = JUNGFRAU(data, modules=modules)

        caldata = cls(det.detector_name, sorted(det.modno_to_source.keys()),
                      client=client, memory_cells=det.frames_per_train,
                      pixels_x=det.module_shape[0],
                      pixels_y=det.module_shape[1])

        if ctrl_device_id is None:
            ctrl_device_id = caldata.detector['karabo_id_control'] + \
                '/DET/CONTROL'

        try:
            ctrl_src = data[ctrl_device_id]
        except SourceNameError:
            raise ValueError(f'control source {ctrl_device_id} not found in '
                             f'DataCollection')

        auto_params = dict()

        try:
            auto_params['sensor_bias_voltage'] = ctrl_src['vHighVoltage.'] \
                .as_single_value()
        except (PropertyNameError, NoDataError):
            pass

        try:
            auto_params['integration_time'] = ctrl_src['exposureTime'] \
                .as_single_value()
        except (PropertyNameError, NoDataError):
            pass

        try:
            auto_params['gain_setting'] = data.get_run_value(
                ctrl_device_id, 'settings')
        except (PropertyNameError, MultiRunError):
            pass

        caldata._condition_params.update(auto_params)
        caldata._condition_params.update(condition_params)

        return caldata

    def condition(self, **condition_params):
        cond = OperatingCondition(condition_params, self._condition_params)
        cond.set_required('Sensor Bias Voltage', 180.0)
        cond.set_required('Memory Cells', 16.0)
        cond.set_required('Pixels X', 1024.0)
        cond.set_required('Pixels Y', 512.0)
        cond.set_required('Integration Time', 10.0)
        cond.set_required('Sensor temperature', 291.0)
        cond.set_optional('Gain Setting')
        return cond


class PNCCD_CalibrationData(CalibrationData):
    pass


class EPIX_CalibrationData(CalibrationData):
    pass
