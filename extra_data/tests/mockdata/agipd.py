import numpy as np

from .base import DeviceBase


class AGIPDMDL(DeviceBase):
    def __init__(
        self, device_id,
        rep_rate=True,
        gain_setting=True,
        integration_time=True
    ):
        super().__init__(device_id)
        self.rep_rate = rep_rate
        self.gain_setting = gain_setting
        self.integration_time = integration_time

        # A sample of some of the available keys.
        self.control_keys = [
            ('acquisitionTime', 'u2', ()),
            ('bunchStructure.nPulses', 'u2', ()),
            ('bunchStructure.firstPulse', 'u2', ()),
            ('bunchStructure.methodIndex', 'u2', ()),
            ('egsgat', 'u2', ()),
            ('g1sgat', 'u2', ()),
            ('g2sgat', 'u2', ()),
            ('pcRowNr', 'u2', ()),
            ('t0Delay', 'u8', ()),
            ('ticolm', 'u2', ()),
            ('vrfcds', 'u2', ()),
        ]

    def write_control(self, f):
        super().write_control(f)

        ctrl_grp = f'CONTROL/{self.device_id}'
        run_grp = f'RUN/{self.device_id}'
        for grp in [ctrl_grp, run_grp]:
            length = self.ntrains if grp == ctrl_grp else 1

            f[f"{grp}/setupr/value"] = np.array([32]*length)
            f[f"{grp}/patternTypeIndex/value"] = np.array([4]*length)
            f[f"{grp}/gainModeIndex/value"] = np.array([0]*length)

            if self.integration_time:
                f[f"{grp}/integrationTime/value"] = np.array([15]*length)

            if self.rep_rate:
                f[f"{grp}/bunchStructure/repetitionRate/value"] = np.array(
                    [4.5]*length)
            if self.gain_setting:
                f[f"{grp}/gain/value"] = np.array([0]*length)


class AGIPD1MFPGA(DeviceBase):
    # A sample of some of the available keys.
    control_keys = [
        ('adcLatency', 'u4', ()),
        ('adcTrigger', 'u4', ()),
        ('asicCS', 'u4', ()),
        ('bootId', 'u4', ()),
        ('commandCounter', 'u4', ()),
        ('delays', 'u4', (8,)),
        ('heartbeatInterval', 'i4', ()),
        ('integrationOffset', 'u4', ()),
        ('integrationPeriod', 'u4', ()),
        ('mask', 'u4', ()),
        ('performanceStatistics.messagingProblems', '|u1', ()),
        ('performanceStatistics.enable', '|u1', ()),
        ('performanceStatistics.processingLatency', 'f4', ()),
        ('performanceStatistics.maxProcessingLatency', 'u4', ()),
        ('performanceStatistics.numMessages', 'u4', ()),
        ('performanceStatistics.maxEventLoopLatency', 'u4', ()),
        ('port', 'i4', ()),
        ('sleepTime', 'f4', ()),
    ]


class AGIPD500KFPGA(DeviceBase):
    # A sample of some of the available keys.
    control_keys = [
        ('highVoltage.actual', 'u2', ()),
        ('highVoltage.target', 'u2', ()),
    ]

    def write_control(self, f):
        super().write_control(f)
        ctrl_grp = f'CONTROL/{self.device_id}'
        run_grp = f'RUN/{self.device_id}'
        for grp in [ctrl_grp, run_grp]:
            length = self.ntrains if grp == ctrl_grp else 1
            f[f"{grp}/highVoltage/actual/value"] = np.array([200]*length)


class AGIPD1MPSC(DeviceBase):
    def __init__(
        self, device_id,
        bias_voltage=True,
    ):
        super().__init__(device_id)
        self.bias_voltage = bias_voltage
        # A sample of some of the available keys.
        self.control_keys = [
            ('applyInProgress', '|u1', ()),
            ('autoRearm', '|u1', ()),
            ('channels.U0.status', 'i4', ()),
            ('channels.U0.switch', 'i4', ()),
            ('channels.U0.voltage', 'f4', ()),
            ('channels.U0.superVisionMaxTerminalVoltage', 'f4', ()),
            ('channels.U0.voltageRampRate', 'f4', ()),
            ('channels.U0.measurementCurrent', 'f4', ()),
            ('channels.U0.current', 'f4', ()),
            ('channels.U0.supervisionMaxCurrent', 'f4', ()),
            ('channels.U0.currentRiseRate', 'f4', ()),
            ('channels.U0.currentFallRate', 'f4', ()),
            ('channels.U0.measurementTemperature', 'i4', ()),
            ('channels.U0.supervisionBehavior', 'i4', ()),
            ('channels.U0.tripTimeMaxCurrent', 'i4', ()),
            ('channels.U0.configMaxSenseVoltage', 'f4', ()),
        ]

    def write_control(self, f):
        super().write_control(f)
        ctrl_grp = f'CONTROL/{self.device_id}'
        run_grp = f'RUN/{self.device_id}'
        if self.bias_voltage:
            for grp in [ctrl_grp, run_grp]:
                length = self.ntrains if grp == ctrl_grp else 1
                f[f"{grp}/channels/U0/measurementSenseVoltage/value"] = np.array(  # noqa
                    [300]*length)
