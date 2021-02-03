from .base import DeviceBase

class JUNGFRAUModule(DeviceBase):
    output_channels = ('daqOutput/data',)

    instrument_keys = [
        ('adc', 'u2', (16, 512, 1024)),
        ('frameNumber', 'u8', (16,)),
        ('gain', 'u1', (16, 512, 1024)),
        ('memoryCell', 'u1', (16,)),
        ('timestamp', 'f8', (16,)),
    ]

class JUNGFRAUControl(DeviceBase):
    control_keys = [
        ('acquisitionTime', 'f4', ()),
        ('angDir', 'i2', (1000,)),
        ('binSize', 'f4', (1000,)),
        ('bitDepth', 'i4', ()),
        ('dataStorage.enable', 'u1', ()),
        ('dataStorage.fileIndex', 'i4', ()),
        ('delayAfterTrigger', 'f4', (1000,)),
        ('detectorHostPort', 'u2', (1000,)),
        ('detectorHostStopPort', 'u2', (1000,)),
        ('exposurePeriod', 'f4', ()),
        ('exposureTime', 'f4', ()),
        ('exposureTimeout', 'u4', ()),
        ('exposureTimer', 'u2', ()),
        ('globalOff', 'f4', (1000,)),
        ('heartbeatInterval', 'i4', ()),
        ('lock', 'i2', (1000,)),
        ('master', 'i2', ()),
        ('maximumDetectorSize', 'i4', (1000,)),
        ('moveFlag', 'i2', (1000,)),
        ('numberOfCycles', 'i8', ()),
        ('numberOfFrames', 'i8', ()),
        ('numberOfGates', 'i8', ()),
        ('online', 'i2', (1000,)),
        ('performanceStatistics.enable', 'u1', ()),
        ('performanceStatistics.maxEventLoopLatency', 'u4', ()),
        ('performanceStatistics.maxProcessingLatency', 'u4', ()),
        ('performanceStatistics.messagingProblems', 'u1', ()),
        ('performanceStatistics.numMessages', 'u4', ()),
        ('performanceStatistics.processingLatency', 'f4', ()),
        ('pollingInterval', 'u4', ()),
        ('progress', 'i4', ()),
        ('rOnline', 'i2', ()),
        ('rxTcpPort', 'u2', (1000,)),
        ('rxUdpPort', 'u2', (1000,)),
        ('rxUdpSocketSize', 'u4', ()),
        ('storageCellStart', 'i2', ()),
        ('storageCells', 'i2', ()),
        ('threaded', 'i2', ()),
        ('triggerPeriod', 'f4', ()),
        ('vHighVoltage', 'u4', (1000,)),
        ('vHighVoltageMax', 'u4', ()),
    ]

class JUNGFRAUMonitor(DeviceBase):
    control_keys = sum(([
        (f'module{n}.adcTemperature', 'f8', ()),
        (f'module{n}.fpgaTemperature', 'f8', ()),
    ] for n in range(1, 9)), [])

class JUNGFRAUPower(DeviceBase):
    control_keys = [
        ('current', 'f8', ()),
        ('pollingInterval', 'f8', ()),
        ('port', 'u2', ()),
        ('temperature', 'f8', ()),
        ('voltage', 'f8', ()),
    ]
