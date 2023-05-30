import ast
from warnings import warn

from ..sourcedata import SourceData

class Scantool:
    """Interface for the XFEL scantool (Karabacon).

    Note that the :func:`repr` function for this class uses :meth:`Scantool.format`
    internally, so evaluating a :class:`Scantool` object in a Jupyter notebook
    cell will print the scantool configuration:

    .. code-block:: ipython

                -----------------------------------------------------------
        In [1]: |scantool = Scantool(run)                                 |
                |scantool                                                 |
                -----------------------------------------------------------
        Out[1]: Scantool (MID_RR_SYS/MDL/KARABACON) configuration:
                  Scan type: dscan
                  Acquisition time: 1.0s

                Motors:
                  DET2_TX (MID_EXP_DES/MOTOR/DET2_TX): -0.05 -> 0.05, 100 steps

    See the `scantool documentation
    <https://rtd.xfel.eu/docs/scantool/en/latest/index.html>`_ for more
    information about the device itself.
    """
    #: The name of the scantool device.
    source_name: str

    #: :class:`SourceData` object for the device.
    source: SourceData

    #: Boolean to indicate whether the scantool was used during the run.
    active: bool

    #: The type of scan configured (ascan, dscan, mesh, etc).
    scan_type: str

    #: Acquisition time in seconds.
    acquisition_time: float

    #: List of aliases of the motors being moved. Note that this is a scantool
    #: alias, not an ``extra_data`` alias.
    motors: list

    #: A dictionary mapping motor aliases to their actual device names.
    #:
    #: .. warning::
    #:    This property is obtained by parsing a configuration string, which may
    #:    not be compatible with previous versions of the scantool. If it was not
    #:    possible to get the device names then a warning will be printed when
    #:    initializing the class, and this property will be ``None``.
    motor_devices: dict

    #: A dictionary mapping motor aliases to the number of steps they were
    #: scanned over.
    steps: dict

    #: A dictionary mapping motor aliases to their start positions.
    start_positions: dict

    #: A dictionary mapping motor aliases to their stop positions.
    stop_positions: dict

    def __init__(self, run, src=None):
        """Read scantool data from a run.

        Parameters
        ----------
        run: :class:`DataCollection`
            A run containing the scantool.
        src: str, optional
            The device name of the scantool. If this is not passed the class
            will try to find the right device automatically.
        """
        if src is None:
            possible_devices = [x for x in run.control_sources if "KARABACON" in x]
            if len(possible_devices) == 0:
                raise RuntimeError("Could not find a KARABACON device in the run, please pass an explicit source name with the `src` argument'")
            elif len(possible_devices) == 1:
                src = possible_devices[0]
            else:
                raise RuntimeError(f"Found multiple possible scantools, please pass one explicitly with the `src` argument: {', '.join(possible_devices)}")

        values = run.get_run_values(src)

        # Get scan metadata and list of motors
        self.source_name = src
        self.source = run[src]
        self.active = self.source["isMoving"].ndarray().any()
        self.scan_type = values["scanEnv.scanType.value"]
        self.acquisition_time = values["deviceEnv.acquisitionTime.value"]
        self.motors = [x.decode() for x in values["deviceEnv.activeMotors.value"] if len(x) > 0]

        # The deviceEnv.activeMotors property stores the motor aliases,
        # but we can try to get the actual device names from the
        # actualConfiguration property.
        self.motor_devices = None
        motors_line = [x for x in values["actualConfiguration.value"].split("---") if "Motors:" in x]
        device_names_warning = "Couldn't extract the Karabo device names for the active motors."
        if len(motors_line) == 1:
            try:
                motors_list = motors_line[0].strip()[len("Motors: "):]
                motors_list = [x.split(":")[0] for x in ast.literal_eval(motors_list)]
                self.motor_devices = dict(zip(self.motors, motors_list))
            except Exception:
                warn(device_names_warning)
        else:
            warn(device_names_warning)

        # Get the number of steps and start/stop positions for each motor
        n_motors = len(self.motors)
        self.steps = dict(zip(self.motors,
                              values["scanEnv.steps.value"][:n_motors]))
        self.start_positions = dict(zip(self.motors,
                                        values["scanEnv.startPoints.value"][:n_motors]))
        self.stop_positions = dict(zip(self.motors,
                                       values["scanEnv.stopPoints.value"][:n_motors]))

    def _motor_fmt(self, name, compact=True):
        """Helper function to format a single motor"""
        motion_info = f"{self.start_positions[name]} -> {self.stop_positions[name]}, {self.steps[name]} steps"

        if compact:
            return f"{name} ({motion_info})"
        elif not compact:
            if self.motor_devices is None:
                return f"{name}: {motion_info}"
            else:
                return f"{name} ({self.motor_devices[name]}): {motion_info}"

    def format(self, compact=True):
        """Format information about the scantool as a string.

        Parameters
        ----------
        compact: bool
            Whether to print the information in a compact 1-line format or a
            multi-line format.
        """
        if not self.active:
            device = " " if compact else f" ({self.source_name}) "
            return f"Scantool{device}not active."
        else:
            if compact:
                motor_info = [self._motor_fmt(name, compact=True) for name in self.motors]
                return f"{self.scan_type} {self.acquisition_time}s: {', '.join(motor_info)}"
            else:
                info = [f"Scantool ({self.source_name}) configuration:",
                        f"  Scan type: {self.scan_type}",
                        f"  Acquisition time: {self.acquisition_time}s",
                        "",
                        "Motors:"]

                info.extend(["  " + self._motor_fmt(name, compact=False)
                             for name in self.motors])
                return "\n".join(info)

    def __repr__(self):
        return self.format(compact=False)

    def __str__(self):
        return self.format(compact=True)
