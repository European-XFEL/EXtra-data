"""
Helpers functions for the euxfel_h5tools package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import os
import sys
from shutil import get_terminal_size


def available_cpu_cores():
    # This process may be restricted to a subset of the cores on the machine;
    # sched_getaffinity() tells us which on some Unix flavours (inc Linux)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        # Fallback, inc on Windows
        ncpu = os.cpu_count() or 2
        return min(ncpu, 8)


def default_num_threads(fixed_limit=16):
    # Default to 16, EXTRA_NUM_THREADS, or available CPU cores (picking lowest)
    threads_limits = ([fixed_limit, available_cpu_cores()])
    try:
        threads_limits.append(int(os.environ['EXTRA_NUM_THREADS']))
    except (KeyError, ValueError):  # Not set, or not an integer
        pass
    return min(threads_limits)


def progress_bar(done, total, suffix=" "):
    line = f"Progress: {done}/{total}{suffix}[{{}}]"
    length = min(get_terminal_size().columns - len(line), 50)
    filled = int(length * done // total)
    bar = "#" * filled + " " * (length - filled)
    return line.format(bar)


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))
