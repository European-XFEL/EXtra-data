"""
Helpers functions for the euxfel_h5tools package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import os


def available_cpu_cores():
    # This process may be restricted to a subset of the cores on the machine;
    # sched_getaffinity() tells us which on some Unix flavours (inc Linux)
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    else:
        # Fallback, inc on Windows
        ncpu = os.cpu_count() or 2
        return min(ncpu, 8)
