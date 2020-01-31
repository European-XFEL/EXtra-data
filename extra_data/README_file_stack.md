# Karabo-bridge-stack-files

That's a CLI script, meant to be executable using the Python3
from our distribution.

The functionality is kind of a combination of karabo-bridge-serve-files
and karabo-bridge-server-sim:

- read run data from a folder, using the EXtra-data machinery
- stack image data to a "file-like" array, shape (pulses, modules, ss, fs)
- put this into a single output source-dict incl. metadata with timestamp
  from "current" time
- stream the data and metadata using ZeroMQ

Motivation:

Provide real data, not mock-data, in a format that OnDA expects

