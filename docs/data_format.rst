
Data files format
=================

Scientific data at European XFEL is saved as structured HDF5 files in format
called EXDF. Each file contains data for one or more *sources* having multiple
*keys* that carry values for certain *trains* [1]_. Most sources, in particular for
raw data, correspond to an entity in the Karabo control system called a *device*,
which may manage physical hardware but also purely perform software functions.

In most cases at European XFEL, such files are encountered with raw data recorded
during an experiment by its data acquisition system (DAQ) or with automatically
processed data created afterwards. Here, sources and trains are generally split
across multiple files with sources grouped by an *aggregator* and trains by
enumerated *sequences*. Sources with a very large data volume often have their own
aggregator (e.g. very fast digitizers) or are even be spread across multiple of
them (e.g. multi-module detectors). It is not guaranteed that the same sequence
for different aggregators will cover the same set of trains.

A continuous DAQ recording over a period of time is a *run* and includes all the
structured HDF5 in a single directory. Here these files follow a naming pattern::

    RAW-R0348-AGIPD04-S00002.h5
    
which denotes the HDF5 file with ``raw`` data for sequence ``2`` of the aggregator
``AGIPD04`` in run ``348``. Within a run the grouping of sources into aggregators
does not change. Each *proposal* can collect any number of runs during their granted
beamtime.

This document describes the most recent version ``1.2`` of this file format. While
earlier version are used for data written at the time, their use is discouraged
for any new files. The appendix lists the differences between each versions.


Data sources
------------

[went back and forth on this having its own section, but too often it seemed confusing to
leave it. I've tried to keep this description agnostic of Karabo, but ultimately of course
it does map back to it]

Sources differ in the semantics of data generation and validity by either being
*control data* (also called slow or broker data) or *instrument data*
(also called fast, pipeline or XTDF data). In terms of the Karabo control system,
control sources represent device properties while instrument sources are the data
sent by device output channels. As such, it is recommended to follow the Karabo
device naming convention [2]_ for source names.

* Control data represents a steady state that retains its value until changed again.
  Examples for this are motor positions or detector configuration like the frame rate.
  In general control data has a single value for every train in a file, as any train
  without it changing carries the value of the previous train. As such each value is
  accompanied by the timestamp this value became current. Based on the Karabo device
  naming convention, control sources should follow the pattern ``DOMAIN/TYPE/MEMBER``.
  Their data is saved in the ``CONTROL`` and ``RUN`` top-level groups described
  further below.
 
* Instrument sources represent momentary data that is only valid for a single train
  or pulse. This covers most scientific detectors such as digitizers, cameras and
  more. These sources are never guaranteed to have data for every train, but may
  also have multiple and varying entries per train. Their names should follow the
  pattern ``DOMAIN/TYPE/MEMBER:PIPELINE/ROOT``. The data is saved in the ``INSTRUMENT``
  top-level groups described further below.

  The last component ``ROOT`` is often considered part of key rathern than the source
  itself and in a Karabo perspective equivalent to the top-level key in pipeline data.
  In files however, a Karabo pipeline source may have different entries based on
  this ``ROOT``.

[insert reference for train structure somewhere?]


HDF5 file structure
-------------------

Every HDF5 file must contain the top-level groups ``METADATA`` and ``INDEX``.
Depending on the included sources, there may additionally be the groups
``CONTROL``, ``RUN`` and ``INSTRUMENT``.
[note 1: is this clear enough it may be a combination of those?]
[note 2: based on actual files, ``RUN`` seems to always be there even without 
``CONTROL`` albeit empty, move it to the first list?]


METADATA
~~~~~~~~

The ``METADATA`` group in an HDF5 file contains auxiliary information as
individual datasets, most of which are constant across a run and or even
proposal. Some of these datasets may not be present in any given file depending
on how it was created. 

This list contains all the datasets to be expected for data recorded with the
EuXFEL DAQ software. Even when only containing a single entry, all these datasets
are 1D with a length of 1 or more.

[`ascii` is supposed to refer to "String (fixed-length)"]

* ``creationDate [ascii]`` [what was this time again?]

* ``daqLibrary [ascii]`` EuXFEL DAQ software version used to write this file, only applicable to raw recordings.

* ``dataFormatVersion [ascii]`` Data file format version of this file.

* ``dataSources`` describes the sources in this file in three different representations.

  * ``dataSources/root [ascii]`` lists the top-level group a source is found in, ``CONTROL`` or ``INSTRUMENT``.

  * ``dataSources/deviceId [ascii]`` lists the source names itself. For instrument sources, this includes the top-level key and the same source may thus be listed multiple times for each of its top-level keys.

  * ``dataSources/dataSourceId [ascii]`` lists the combination of the prior two, i.e. the full path to each source's top-level group.

* ``karaboFramework [ascii]`` Karabo framework version the DAQ software ran in, only applicable to raw recordings.

* ``proposalNumber [uint32]`` Proposal number this file belongs to.

* ``runNumber [uint32]``  Run number this file belongs to.

* ``sequenceNumber [uint32]``  Sequence number this file has for the aggregator it belongs to.

* ``updateDate [ascii]``  [probably last change to this file?]

[we should add the ```pycalibration`` release here, or some form of version in case of processing]


INDEX
~~~~~

The ``INDEX`` group contains information about the *trains* contained in the file and how
the actual data rows in `CONTROL` and `INSTRUMENT` relate to them. All datasets in this group
are 1D and have a length identical to the number of trains in the file.

There are three datasets at the top-level of this group:

* ``trainId [uint64]`` lists the global train ID for this train entry.

* ``timestamp [uint64]`` lists the number of nanseconds since the Epoch for this train entry.

* ``flag [int32]`` lists ``1`` for safe train entries and ``0`` for train entries where the timing
  may be unreliable, e.g. because it is attributed to the wrong train ID. For DAQ recordings up
  to `EXDF-v1.2`, this is only the case when a source different than the timeserver sent the first
  data entry for a given train.

* ``origin [int32]`` lists the actual source index into `METADATA/dataSources` that sent that first
  entry for each given train entry, or ``-1`` if it is the timeserver. For DAQ recordings up to
  `EXDF-v1.2`, every entry with a non-negative ``origin`` will have a ``flag`` of ``0``.

For each source in ``METADATA/dataSources/deviceId``, the ``INDEX`` group then also contains two
datasets that map the train entries in the top-level datasets above to each source's data rows
in ``CONTROL`` or ``INSTRUMENT``:

* ``INDEX/{deviceId}/count [uint64]``: For each train ID, how many data samples did
  this source record. This may be 0 if no data was recorded for this train.
* ``INDEX/{deviceId}/first [uint64]``: for each train ID, the index at which the
  corresponding data starts in the arrays for this device.

Thus, to find the data for a given train ID, we could do::

    train_index = list(file['INDEX/trainId']).index(train_id)
    first = file[f'INDEX/{device_id}/first'][train_index]
    count = file[f'INDEX/{device_id}/count'][train_index]
    train_data = file[f'INSTRUMENT/{device_id}/{key}][first:first+count]

Some older files use a different index format with first/last/status instead of
first/count. In this case, a status of 0 means that no data was recorded
for that train. [never saw those files, is it relevant enough to list it?]

CONTROL and RUN
~~~~~~~~~~~~~~~

For each *CONTROL* entry in ``METADATA/dataSources``, there is a group with
that name in the file with further arbitrarily nested subgroups representing different
keys of that device, e.g. ``/CONTROL/SA1_XTD2_XGM/DOOCS/MAIN/current/bottom/output``
for the key ``current/bottom/output`` of source ``SA1_XTD2_XG/DOOCs/MAIN``.

The leaves of this tree are pairs of datasets called ``timestamp`` and ``value``.
Each dataset has one entry per train, and the ``timestamp`` record when the
current value was updated, which is typically less than once per train and thus
likely in the past.

``RUN`` holds a complete duplicate of the ``CONTROL`` hierarchy, but each pair
of ``timestamp`` and ``value`` contain only one entry taken at the start of
the run. All datasets continue to be vectors, so even for scalar values the
first dimension has length 1.

INSTRUMENT
~~~~~~~~~~

For each *INSTRUMENT* entry in ``METADATA/dataSourceId``, there is a group with
that name in the file. All these datasets have the same length in the first dimension:
this represents the successive readings taken. The slices defined by the corresponding
datasets in *INDEX* work on this dimension.

Format versions
---------------

1.2, 1.0, 0.5: TBD




References
----------


.. [1] Decking et al: *A MHz-repetition-rate hard X-ray free-electron laser driven by a superconducting linear accelerator*, Nature Photonics 391-397, 2020
.. [2] European XFEL DAQ and Control systems naming convention: https://docs.xfel.eu/share/s/dDHQtDIkRUiXPr9DM6WQ-Q
