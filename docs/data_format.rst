
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

which denotes an HDF5 file for the ``RAW`` data class of aggregator ``AGIPD04``
of sequence ``2`` in run ``348``. Within a run the grouping of sources
into  aggregators does not change. Each *proposal* can collect any number of runs
during their granted beamtime.

This document describes the most recent version **1.3** of this file format. While
earlier version are used for data written at the time, their use is discouraged
for any new files. The appendix lists the differences between each versions.


Data sources
------------

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
  pattern ``DOMAIN/TYPE/MEMBER:PIPELINE/GROUP``. The data is saved in the ``INSTRUMENT``
  top-level groups described further below.

  The last component ``GROUP`` or index group is generally considered part of the key
  rather than the source itself. In a Karabo perspective it is equivalent to the
  top-level key in pipeline data. In files however, an instrument source may have
  a different number of entries per train for each of its index groups and it is
  thus treated differently than keys further down in the hierarchy.


HDF5 file structure
-------------------

Every HDF5 file must contain the top-level groups ``/METADATA`` and ``/INDEX``.
Depending on the included sources, there may additionally be the groups
``/CONTROL``, ``/RUN`` and ``/INSTRUMENT``.


METADATA
~~~~~~~~

The ``METADATA`` group in an HDF5 file contains auxiliary information as individual
datasets, most of which are constant across a run and or even proposal. Even when only
containing a single entry, all these datasets are 1D with a length of 1 or more.

For any given collections, not all of these datasets may be present depending on how
it was created. The following datasets however are considered mandatory to allow
proper interpretation of a file's structure:

* ``dataFormatVersion [str]`` Data file format version of this file.

* ``dataSources`` describes the sources in this file in three different representations.

  * ``dataSources/root [str]`` lists the top-level group a source is found in, ``CONTROL``
    or ``INSTRUMENT``.

  * ``dataSources/deviceId [str]`` lists the source names itself. For instrument sources,
    this includes the top-level key called index group, and the same source may thus be listed
    multiple times for each of its index groups.

  * ``dataSources/dataSourceId [str]`` lists the combination of the prior two, i.e. the
    full path to each source's index group.

For scientific data, it is recommended to include the following datasets to describe their
origin:

* ``creationDate [str]`` [what was this time again?]

* ``updateDate [str]``  [probably last change to this file?]

* ``proposalNumber [uint32]`` Proposal number this file belongs to.

* ``runNumber [uint32]``  Run number this file belongs to.

* ``sequenceNumber [uint32]``  Sequence number this file has for the aggregator it belongs to.

Raw data recorded with the EuXFEL DAQ software will contain the datasets to indicate the
software versions used in this process:

* ``daqLibrary [str]`` EuXFEL DAQ software version used to write this file

* ``karaboFramework [str]`` Karabo framework version the DAQ software ran in

INDEX
~~~~~

The ``INDEX`` group contains information about the *trains* contained in the file and how
the actual data rows in ``CONTROL`` and ``INSTRUMENT`` relate to them. All datasets in this group
are 1D and have a length identical to the number of trains in the file.

There are three datasets at the top-level of this group:

* ``trainId [uint64]`` lists the global train ID for this train entry.

* ``timestamp [uint64]`` lists the number of nanseconds since the Epoch for this train entry.

* ``flag [int32]`` lists ``1`` for safe train entries and ``0`` for train entries where the timing
  may be unreliable, e.g. because it is attributed to the wrong train ID. For DAQ recordings up
  to version **1.2**, this is only the case when a source different than the timeserver sent the first
  data entry for a given train.

* ``origin [int32]`` lists the actual source index into ``METADATA/dataSources`` that sent that first
  entry for each given train entry, or ``-1`` if it is the timeserver. For DAQ recordings up to
  version **1.2**, every entry with a non-negative ``origin`` will have a ``flag`` of ``0``.

For each source in ``METADATA/dataSources/deviceId``, the ``INDEX`` group then also contains two
datasets that map the train entries in the top-level datasets above to each source's data rows
in ``CONTROL`` or ``INSTRUMENT``:

* ``INDEX/{deviceId}/count [uint64]`` counts how many data samples did
  this source record for each train. This may be 0 if no data was recorded.
* ``INDEX/{deviceId}/first [uint64]`` contains the index at which the
  corresponding data for each train starts in the arrays for this device.

Thus, to find the data for a given train ID::

    train_index = list(file['INDEX/trainId']).index(train_id)
    first = file[f'INDEX/{device_id}/first'][train_index]
    count = file[f'INDEX/{device_id}/count'][train_index]
    train_data = file[f'INSTRUMENT/{device_id}/{key}][first:first+count]


CONTROL and RUN
~~~~~~~~~~~~~~~

For each *CONTROL* entry in ``METADATA/dataSources``, there is a group with
that name in the file with further arbitrarily nested subgroups representing different
keys of that source, e.g. ``CONTROL/SA1_XTD2_XGM/DOOCS/MAIN/current/bottom/output``
for the key ``current.bottom.output`` of source ``SA1_XTD2_XG/DOOCS/MAIN``. Note that
while the key hierarchy is expressed using groups in files, a dot is commonly used
to separate the components.

The leaves of this tree are pairs of datasets called ``timestamp`` and ``value``.
Each dataset has one entry per train, and the ``timestamp`` record when the
current value was updated, which is typically less than once per train and thus
likely in the past.

The key groups themselves may have one or more HDF attributes attached with
additional metadata:

* ``displayedName [str]`` may denote a more exhaustive name for this key, e.g.
  ``Complete Target Burst duration`` for ``totBurstDuration``.
* ``alias [str]`` may specify an alternative name depending on context, e.g.
  a hardware-specific designation for the value of a key.
* ``description [str]`` may contain a full text explaining this key.
* ``metricPrefixSymbol [str]`` may specify the metric prefix symbol for the unit
  this key's values are expressed in, e.g. ``G``, ``k`` or ``n``.
* ``unitSymbol [str]`` may specify the unit symbol this key's values are expressed
  in, e.g. ``A``, ``Hz`` or ``eV``. Enumerations may use the symbol ``#`` and ratios
  the symbol ``%``.

EuXFEL DAQ recording often contain further attributes corresponding to attributes in
the Karabo control system.

``RUN`` holds a complete duplicate of the ``CONTROL`` hierarchy, but each pair
of ``timestamp`` and ``value`` contain only one entry taken at the start of
the run. All datasets continue to be vectors, so even for scalar values the
first dimension has length 1. It may also contain additional keys not present in
``CONTROL``, e.g. whose values either do not change or is not relevant across trains.


INSTRUMENT
~~~~~~~~~~

For each *INSTRUMENT* entry in ``METADATA/dataSources``, there is a group with
that name in the file with further arbitrarily nested subgroups representing different
keys of that source, e.g. ``INSTRUMENT/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data``
for the key ``image.data`` of source ``SPB_DET_AGIPD1M-1/DET/0CH:xtdf``. Unlike for
*CONTROL* sources, the top-level part of the key called index group (in this example,
``image``) is part of the entry in ``METADATA/dataSources`` to allow a variable number
of data entries per train for each of these index groups. Note that while the key
hierarchy is expressed using groups in files, a dot is commonly used to separate
the components.

The leafs of this tree directly contain the datasets holding the key values. Those
datasets of the same index group of a given source have the same length in the first
dimension, with each row representing a successive reading. The index group's ``INDEX``
records can be used to connect them to the respective trains.

As with *CONTROL* sources, the keys of *INSTRUMENT* sources may have the same HDF
attributes attached with additional metadata.


Format versions
---------------

1.3
~~~

The EuXFEL DAQ software is using this format version since January 2023.

This section only lists the differences to past format versions.

1.2
~~~

* There are no metadata attributes for keys in ``CONTROL``, ``RUN``  and ``INSTRUMENT``.

The EuXFEL DAQ software used this format version between July 2021 and Februrary 2023.

1.1
~~~

* ``INDEX/flag`` dataset is similar to ``INDEX/origin`` in later versions, listing the index into ``METADATA/dataSources`` of the source that sent the first entry for a given train. Unlike ``INDEX/origin`` however, the time server itself is a virtual source with index ``0`` rather than ``-1``.

  **Warning:** This flips the meaning compared to earlier versions with ``0`` indicating a *safe* train and a positive number for unreliable timing.
* ``METADATA/dataSources`` contains a static virtual source ``Karabo_TimeServer`` with an empty entry in ``METADATA/dataSources/root``.

The EuXFEL DAQ software used this format version only briefly around July 2021.

1.0
~~~

* ``INDEX`` group contains only the top-level datasets ``trainId``, ``timestamp``, ``flag``.

The EuXFEL DAQ software used this format version between February 2020 and September 2021.

0.5
~~~

**Warning:** This file format version is lacking the ``METADATA/dataFormatVersion`` dataset and can thus only be inferred from its structure.

* ``INDEX`` group contains only the top-level dataset ``trainId``.
* ``METADATA`` group is identical to ``METADATA/dataSources`` in later versions,
  i.e. directly contains the datasets ``root``, ``deviceId`` and ``dataSourceId``.

The EuXFEL DAQ software used this format version between February 2018 and April 2020.

0.1
~~~

**Warning:** This file format version is lacking the ``METADATA/dataFormatVersion`` dataset and can thus only be inferred from its structure.

Same as 0.5 in addition to:

* ``INDEX/{deviceId}`` group specifies the mapping from trains to data rows of each source via ``first``/``last`` datasets with ``last = first + count - 1`` denoting the last row index belonging to a particular train.

The EuXFEL DAQ software used this format version until April 2018.


References
----------

.. [1] Decking et al: *A MHz-repetition-rate hard X-ray free-electron laser driven by a superconducting linear accelerator*, Nature Photonics 391-397, 2020
.. [2] European XFEL DAQ and Control systems naming convention: https://docs.xfel.eu/share/s/dDHQtDIkRUiXPr9DM6WQ-Q
