Architecture of Writer
======================

Structure
---------

.. py:class:: FileWriter

    .. py:attribute:: FileWriter._meta
        :type: Options

    .. py:attribute:: FileWriter.datasets
        :type: dict(str, DatasetBase)

    .. py:attribute:: FileWriter.sources
        :type: dict(str, Source)

.. py:class:: Source
    
    .. py:attribute:: dsno
        :type: dict(str, int)
            
    .. py:attribute:: Source.datasets
        :type: list(DatasetBase)
            
    .. py:attribute:: Source.file_ds
        :type: list(DatasetWriterBase)
            
    .. py:attribute:: Source.data
        :type: list(DataQueue)

Classes
-------

:class:`FileWriter` is a class that represents a file or sequence of files.
It is a virtual mapping of a file, meaning data added to :class:`FileWriter`
is actually written to the file if matches the file format.

:Invariant: :class:`FileWriter` manages structure information such as
    a collection of sources and their keys but knows nothing how they
    actually work.

:class:`Source` is a class that represents one source. It controls the
correspondence of records in all keys with each other, when one or more trains
are filled correctly, writes them down, forms the index.

:Invariant: manages keys but knows nothing how they actually work.

:class:`DataQueue` is a class that queues different data chunks to be written.
Retains data added to one key in memory until all source keys are filled with
data.


knows nothing about the data format, whether they match
the key and how to write it out.

:Invariant: the data held by the :class:`DataQueue` cannot be written to
    disk because such writing would violate the file format. If the data
    is still in the :class:`DataQueue` when writing is done, it drops.

:class:`DatasetWriterBase` is a class that writers data. It determines how
exactly and in what chunks the data will be written.

:Invariant: only it knows how to write data.

:Invariant: the data passed to the :class:`DatasetWriterBase` must appear in the
    file. If writing is buffered all data should be equally flushed
    on successful writing or on exception.

:Subclasses:
    :class:`DatasetDirectWriter` provides the direct writing
    
    :class:`DatasetBufferedWriter` provides the buffering writing

:class:`DatasetBase` is a class that describes one key of a data source.
The set of :class:`DatasetBase` descriptors fully defines the structure of
the file.

:Invariant: descriptors are attributes of a class, not of an instance. Therefore,
    they only modify themselves in the constructor and associated methods. The
    rest of the methods are passed an instance of the :class:`FileWriter` class,
    which they can modify.

:Invariant: descriptors provide constructors for different elements of the entire
    machinery: :class:`Source`, :class:`DatasetWriterBase`, :class:`DataSetterBase`.
    Thus, the machinery is controlled by overriding constructors in the descriptor
    class.

:Invariant: only it knows what data fit this key.

:Subclasses:
    :class:`Dataset` - standard generalized descriptor for EuXFEL datasets.

:class:`DataSetterBase` is a descriptor class that allows data to be added to
a file via assignment to an attribute of the `FileWriter` class.

:Subclasses:
    :class:`BlockedSetter` - blocks the attribute interface raising an exception
    
    :class:`DataSetter` - adds the data corresponded to the `Dataset` descriptor

:class:`MultiTrainData` is a proxy class that groups the number of entries
array and data itself.

:class:`Options` is a helper class that organizes the initialization and
overriding of options for :class:`FileWriter` subclasses.
