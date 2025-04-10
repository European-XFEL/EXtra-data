"""Exception classes specific to extra_data."""


class FileStructureError(Exception):
    pass


class SourceNameError(KeyError):
    def __init__(self, source=None, custom_message=None):
        self.source = source
        self.custom_message = custom_message

    def __str__(self):
        return self.custom_message if self.custom_message is not None else (
            "This data has no source named {!r}.\n"
            "See data.all_sources for available sources.".format(self.source)
        )


class PropertyNameError(KeyError):
    def __init__(self, prop, source):
        self.prop = prop
        self.source = source

    def __str__(self):
        return "No property {!r} for source {!r}".format(self.prop, self.source)


class TrainIDError(KeyError):
    def __init__(self, train_id):
        self.train_id = train_id

    def __str__(self):
        return "Train ID {!r} not found in this data".format(self.train_id)


class AliasError(KeyError):
    def __init__(self, alias):
        self.alias = alias

    def __str__(self):
        return f"'{self.alias}' not known as alias for this data"


class MultiRunError(ValueError):
    def __str__(self):
        return (
            "The requested data is only available for a single run. This "
            "EXtra-data DataCollection may have data from multiple runs, e.g. "
            "because you have used .union() to combine data. Please retrieve "
            "this information before combining."
        )


class NoDataError(ValueError):
    def __init__(self, source, key=None):
        self.source = source
        self.key = key

    def __str__(self):
        if self.key is not None:
            return 'This data is empty for key {!r} of source {!r}'.format(
                self.key, self.source)
        else:
            return 'This data is empty for source {!r}'.format(self.source)
