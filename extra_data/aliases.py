from collections import defaultdict
from collections.abc import Iterable

from .exceptions import AliasError

class AliasIndexer:
    """Enables item access via source and key aliases."""

    __slots__ = ('data',)

    def __init__(self, data):
        self.data = data

    def _resolve_any_alias(self, alias):
        alias = alias.lower().replace('_', '-')
        try:
            literal = self.data._aliases[alias]
        except KeyError:
            raise AliasError(alias) from None

        return literal

    def _resolve_source_alias(self, alias):
        source = self._resolve_any_alias(alias)

        if isinstance(source, tuple):
            raise ValueError(f'{alias} not aliasing a source for this data')

        return source

    def __getitem__(self, aliased_item):
        if isinstance(aliased_item, tuple) and len(aliased_item) == 2:
            # Source alias with key literal.
            return self.data[self._resolve_source_alias(aliased_item[0]),
                                aliased_item[1]]
        elif isinstance(aliased_item, str):
            # Source or key alias.
            return self.data[self._resolve_any_alias(aliased_item)]

        raise TypeError('expected alias or (source alias, key) tuple')

    def _ipython_key_completions_(self):
        return list(self.data._aliases.keys())

    def __contains__(self, aliased_item):
        try:
            self[aliased_item]
            return True
        except KeyError:
            return False

    def __repr__(self):
        """
        Pretty-print all the aliases.
        """
        RED = "\033[91m"
        END_COLOR = "\033[0m"

        # Get the right icon for an alias
        def alias_icon(exists):
            if isinstance(exists, str):
                exists = exists in self

            return " " if exists else f"{RED}✗{END_COLOR}"

        # Find the alias for a source, if one exists
        def source_alias(source):
            for alias, alias_ident in self.data._aliases.items():
                if isinstance(alias_ident, str) and source == alias_ident:
                    return alias

            return None

        # Group all the aliases by source. The keys of this
        # dictionary can be either just the source name, or a
        # tuple of (alias, source). The values are a list of
        # tuples of (alias, key).
        source_key_aliases = defaultdict(list)
        for alias in self.data._aliases.keys():
            alias_ident = self.data._aliases[alias]

            if isinstance(alias_ident, tuple):
                source = alias_ident[0]
                if source_alias(source) is not None:
                    dict_key = (source_alias(source), source)
                else:
                    dict_key = source

                source_key_aliases[dict_key].append((alias, alias_ident[1]))
            elif isinstance(alias_ident, str):
                source_key_aliases[(alias, alias_ident)].extend([])

        if len(source_key_aliases) == 0:
            return "No aliases have been loaded."

        # Print links to the alias files
        output_lines = []
        n_files = len(self.file_paths())
        links = self.jhub_links()
        if n_files == 1:
            path = self.file_paths()[0]
            output_lines.append(f"Alias file: {links.get(path, path)}\n")
        elif n_files > 1:
            output_lines.append(f"Alias files:")
            for i, path in enumerate(self.file_paths()):
                line_end = "\n" if i == n_files - 1 else ""
                output_lines.append(f"- {links.get(path, path)}{line_end}")

        # Print the aliases
        output_lines.append("Loaded aliases:")
        for source, alias_keys in source_key_aliases.items():
            if len(alias_keys) == 0:
                # If there are no keys then this is a plain source alias
                alias, source = source
                output_lines.append(f"{alias_icon(alias)} {alias}: {source}")
            else:
                # Check if all the key aliases for the source are valid,
                # and use that to select an icon for the source
                keys_exists = [alias in self for alias, _ in alias_keys]
                if all(keys_exists):
                    source_icon = alias_icon(True)
                elif not any(keys_exists):
                    source_icon = alias_icon(False)
                else:
                    source_icon = "~"

                # Extract the source alias, if it exists
                if isinstance(source, tuple):
                    source_alias = source[0]
                    source = source[1]
                else:
                    source_alias = None

                # If a source has a single key alias, print it on one
                # line. Otherwise we print the keys indented under the source.
                if len(alias_keys) == 1:
                    alias, key = alias_keys[0]
                    output_lines.append(f"{alias_icon(alias)} {alias}: ({source}, {key})")
                else:
                    # If there's an alias, include it in the source header
                    if source_alias is None:
                        source_str = f"{source}"
                    else:
                        source_str = f"{source_alias} ({source})"
                    output_lines.append(f"{source_icon} {source_str}:")

                    for alias, key in alias_keys:
                        output_lines.append(f"  {alias_icon(alias)} {alias}: {key}")

            # Add a newline to the last line added. We can't add a newline by
            # itself because otherwise it would double up with other newlines
            # when being joined together at the end
            output_lines[-1] = output_lines[-1] + "\n"

        return "\n".join(output_lines)

    def __str__(self):
        return f"<extra_data.AliasIndexer with {len(self.data._aliases)} aliases>"

    def _resolve_aliased_selection(self, selection):
        if isinstance(selection, dict):
            res = {self._resolve_source_alias(alias): keys
                    for alias, keys in selection.items()}

        elif isinstance(selection, Iterable):
            res = []

            for item in selection:
                if isinstance(item, tuple) and len(item) == 2:
                    # Source alias and literal key.
                    item = (self._resolve_source_alias(item[0]), item[1])
                elif isinstance(item, str):
                    item = self._resolve_any_alias(item)

                    if isinstance(item, str):
                        # Source alias.
                        item = (item, '*')

                res.append(item)

        return res

    def file_paths(self):
        """Return any file paths that were used to add aliases."""
        return self.data._alias_files

    def jhub_links(self):
        """Return a dict of alias file paths to clickable Jupyterhub links.

        Note that a link will only be generated if the file path is under ``/gpfs``.
        """
        links = { }
        for p in self.file_paths():
            path_from_home = str(p).replace("/gpfs", "GPFS")
            if path_from_home.startswith("GPFS"):
                links[p] = f"https://max-jhub.desy.de/hub/user-redirect/lab/tree/{path_from_home}"

        return links

    def select(self, seln_or_alias, key_glob='*', require_all=False,
               require_any=False):
        """Select a subset of sources and keys from this data using aliases.

        This method is only accessible through the :attr:`DataCollection.alias`
        property.

        In contrast to :meth:`DataCollection.select`, only a subset of
        ways to select data via aliases is supported:

        1. With a source alias and literal key glob pattern::

            # Select all pulse energy keys for an aliased XGM fast data.
            sel = run.alias.select('sa1-xgm', 'data.intensity*')

        2. With an iterable of aliases and/or (source alias, key pattern) tuples::

            # Select specific keys for an aliased XGM fast data.
            sel = run.alias.select([('sa1-xgm', 'data.intensitySa1TD'),
                                    ('sa1-xgm', 'data.intensitySa3TD')]

            # Select several aliases, may be both source and key aliases.
            sel = run.alias.select(['sa1-xgm', 'mono-hv'])

           Data is included if it matches any of the aliases. Note that
           this method does not support glob patterns for the source alias.

        3. With a dict of source aliases mapped to sets of key names
           (or empty sets to get all keys)::

                # Select image.data from an aliased AGIPD and all data
                # from an aliased XGM.
                sel = run.select({'agipd': {'image.data'}, 'sa1-xgm': set()})

        The optional `require_all` and `require_any` arguments restrict the
        trains to those for which all or at least one selected sources and
        keys have at least one data entry. By default, all trains remain selected.

        Returns a new :class:`DataCollection` object for the selected data.
        """

        if isinstance(seln_or_alias, str):
            seln_or_alias = [(seln_or_alias, key_glob)]

        return self.data.select(self._resolve_aliased_selection(
            seln_or_alias), require_all=require_all, require_any=require_any)

    def deselect(self, seln_or_alias, key_glob='*'):
        """Select everything except the specified sources and keys using aliases.

        This method is only accessible through the :attr:`DataCollection.alias`
        property.

        This takes the same arguments as :meth:`select`, but the sources
        and keys you specify are dropped from the selection.

        Returns a new :class:`DataCollection` object for the remaining data.
        """

        if isinstance(seln_or_alias, str):
            seln_or_alias = [(seln_or_alias, key_glob)]

        return self.data.deselect(self._resolve_aliased_selection(
            seln_or_alias))
