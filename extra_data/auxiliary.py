
from itertools import groupby
import fnmatch
import re

from . import DataCollection, H5File, FileAccess, SourceData, SourceNameError


class AuxiliaryIndexer:
    """Enables item access to auxiliary sources."""

    def __init__(self, train_ids, files, is_single_run, inc_suspect_trains):
        # Resolve any voview files if present as they lack the auxiliary
        # sources present in their source files.
        files = self._resolve_voview_files(files)

        # {(source, section) -> {aggregator -> files}}
        files_by_sources = dict()

        for f in files:
            aggregator = f.aggregator

            for source in f.reduction_sources:
                files_by_sources.setdefault((source, 'REDUCTION'), dict()) \
                    .setdefault(aggregator, []).append(f)

            for source in f.errata_sources:
                files_by_sources.setdefault((source, 'ERRATA'), dict()) \
                    .setdefault(aggregator, []).append(f)

        self._sources_data = dict()

        for (src, section), files_by_aggregator in files_by_sources.items():
            # Prefix aggregator if necessary.
            with_prefix = len(files_by_aggregator) > 1

            self._sources_data.update({
                f'{aggregator}@{src}' if with_prefix else src: SourceData(
                    src, sel_keys=None, train_ids=train_ids,
                    files=files, section=section, canonical_name=src,
                    is_single_run=is_single_run,
                    inc_suspect_trains=inc_suspect_trains)
                for aggregator, files in files_by_aggregator.items()})

    @staticmethod
    def _resolve_voview_files(files):
        new_files = []

        for fa in files:
            if '.source_files' not in fa.file:
                new_files.append(fa)
                continue

            dc = H5File(fa.filename)

            source_file_paths = set()
            for source in dc.all_sources:
                sd = dc[source]
                source_file_paths |= set(sd[sd.one_key()].source_file_paths)

            for file in source_file_paths:
                new_files.append(FileAccess(file))

        return new_files

    @property
    def reduction_sources(self):
        return frozenset([src for src, sd in self._sources_data.items()
                          if sd.is_reduction])

    @property
    def errata_sources(self):
        return frozenset([src for src, sd in self._sources_data.items()
                          if sd.is_errata])

    @property
    def all_sources(self):
        return frozenset(self._sources_data.keys())

    def __contains__(self, item):
        return DataCollection.__contains__(self, item)

    def __str__(self):
        num_trains = len(next(iter(self._sources_data.values())).train_ids) \
            if self._sources_data else 0

        return f'<extra_data.AuxiliaryIndexer for {len(self.all_sources)} ' \
               f'sources and {num_trains} trains>'

    def _source_info(self, label, sources, details_for_sources=(),
                     with_aggregators=False):
        details_sources_re = [re.compile(fnmatch.translate(p))
                              for p in details_for_sources]
        info_str = ''

        def src_data_detail(s, keys, prefix=''):
            """Detail for how much data is present for an instrument group"""
            if not keys:
                return ''

            sd = self[s]
            counts = sd[list(keys)[0]].data_counts()
            ntrains_data = (counts > 0).sum()
            return f'{prefix}data for {ntrains_data} trains ' \
                   f'({ntrains_data / len(sd.train_ids):.2%}), ' \
                   f'up to {counts.max()} entries per train\n'

        def keys_detail(s, keys, prefix=''):
            """Detail for a group of keys"""
            keys_str = ''

            for k in keys:
                kd = self[s, k]
                entry_info = f', entry shape {kd.entry_shape}'  \
                    if kd.entry_shape else ''

                keys_str += f'{prefix}{k}\t[{kd.dtype}{entry_info}]\n'

            return keys_str

        info_str += f'{len(sources)} {label} sources:\n'

        for s in sorted(sources):
            agg_str = f' [{self[s].aggregator}]' if with_aggregators else ''
            info_str += f'  - {agg_str} {s}\n'
            if not any(p.match(s) for p in details_sources_re):
                continue

            # Detail for instrument sources:
            for group, keys in groupby(sorted(self[s].keys()),
                                       key=lambda k: k.split('.')[0]):
                info_str += f'    - {group}:\n'
                keys = list(keys)
                info_str += src_data_detail(s, keys, prefix='      ')
                info_str += keys_detail(s, keys, prefix='      - ')

        return info_str

    def __repr__(self, details_for_source=()):
        return self._source_info('reduction', self.reduction_sources) + \
            '\n' + self._source_info('errata', self.errata_sources)

    def info(self, details_for_sources=(), with_aggregators=False):
        print(self._source_info('reduction', self.reduction_sources,
                                details_for_sources, with_aggregators))
        print(self._source_info('errata', self.errata_sources,
                                 details_for_sources, with_aggregators))

    def __getitem__(self, item):
        if (
            isinstance(item, tuple) and
            len(item) == 2 and
            all(isinstance(e, str) for e in item)
        ):
            return self._get_key_data(*item)
        elif isinstance(item, str):
            return self._get_source_data(item)

        raise TypeError('Expected data.auxiliary[source] or '
                        'data.auxiliary[source, key]')

    def _get_source_data(self, source):
        if source not in self._sources_data:
            # Check whether this source may need to be prefixed with an
            # aggregator to display a more specific message.
            if source in {s.partition('@')[2] for s in self._sources_data}:
                msg = 'This data has one or more duplicates of an auxiliary ' \
                      f'source named {source!r}.\nSee ' \
                      'data.auxiliary.all_sources for available prefixes.'
            else:
                msg = f'This data has no auxiliary source named {source!r}.' \
                      '\nSee data.auxiliary.all_sources for available ' \
                      'auxiliary sources.'

            raise SourceNameError(source, msg)

        return self._sources_data[source]

    def _get_key_data(self, source, key):
        return self._get_source_data(source)[key]
