import fnmatch
import re
from collections import defaultdict
from datetime import timedelta
from itertools import groupby

from .reader import DataCollection
from .read_machinery import DETECTOR_SOURCE_RE
from .sourcedata import SourceData

def info(dc: DataCollection, details_for_sources=(), with_aggregators=False,
         with_auxiliary=False):
    """Show information about the selected data."""
    InfoPrinter(dc, details_for_sources, with_aggregators).show(with_auxiliary)


class InfoPrinter:
    def __init__(self, dc: DataCollection, details_for_sources=(), with_aggregators=False):
        self.dc = dc
        self.details_for_sources = details_for_sources
        self.details_sources_re = [re.compile(fnmatch.translate(p))
                                   for p in details_for_sources]
        self.with_aggregators = with_aggregators

        # Invert aliases for faster lookup.
        self.src_aliases = defaultdict(set)
        self.srckey_aliases = defaultdict(lambda: defaultdict(set))

        for alias, literal in dc._aliases.items():
            if isinstance(literal, str):
                self.src_aliases[literal].add(alias)
            else:
                self.srckey_aliases[literal[0]][literal[1]].add(alias)

    def trains(self):
        # time info
        train_count = len(self.dc.train_ids)
        if train_count == 0:
            first_train = last_train = '-'
            span_txt = '0.0'
        else:
            first_train = self.dc.train_ids[0]
            last_train = self.dc.train_ids[-1]
            seconds, deciseconds = divmod((last_train - first_train + 1), 10)
            try:
                td = timedelta(seconds=int(seconds))
            except OverflowError:  # Can occur if a train ID is corrupted
                span_txt = "OverflowError (one or more train IDs are probably wrong)"
            else:
                span_txt = f'{td}.{int(deciseconds)}'

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

    def xtdf(self):
        sources_by_detector = {}
        for source in self.dc.detector_sources:
            name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
            sources_by_detector.setdefault(name, {})[modno] = source

        for detector_name in sorted(sources_by_detector.keys()):
            detector_modules = sources_by_detector[detector_name]

            print("{} XTDF detector modules of {}/*".format(
                len(detector_modules), detector_name
            ))
            if len(detector_modules) > 0:
                # Show detail on the first module (the others should be similar)
                mod_key = sorted(detector_modules)[0]
                mod_source = detector_modules[mod_key]
                dinfo = self.dc.detector_info(mod_source)
                module = ' '.join(mod_key)
                dims = ' x '.join(str(d) for d in dinfo['dims'])
                print("  e.g. module {} : {} pixels".format(module, dims))
                print("  {}".format(mod_source))
                print("  {} frames per train, up to {} frames total".format(
                    dinfo['frames_per_train'], dinfo['total_frames']
                ))
            print()

    def src_alias_list(self, s):
        if self.src_aliases[s]:
            alias_str = ', '.join(self.src_aliases[s])
            return f'<{alias_str}>'
        return ''

    def inst_sources(self):
        srcs = self.dc.instrument_sources
        if self.details_sources_re:
            # All instrument sources with details enabled.
            displayed_inst_srcs = srcs - self.dc.legacy_sources.keys()
            print(len(displayed_inst_srcs), 'instrument sources:')
        else:
            # Only non-XTDF instrument sources without details enabled.
            displayed_inst_srcs = (
                srcs - self.dc.detector_sources - self.dc.legacy_sources.keys()
            )
            print(len(displayed_inst_srcs), 'instrument sources (excluding XTDF detectors):')

        for s in sorted(displayed_inst_srcs):
            sd = self.dc[s]
            agg_str = f' [{sd.aggregator}]' if self.with_aggregators else ''
            print('  -' + agg_str, s, self.src_alias_list(s))
            if not any(p.match(s) for p in self.details_sources_re):
                continue

            sif = SourceInfoFormatter(sd, self.srckey_aliases[s])

            # Detail for instrument sources:
            for group, keys in groupby(sorted(sd.keys()),
                                       key=lambda k: k.split('.')[0]):
                print(f'    - {group}:')
                keys = list(keys)
                print('      ' + sif.src_data_detail(keys))
                for l in sif.keys_detail(keys):
                    print("      " + l)
        print()

    def ctrl_sources(self):
        print(len(self.dc.control_sources), 'control sources:')
        for s in sorted(self.dc.control_sources):
            sd = self.dc[s]
            agg_str = f' [{sd.aggregator}]' if self.with_aggregators else ''
            print('  -' + agg_str, s, self.src_alias_list(s))
            if any(p.match(s) for p in self.details_sources_re):
                sif = SourceInfoFormatter(sd, self.srckey_aliases[s])
                # Detail for control sources: list keys
                print('    - Control keys (1 entry per train):')
                for l in sif.keys_detail():
                    print("      " + l)

                if rok_detail := list(sif.run_only_keys_detail()):
                    print('    - Additional run keys (1 entry per run):')
                    for l in rok_detail:
                        print("      " + l)
        print()

    def legacy_sources(self):
        # Collect legacy souces matching DETECTOR_SOURCE_RE
        # separately for a condensed view.
        detector_legacy_sources = defaultdict(set)

        print(len(self.dc.legacy_sources), 'legacy source names:')
        for s in sorted(self.dc.legacy_sources.keys()):
            m = DETECTOR_SOURCE_RE.match(s)

            if m is not None:
                detector_legacy_sources[m[1]].add(s)
            else:
                # Only print non-XTDF legacy sources.
                print(' -', s, '->', self.dc.legacy_sources[s])

        for legacy_det, legacy_sources in detector_legacy_sources.items():
            canonical_mod = self.dc.legacy_sources[next(iter(legacy_sources))]
            canonical_det = DETECTOR_SOURCE_RE.match(canonical_mod)[1]

            print(' -', f'{legacy_det}/*', '->', f'{canonical_det}/*',
                  f'({len(legacy_sources)})')
        print()

    def show(self, with_auxiliary=False):
        self.trains()
        if not self.details_sources_re:
            self.xtdf()
        self.inst_sources()
        self.ctrl_sources()
        if self.dc.legacy_sources:
            self.legacy_sources()
        if with_auxiliary:
            self.dc.auxiliary.info(
                details_for_sources=self.details_for_sources,
                with_aggregators=self.with_aggregators
            )


class SourceInfoFormatter:
    def __init__(self, src: SourceData, key_aliases=None):
        self.src = src
        self.key_aliases = key_aliases or {}

    def src_data_detail(self, keys):
        """Detail for how much data is present for an instrument group"""
        if not keys:
            return
        counts = self.src[list(keys)[0]].data_counts()
        ntrains_data = (counts > 0).sum()
        return (
            f'data for {ntrains_data} trains '
            f'({ntrains_data / len(self.src.train_ids):.2%}), '
            f'up to {counts.max()} entries per train'
        )

    def keys_detail(self, keys=None):
        """Detail for a group of keys"""
        if keys is None:
            keys = sorted(self.src.keys(inc_timestamps=False))

        for k in keys:
            kd = self.src[k]

            if aliases := self.key_aliases.get(k):
                alias_str = ' <' + ', '.join(sorted(aliases)) + '>'
            else:
                alias_str = ''

            entry_info = f", entry shape {kd.entry_shape}" if kd.entry_shape else ""
            yield f"- {k}{alias_str}\t[{kd.dtype}{entry_info}]"

    def run_only_keys_detail(self):
        run_values = self.src.run_values(inc_timestamps=False)
        run_only_keys = set(run_values) - self.src.keys(inc_timestamps=False)

        for k in sorted(run_only_keys):
            if aliases := self.key_aliases.get(k):
                alias_str = ' <' + ', '.join(aliases) + '>'
            else:
                alias_str = ''

            val = run_values[k]
            if isinstance(val, str):
                shape = ()
                dt = 'string'
            else:
                shape = val.shape
                dt = val.dtype
            entry_info = f", entry shape {val.shape}" if shape else ""
            yield f"- {k}{alias_str}\t[{dt}{entry_info}]"
