import fnmatch
import re
from collections import defaultdict
from datetime import timedelta
from itertools import groupby

import h5py

from .reader import DataCollection
from .read_machinery import DETECTOR_SOURCE_RE

def info(dc: DataCollection, details_for_sources=(), with_aggregators=False,
         with_auxiliary=False):
    """Show information about the selected data."""
    details_sources_re = [re.compile(fnmatch.translate(p))
                          for p in details_for_sources]

    # time info
    train_count = len(dc.train_ids)
    if train_count == 0:
        first_train = last_train = '-'
        span_txt = '0.0'
    else:
        first_train = dc.train_ids[0]
        last_train = dc.train_ids[-1]
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

    if not details_for_sources:
        # Include summary section for multi-module detectors unless
        # source details are enabled.

        sources_by_detector = {}
        for source in dc.detector_sources:
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
                dinfo = dc.detector_info(mod_source)
                module = ' '.join(mod_key)
                dims = ' x '.join(str(d) for d in dinfo['dims'])
                print("  e.g. module {} : {} pixels".format(module, dims))
                print("  {}".format(mod_source))
                print("  {} frames per train, up to {} frames total".format(
                    dinfo['frames_per_train'], dinfo['total_frames']
                ))
            print()

    # Invert aliases for faster lookup.
    src_aliases = defaultdict(set)
    srckey_aliases = defaultdict(lambda: defaultdict(set))

    for alias, literal in dc._aliases.items():
        if isinstance(literal, str):
            src_aliases[literal].add(alias)
        else:
            srckey_aliases[literal[0]][literal[1]].add(alias)

    def src_alias_list(s):
        if src_aliases[s]:
            alias_str = ', '.join(src_aliases[s])
            return f'<{alias_str}>'
        return ''

    def src_data_detail(s, keys, prefix=''):
        """Detail for how much data is present for an instrument group"""
        if not keys:
            return
        counts = dc.get_data_counts(s, list(keys)[0])
        ntrains_data = (counts > 0).sum()
        print(
            f'{prefix}data for {ntrains_data} trains '
            f'({ntrains_data / train_count:.2%}), '
            f'up to {counts.max()} entries per train'
        )

    def keys_detail(s, keys, prefix=''):
        """Detail for a group of keys"""
        for k in keys:
            entry_shape = dc.get_entry_shape(s, k)
            if entry_shape:
                entry_info = f", entry shape {entry_shape}"
            else:
                entry_info = ""
            dt = dc.get_dtype(s, k)

            if k in srckey_aliases[s]:
                alias_str = ' <' + ', '.join(srckey_aliases[s][k]) + '>'
            else:
                alias_str = ''

            print(f"{prefix}{k}{alias_str}\t[{dt}{entry_info}]")

    if details_for_sources:
        # All instrument sources with details enabled.
        displayed_inst_srcs = dc.instrument_sources - dc.legacy_sources.keys()
        print(len(displayed_inst_srcs), 'instrument sources:')
    else:
        # Only non-XTDF instrument sources without details enabled.
        displayed_inst_srcs = dc.instrument_sources - dc.detector_sources - dc.legacy_sources.keys()
        print(len(displayed_inst_srcs), 'instrument sources (excluding XTDF detectors):')

    for s in sorted(displayed_inst_srcs):
        agg_str = f' [{dc[s].aggregator}]' if with_aggregators else ''
        print('  -' + agg_str, s, src_alias_list(s))
        if not any(p.match(s) for p in details_sources_re):
            continue

        # Detail for instrument sources:
        for group, keys in groupby(sorted(dc.keys_for_source(s)),
                                   key=lambda k: k.split('.')[0]):
            print(f'    - {group}:')
            keys = list(keys)
            src_data_detail(s, keys, prefix='      ')
            keys_detail(s, keys, prefix='      - ')

    print()
    print(len(dc.control_sources), 'control sources:')
    for s in sorted(dc.control_sources):
        agg_str = f' [{dc[s].aggregator}]' if with_aggregators else ''
        print('  -' + agg_str, s, src_alias_list(s))
        if any(p.match(s) for p in details_sources_re):
            # Detail for control sources: list keys
            ctrl_keys = dc[s].keys(inc_timestamps=False)
            print('    - Control keys (1 entry per train):')
            keys_detail(s, sorted(ctrl_keys), prefix='      - ')

            run_keys = dc._sources_data[s].files[0].get_run_keys(s)
            run_keys = {k[:-6] for k in run_keys if k.endswith('.value')}
            run_only_keys = run_keys - ctrl_keys
            if run_only_keys:
                print('    - Additional run keys (1 entry per run):')
                for k in sorted(run_only_keys):
                    if k in srckey_aliases[s]:
                        alias_str = ' <' + ', '.join(srckey_aliases[s][k]) + '>'
                    else:
                        alias_str = ''

                    ds = dc._sources_data[s].files[0].file[
                        f"/RUN/{s}/{k.replace('.', '/')}/value"
                    ]
                    entry_shape = ds.shape[1:]
                    if entry_shape:
                        entry_info = f", entry shape {entry_shape}"
                    else:
                        entry_info = ""
                    dt = ds.dtype
                    if h5py.check_string_dtype(dt):
                        dt = 'string'
                    print(f"      - {k}{alias_str}\t[{dt}{entry_info}]")

    print()

    if dc.legacy_sources:
        # Collect legacy souces matching DETECTOR_SOURCE_RE
        # separately for a condensed view.
        detector_legacy_sources = defaultdict(set)

        print(len(dc.legacy_sources), 'legacy source names:')
        for s in sorted(dc.legacy_sources.keys()):
            m = DETECTOR_SOURCE_RE.match(s)

            if m is not None:
                detector_legacy_sources[m[1]].add(s)
            else:
                # Only print non-XTDF legacy sources.
                print(' -', s, '->', dc.legacy_sources[s])

        for legacy_det, legacy_sources in detector_legacy_sources.items():
            canonical_mod = dc.legacy_sources[next(iter(legacy_sources))]
            canonical_det = DETECTOR_SOURCE_RE.match(canonical_mod)[1]

            print(' -', f'{legacy_det}/*', '->', f'{canonical_det}/*',
                  f'({len(legacy_sources)})')
        print()

    if with_auxiliary:
        dc.auxiliary.info(details_for_sources=details_for_sources,
                            with_aggregators=with_aggregators)
