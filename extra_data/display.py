import fnmatch
import re
from collections import defaultdict
from collections.abc import Callable
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

    def source_line(self, s):
        agg_str = f" [{self.dc[s].aggregator}]" if self.with_aggregators else ""
        print(f"  -{agg_str} {s} {self.src_alias_list(s)}")

    def list_sources(self, srcs: list, detail: Callable):
        current_group = SourceGroup()
        def flush_group():
            nonlocal current_group
            if current_group:
                print(f"  - {current_group}")
            current_group = SourceGroup()

        for s in sorted(srcs):
            show_detail = any(p.match(s) for p in self.details_sources_re)
            show_solo = show_detail or self.with_aggregators or self.src_aliases[s]
            if show_solo:
                flush_group()
                self.source_line(s)
                if show_detail:
                    detail(s)
            else:
                if not current_group.add(s):
                    flush_group()
                    current_group.add(s)
        flush_group()

    def inst_sources(self):
        srcs = self.dc.instrument_sources
        if self.details_sources_re:
            # All instrument sources with details enabled.
            displayed_inst_srcs = srcs - self.dc.legacy_sources.keys()
            print(len(displayed_inst_srcs), "instrument sources:")
        else:
            # Only non-XTDF instrument sources without details enabled.
            displayed_inst_srcs = (
                srcs - self.dc.detector_sources - self.dc.legacy_sources.keys()
            )
            print(len(displayed_inst_srcs), "instrument sources (excluding XTDF detectors):")

        self.list_sources(displayed_inst_srcs, self.inst_detail)
        print()

    def inst_detail(self, s):
        # Detail for instrument sources:
        sd = self.dc[s]
        sif = SourceInfoFormatter(self.dc[s], self.srckey_aliases[s])
        for group, keys in groupby(sorted(sd.keys()), key=lambda k: k.split(".")[0]):
            print(f"    - {group}:")
            keys = list(keys)
            print("      " + sif.src_data_detail(keys))
            for l in sif.keys_detail(keys):
                print("      " + l)

    def ctrl_sources(self):
        print(len(self.dc.control_sources), "control sources:")
        self.list_sources(self.dc.control_sources, self.ctrl_detail)
        print()

    def ctrl_detail(self, s):
        sif = SourceInfoFormatter(self.dc[s], self.srckey_aliases[s])
        # Detail for control sources: list keys
        print("    - Control keys (1 entry per train):")
        for l in sif.keys_detail():
            print("      " + l)

        if rok_detail := list(sif.run_only_keys_detail()):
            print("    - Additional run keys (1 entry per run):")
            for l in rok_detail:
                print("      " + l)

    def legacy_sources(self):
        # Collect legacy sources matching DETECTOR_SOURCE_RE
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

            print(f" - {legacy_det}/* -> {canonical_det}/* ({len(legacy_sources)})")
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
                with_aggregators=self.with_aggregators,
            )


class SourceGroup:
    _piece_re = re.compile(r"(\d+|[a-zA-Z]+)")

    def __init__(self):
        self.entries = []

    def __bool__(self):
        return bool(self.entries)

    @classmethod
    def split_name(cls, name: str) -> list[str]:
        """Split up a source name for possible grouping

        Based on the behaviour of re.split(), elements 1, 3, 5... in the result
        are the pieces where variations might make a group, and 0, 2, 4... must
        match exactly. Grouping is typically on alphabetic & numeric pieces,
        with some heuristics to avoid over-grouping. The list always has an
        odd number of elements, starting and ending with separator parts.
        """
        components = name.split("/")
        components[-1], _, pipeline = components[-1].partition(':')
        res = cls._piece_re.split(components[0])
        for c in components[1:]:
            head, *tail = cls._piece_re.split(c)
            # The first part is always a separator or empty string. Combine it
            # with the previous separator/empty string.
            res[-1] += "/" + head
            match tail:
                case []:  # Component empty or just a separator
                    pass
                case [s, '']:
                    # The whole component is a simple name/number.
                    # Try grouping only on short names (e.g. 'X', 'Y') & numbers.
                    if len(s) <= 2 or s.isdigit():
                        res.extend(tail)
                    else:
                        res[-1] += s  # Add to a non-grouping part
                case _:
                    # Component has >1 name/number parts we can group on
                    res.extend(tail)
        if pipeline:
            head, *tail = cls._piece_re.split(pipeline)
            res[-1] += ":" + head
            res.extend(tail)
        return res

    def add(self, name):
        """Try to add a name to the group. Returns True if it's added."""
        split = self.split_name(name)  # parts 1, 3, ... are matches
        if not self.entries:
            self.entries = [split]
            return True

        grp = self.entries
        if len(split) != len(grp[0]):
            return False

        diffs_at = [i for i, (p1, p2) in enumerate(zip(grp[-1], split)) if p1 != p2]
        if len(diffs_at) != 1 or ((diff_at := diffs_at[0]) % 2 == 0):
            return False  # >1 part changed, or separator part changed

        if len(grp) > 2 and (grp[-1][diff_at] == grp[0][diff_at]):
            return False  # different part changing from current group

        self.entries.append(split)
        return True

    def is_numeric(self):
        """Whether the varying part holds numbers"""
        e = self.entries
        if len(e) < 2:
            return False
        diff_at = [i for i, (p1, p2) in enumerate(zip(e[0], e[1])) if p1 != p2][0]
        return all(parts[diff_at].isdigit() for parts in e)

    def num_ranges(self, part_set):
        """Find contiguous ranges of numeric parts

        Returns a list of [start, end] pairs: '1' '2' '3' '5' -> [1 3] [5 5]
        """
        # We keep the parts as strings, to preserve '01' vs '1'
        pl = sorted(part_set, key=int)
        ranges = [[pl[0], pl[0]]]
        for part in pl[1:]:
            if int(part) == int(ranges[-1][-1]) + 1:
                ranges[-1][-1] = part
            else:
                ranges.append([part, part])

        return [
            f"{start}" if start == end else f"{start}-{end}"
            for (start, end) in ranges
        ]

    def __str__(self):
        res = []
        for matched_parts in zip(*self.entries):
            if len(ps := set(matched_parts)) == 1:
                res.append(ps.pop())
            else:
                if self.is_numeric():
                    pl = self.num_ranges(ps)
                else:
                    pl = sorted(ps)
                res.append("{" + ", ".join(pl) + "}")
        return "".join(res)


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
