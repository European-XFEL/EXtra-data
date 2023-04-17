
from copy import copy
import re

import numpy as np

from euxfel_bunch_pattern import indices_at_sase, is_sase
from .. import by_id


try:
    from itertools import pairwise
except ImportError:
    # Compatibility for Python < 3.10
    from itertools import tee

    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


__all__ = [
    'PulseTiming'
]


class PulseTiming:
    """An interface to timing of FEL and PPL pulses.

    Args:
        data (DataCollection): Data to access timing data from.
        timeserver (str, optional): Source name of a timeserver, only
            needed if the data includes more than one timeserver or it
            could not be detected automatically.
        bam (str or bool, optional): Full or partial source name of a
            bunch arrival monitor to pick, only needed if the data
            includes more than one BAM or it could not be detected
            automatically. BAM support may be disabled entirely by
            passing False even when BAM data is present.
        sase (int, optional): SASE beamline at which this data was
            taken, only needed if the data includes sources from more
            than one beamline or it could not be detected automatically.
    """

    # Source prefixes in use at each SASE.
    _sase_topics = {
        1: {'SA1', 'LA1', 'SPB', 'FXE'},
        2: {'SA2', 'LA2', 'MID', 'HED'},
        3: {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'},
    }

    # Regular expressions for timeserver control and pipeline data.
    _timeserver_control_re = re.compile(
        r'^({})_(BR|RR)_(UTC|SYS)/TSYS/TIMESERVER$'.format(
            '|'.join(set.union(*_sase_topics.values()))))
    _timeserver_pipeline_re = re.compile(r'^{}:outputBunchPattern'.format(
        _timeserver_control_re.pattern[:-1]))

    # Class IDs for timeserver and BAM devices.
    _timeserver_class = 'TimeServer'
    _bam_class = 'DoocsBunchArrivalMonitors'

    def __init__(self, data, timeserver=None, bam=None, sase=None):
        if timeserver is None:
            timeserver = self._find_timeserver(data)

        self.timeserver_src = data[timeserver]

        if ':' in timeserver:
            self.bpt_key = self.timeserver_src['data.bunchPatternTable']
        else:
            self.bpt_key = self.timeserver_src['bunchPatternTable']

        if (bam is False) and (bam is None or bam not in data.all_sources):
            # BAM is not disabled and not given or only partially.
            bam = self._find_bam(data, bam or '')

        if bam:
            self.bam_src = data[bam]

            # The key for arrival times has been changed in the past
            # without change to the underlying format.
            for key in ['data.absoluteTD', 'data.lowChargeArrivalTime']:
                if key in self.bam_src:
                    self.bam_key = self.bam_src[key]
            else:
                raise ValueError(f'BAM source {bam} contains none of the '
                                 f'known keys for arrival time.')

        else:
            self.bam_src = None
            self.bam_key = None

        if sase not in {1, 2, 3}:
            sase = self._identify_sase(data)

        self.sase = sase

    def __repr__(self):
        return "<{} for SA{} using timeserver={}, bam={}>".format(
            type(self).__name__, self.sase, self.timeserver_src.source,
            self.bam_src.source if self.bam_src is not None else 'n/a')

    @classmethod
    def _find_timeserver(cls, data):
        """Try to find a timeserver souce.
        """

        # Try to go by device class first.
        # By the time the device class was recorded, time servers also
        # were changed to output the bunch pattern via a pipeline.
        timeserver_sources = {
            f'{source}:outputBunchPattern'
            for source in data.control_sources
            if (data[source].device_class == cls._timeserver_class and
                f'{source}:outputBunchPattern' in data.instrument_sources)
        }

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver sources found via device '
                             'class, please pass explicit one:\n' +
                             ', '.join(sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        # Next check for timeserver instrument data.
        for source in data.instrument_sources:
            m = cls._timeserver_pipeline_re.match(source)
            if m is not None:
                timeserver_sources.add(m[0])

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver instrument sources found, '
                             'please pass explicit one:\n' + ', '.join(
                                sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        # Last chance for timeserver control data, likely
        # empty for more recent data after 2020.
        for source in data.control_sources:
            m = cls._timeserver_control_re.match(source)
            if m is not None:
                timeserver_sources.add(m[0])

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver control sources found, '
                             'please pass explicit one:\n' + ', '.join(
                                sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        raise ValueError('no timeserver found, please pass explicit source')

    @classmethod
    def _find_bam(cls, data, needle):
        """Try to find a BAM source.
        """

        # Try to go by device class first.
        bam_sources = {source for source in data.control_sources
                       if (data[source].device_class == cls._bam_class and
                           (not needle or needle in source))}

        # And try something by name second.
        for source in data.instrument_sources:
            maybe_bam = 'BAM' in source and 'DOOCS' in source

            if maybe_bam and (not needle or needle in source):
                # Add this source as a BAM candidate if
                #   a) it looks like one and
                #   b) either no needle was given or needle matches.
                bam_sources.add(source)

        if len(bam_sources) > 1:
            raise ValueError('multiple potential BAM instrument sources '
                             'found, please pass partial name or explicit '
                             'one:\n' + ', '.join(sorted(bam_sources)))
        elif bam_sources:
            return bam_sources.pop()

    @classmethod
    def _identify_sase(cls, data):
        """Try to identify which SASE this data belongs to.
        """

        sases = {sase
                 for src in data.all_sources
                 for sase, topics in cls._sase_topics.items()
                 if src[:src.find('_')] in topics}

        if len(sases) == 1:
            return sases.pop()
        elif sases == {1, 3}:
            # SA3 data often contains one or more SA1 sources
            # from shared upstream components.
            return 3
        else:
            raise ValueError('sources from ambiguous sase branches {} found, '
                             'please pass explicit sase'.join(
                                ', '.join(sases)))

    @staticmethod
    def _search_pulse_pattern_regions(pulse_mask, tids=None):
        """Search pulse index pattern regions.
        """

        # Find the unique patterns and the respective indices for each
        # unique pattern.
        unique_patterns, pattern_indices = np.unique(
            pulse_mask, axis=0, return_inverse=True)

        # Find positions of pattern changes plus beginning and end.
        pattern_changes = np.concatenate([
            [-1],
            (pattern_indices[1:] - pattern_indices[:-1]).nonzero()[0],
            [len(pulse_mask)]
        ])

        if tids is not None:
            one = np.uint64(1)  # Avoid conversion to float64.

            def gen_slice(start, stop):
                return by_id[tids[start]:tids[stop-1]+one]
        else:
            def gen_slice(start, stop):
                return np.s_[start:stop]

        # Build list of (train_slice, pattern) tuples.
        patterns = [
            (gen_slice(start+1, stop), pulse_mask[start+1].nonzero()[0])
            for start, stop in pairwise(pattern_changes)
        ]

        return patterns

    @staticmethod
    def _filter_bam(arrival_times, pulse_indices):
        """Pick bunch arrival times for given pulse indices.
        """

        return arrival_times[..., 2 * pulse_indices]

    def select_trains(self, trains):
        """Select a subset of trains in this data.
        """

        res = copy(self)

        res.timeserver_src = self.timeserver_src.select_trains(trains)
        res.bpt_key = self.bpt_key.select_trains(trains)

        if self.bam_src is not None:
            res.bam_src = self.bam_src.select_trains(trains)
            res.bam_key = self.bam_key.select_trains(trains)

        return res

    def align_trains(self):
        """Align timeserver and BAM trains in this data.
        """

        # Collect a list of train ID lists to be aligned.
        tids = [set(self.bpt_key.drop_empty_trains().train_ids)]

        if self.bam_key is not None:
            tids.append(set(self.bam_key.drop_empty_trains().train_ids))

        # Select the intersection of all train ID lists.
        return self.select_trains(by_id[list(set.intersection(*tids))])

    def drop_bam(self):
        """Drop BAM source.
        """

        new_self = copy(self)
        new_self.bam_src = None
        new_self.bam_key = None

        return new_self

    def get_pulse_indices(self, pattern_search=False, by_id=False):
        """Get pulse indices in this data.

        Reads the bunch pattern table and gathers the indices for pulses
        that went to this SASE branch.
        When pattern_search is False (default), an exception is raised
        if pulse indices are not constant over all trains. When enabled,
        contiguous train regions of constant pulse pattern are searched
        instead and returned as a list of train slices and correspondig
        pulse indices.

        Args:
            pattern_search (bool, optional): Whether to only return a
                single constant pulse indices result (default) or search
                for contiguous train regions of constant indices.
            by_id (bool, optional): Whether to express train slices in a
                full search by index (default) or train ID, ignored when
                pattern_search is False.

        Returns:
            (ndarray) Pulse indices.
                (only returned if pattern_search is False)

            (list of (slice, ndarray) tuples) List of train regions
                identified by index slices with identical pulse
                patterns.
                (only returned if pattern_search is True)

        Raises:
            ValueError: pulse indices not constant across data
                (only raised if pattern_search is False)
        """

        pulse_mask = is_sase(self.bpt_key.ndarray(), sase=self.sase)

        if pattern_search:
            return self._search_pulse_pattern_regions(
                pulse_mask,
                self.bpt_key.train_id_coordinates() if by_id else None)
        else:
            if (pulse_mask != pulse_mask[0]).any():
                raise ValueError('pulse indices not constant across data')

            return pulse_mask[0].nonzero()[0]

    def get_bunch_arrival_times(self, labelled=False, pattern_search=False,
                                by_id=False):
        """Get bunch arrival times.

        Args:
            labelled (bool, optional): Whether to return an unlabelled
                np.ndarray (default) or labelled xarray.DataArray with
                train ID coordinates.
            pattern_search (bool, optional): Whether to accept constant
                pulse indices for all trains (default) or search for
                contiguous train regions of constant indices.
            by_id (bool, optional): Whether to express train slices in a
                full search by index (default) or train ID, ignored when
                pattern_search is False.

        Returns:
            (ndarray) Bunch arrival times.
                (only returned if pattern_search is False)

            (list of (slice, ndarray) tuples) List of bunch arrival
                times for train regions identified by index slices with
                identical pulse patterns.
                (only returned if pattern_search is True)

        Raises:
            ValueError: no BAM source configured.
            ValueError: pulse indices not constant across data
                (only raised if pattern_search is False)
        """

        if self.bam_key is None:
            raise ValueError('no BAM source configured')

        if labelled:
            def _get(obj):
                return obj.xarray()
        else:
            def _get(obj):
                return obj.ndarray()

        if pattern_search:
            # With pattern search enabled, always use aligned trains.
            self = self.align_trains()

        pulses = self.get_pulse_indices(
            pattern_search=pattern_search, by_id=by_id)

        if pattern_search:
            return [
                (region, self._filter_bam(_get(self.bam_key[region]), indices))
                for region, indices in pulses
            ]
        else:
            return self._filter_bam(_get(self.bam_key), pulses)

    def trains(self):
        """Iterate over pulse indices (and bunch arrival times) by train.

        Yields:
            (int, ndarray) Train ID and pulse indices, if no BAM data is
                present.

            (int, ndarray, ndarray) Train ID, pulse indices and bunch
                arrival times, if BAM data is present.
        """

        if self.bam_key is not None:
            # Always use aligned trains with BAM.
            self = self.align_trains()
            it = zip(self.bpt_key.trains(),
                     self.bam_key.trains())

            for (train_id, table), (_, bam) in it:
                indices = indices_at_sase(table, sase=self.sase)
                yield train_id, indices, self._filter_bam(bam, indices)
        else:
            for train_id, table in self.bpt_key.trains():
                yield train_id, indices_at_sase(table, sase=self.sase)
