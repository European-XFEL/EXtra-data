
import pytest

from extra_data import RunDirectory, SourceNameError


def test_auxiliary_info(mock_reduced_spb_raw_run):
    run = RunDirectory(mock_reduced_spb_raw_run)

    # Smoke tests
    str(run.auxiliary)
    repr(run.auxiliary)
    run.auxiliary.info()
    run.auxiliary.info(details_for_sources=['SPB*'])
    run.auxiliary.info(with_aggregators=True)


def test_auxiliary_sources(mock_reduced_spb_raw_run):
    run = RunDirectory(mock_reduced_spb_raw_run)

    assert len(run.auxiliary.reduction_sources) == 32
    assert 'PULSE_REDUCTION/SPB_DET_AGIPD1M-1/DET/3CH0:xtdf' \
        in run.auxiliary.reduction_sources
    assert 'AGIPD11@SPB_IRU_AGIPD1M1/REDU/LITFRM:daqFilter' \
        in run.auxiliary.reduction_sources
    assert run.auxiliary.errata_sources == {'DA01@TRAINS_OUTSIDE_BUFFER_RANGE'}
    assert len(run.auxiliary.all_sources) == 33

    sd = run.auxiliary['DA01@TRAINS_OUTSIDE_BUFFER_RANGE']
    assert len(sd.files) == 2
    assert sd.section == 'ERRATA'
    assert sd.source == 'TRAINS_OUTSIDE_BUFFER_RANGE'
    assert not sd.is_reduction and sd.is_errata and sd.is_auxiliary
    assert not sd.is_control and not sd.is_instrument and not sd.is_run_only
    assert sd.index_groups == {'event'}
    assert 'event.deviceIds' in sd
    assert sd.keys() == {'event.deviceIds', 'event.originalTrainIds',
                           'event.properties', 'event.types', 'event.values'}
    assert sd['event.deviceIds'].entry_shape == (11,)

    sd = run.auxiliary['AGIPD11@SPB_IRU_AGIPD1M1/REDU/LITFRM:daqFilter']
    assert sd.is_reduction and not sd.is_errata and sd.is_auxiliary

    with pytest.raises(SourceNameError):
        run.auxiliary['ABC']

    with pytest.raises(SourceNameError):
        run.auxiliary['SPB_IRU_AGIPD1M1/REDU/LITFRM:daqFilter']
