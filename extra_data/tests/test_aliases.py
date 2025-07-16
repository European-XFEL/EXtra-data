from itertools import islice

import pytest
import numpy as np

from extra_data import (
    H5File, KeyData, by_index, by_id,
    AliasError, SourceNameError, PropertyNameError
)

def test_with_aliases(mock_sa3_control_data, mock_sa3_control_aliases, mock_sa3_control_aliases_yaml):
    run_without_aliases = H5File(mock_sa3_control_data)
    run = run_without_aliases.with_aliases(mock_sa3_control_aliases)

    # There should be no files since we're adding aliases explicitly as a dict
    assert len(run._alias_files) == 0

    def assert_equal_keydata(kd1, kd2):
        assert isinstance(kd1, KeyData)
        assert isinstance(kd2, KeyData)
        assert kd1.source == kd2.source
        assert kd1.key == kd2.key
        assert kd1.train_ids == kd2.train_ids

    # Test whether source alias yields identical SourceData.
    assert run.alias['sa3-xgm'] is run['SA3_XTD10_XGM/XGM/DOOCS']

    # Test alternative capitalisation and _ instead of -
    assert run.alias['SA3_XGM'] is run['SA3_XTD10_XGM/XGM/DOOCS']

    # Test __contains__()
    assert "sa3-xgm" in run.alias
    assert not "sa42-xgm" in run.alias
    with pytest.raises(TypeError):
        42 in run.alias

    # Test whether source alias plus literal key yields equal KeyData.
    assert_equal_keydata(
        run.alias['sa3-xgm', 'pulseEnergy.wavelengthUsed'],
        run['SA3_XTD10_XGM/XGM/DOOCS', 'pulseEnergy.wavelengthUsed'])

    # Test whether key alias yields equal KeyData.
    assert_equal_keydata(
        run.alias['hv'],
        run['SA3_XTD10_XGM/XGM/DOOCS', 'pulseEnergy.wavelengthUsed'])

    # Test undefined aliases.
    with pytest.raises(AliasError):
        run.alias['foo']
        run.alias['foo', 'bar']

    # Test using a literal key with a key alias.
    with pytest.raises(ValueError):
        run.alias['hv', 'pressure']

    # Test using an existing source alias for a non-existing source.
    with pytest.raises(SourceNameError):
        run.alias['bogus-source']

    # Test using an existing key alias for a non-existing key.
    with pytest.raises(PropertyNameError):
        run.alias['bogus-key']

    # Test re-applying the same aliases.
    run2 = run.with_aliases(mock_sa3_control_aliases)
    assert run._aliases == run2._aliases

    # Test adding additional aliases.
    run3 = run.with_aliases({'foo': 'bar'})
    assert set(run._aliases.keys()) < set(run3._aliases.keys())
    assert 'foo' in run3._aliases

    # Test adding conflicting aliases
    with pytest.raises(ValueError):
        run.with_aliases({'sa3-xgm': 'x'})

    # Test dropping aliases again.
    run4 = run.drop_aliases()
    assert not run4._aliases

    # Test that file paths are recorded and dropped appropriately
    run = run_without_aliases.with_aliases(mock_sa3_control_aliases_yaml)
    assert run._alias_files == [mock_sa3_control_aliases_yaml]
    assert run.drop_aliases()._alias_files == []

    # Smoke tests for __str__() and __repr__()
    assert "Loaded aliases" in repr(run.alias)
    assert "No aliases" in repr(run_without_aliases.alias)
    str(run.alias)

    # Add a dummy alias file and check that the link is shown by repr()
    run._alias_files[0] = "/gpfs/foo.yaml"
    assert "Alias file: https://max-jhub.desy" in repr(run.alias)
    run._alias_files.append("/gpfs/bar.yaml")
    assert "Alias files:" in repr(run.alias)


def test_alias_clash(mock_sa3_control_data, mock_sa3_control_aliases):
    run_without_aliases = H5File(mock_sa3_control_data)
    # The aliases include 'mcp-adc' - test with an equivalent name
    with pytest.raises(ValueError, match='conflicting alias'):
        mock_sa3_control_aliases.update({'MCP_ADC': 'SA3_XTD10_MCP/ADC/2'})
        run_without_aliases.with_aliases(mock_sa3_control_aliases)


def test_json_alias_file(mock_sa3_control_data, mock_sa3_control_aliases, tmp_path):
    aliases_path = tmp_path / 'aliases.json'
    aliases_path.write_text('''
{
    "sa3-xgm": "SA3_XTD10_XGM/XGM/DOOCS",
    "SA3_XTD10_XGM/XGM/DOOCS": {
        "hv": "pulseEnergy.wavelengthUsed",
        "beam-x": "beamPosition.ixPos",
        "beam-y": "beamPosition.iyPos"
    },

    "imgfel-frames": ["SA3_XTD10_IMGFEL/CAM/BEAMVIEW:daqOutput", "data.image.pixels"],
    "imgfel-frames2": ["SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput", "data.image.pixels"],
    "imgfel-screen-pos": ["SA3_XTD10_IMGFEL/MOTOR/SCREEN", "actualPosition"],
    "imgfel-filter-pos": ["SA3_XTD10_IMGFEL/MOTOR/FILTER", "actualPosition"],

    "mcp-adc": "SA3_XTD10_MCP/ADC/1",
    "mcp-mpod": "SA3_XTD10_MCP/MCPS/MPOD",
    "mcp-voltage": ["SA3_XTD10_MCP/MCPS/MPOD", "channels.U3.voltage"],
    "mcp-trace": ["SA3_XTD10_MCP/ADC/1:channel_5.output", "data.rawData"],

    "bogus-source": "SA4_XTD20_XGM/XGM/DOOCS",
    "bogus-key": ["SA3_XTD10_XGM/XGM/DOOCS", "foo"]
}
    ''')

    run = H5File(mock_sa3_control_data).with_aliases(aliases_path)
    assert run._aliases == mock_sa3_control_aliases

    # Since we're loading from a file the file path should be recorded
    assert run._alias_files == [aliases_path]


def test_yaml_alias_file(mock_sa3_control_data, mock_sa3_control_aliases, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)
    assert run._aliases == mock_sa3_control_aliases


def test_toml_alias_file(mock_sa3_control_data, mock_sa3_control_aliases, tmp_path):
    aliases_path = tmp_path / 'aliases.toml'
    aliases_path.write_text('''
sa3-xgm = "SA3_XTD10_XGM/XGM/DOOCS"

imgfel-frames = ["SA3_XTD10_IMGFEL/CAM/BEAMVIEW:daqOutput", "data.image.pixels"]
imgfel-frames2 = ["SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput", "data.image.pixels"]
imgfel-screen-pos = ["SA3_XTD10_IMGFEL/MOTOR/SCREEN", "actualPosition"]
imgfel-filter-pos = ["SA3_XTD10_IMGFEL/MOTOR/FILTER", "actualPosition"]

mcp-adc = "SA3_XTD10_MCP/ADC/1"
mcp-mpod = "SA3_XTD10_MCP/MCPS/MPOD"
mcp-voltage = ["SA3_XTD10_MCP/MCPS/MPOD", "channels.U3.voltage"]
mcp-trace = ["SA3_XTD10_MCP/ADC/1:channel_5.output", "data.rawData"]

bogus-source = "SA4_XTD20_XGM/XGM/DOOCS"
bogus-key = ["SA3_XTD10_XGM/XGM/DOOCS", "foo"]

["SA3_XTD10_XGM/XGM/DOOCS"]
hv = "pulseEnergy.wavelengthUsed"
beam-x = "beamPosition.ixPos"
beam-y = "beamPosition.iyPos"
    ''')

    run = H5File(mock_sa3_control_data).with_aliases(aliases_path)
    assert run._aliases == mock_sa3_control_aliases


def test_only_aliases(mock_sa3_control_data, mock_sa3_control_aliases, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases)
    subrun = H5File(mock_sa3_control_data).only_aliases(mock_sa3_control_aliases)

    # Assume that aliases work when the _aliases property is equal.
    assert run._aliases == subrun._aliases

    # Test whether only the sources used in aliases are present.
    assert subrun.all_sources == {
        'SA3_XTD10_XGM/XGM/DOOCS',
        'SA3_XTD10_IMGFEL/CAM/BEAMVIEW:daqOutput',
        'SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput',
        'SA3_XTD10_IMGFEL/MOTOR/SCREEN',
        'SA3_XTD10_IMGFEL/MOTOR/FILTER',
        'SA3_XTD10_MCP/ADC/1',
        'SA3_XTD10_MCP/MCPS/MPOD',
        'SA3_XTD10_MCP/ADC/1:channel_5.output',
    }

    # Test whether all keys are present for an aliased source.
    assert subrun['SA3_XTD10_XGM/XGM/DOOCS'].keys() == run['SA3_XTD10_XGM/XGM/DOOCS'].keys()

    # Test whether all keys are present for an aliased source, even if
    # there are key aliases for it as well.
    assert subrun['SA3_XTD10_MCP/MCPS/MPOD'].keys() == run['SA3_XTD10_MCP/MCPS/MPOD'].keys()

    # Test whether only aliased keys are present for unaliased sources.
    assert subrun['SA3_XTD10_IMGFEL/MOTOR/SCREEN'].keys() == {'actualPosition.value'}

    # Test strict selection.
    with pytest.raises(ValueError):
        H5File(mock_sa3_control_data).only_aliases(
            mock_sa3_control_aliases, strict=True)

    # Remove bogus aliases and test strict selection again.
    strict_aliases = mock_sa3_control_aliases.copy()
    del strict_aliases['bogus-source']
    del strict_aliases['bogus-key']
    H5File(mock_sa3_control_data).only_aliases(strict_aliases, strict=True)

    # Prepare a run with less trains for a single source
    # (SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput) by removing all sources
    # without any trains.
    run = H5File(mock_sa3_control_data) \
        .deselect([('SA3_XTD10_MCP/ADC/1:*', '*'),
                   ('SA3_XTD10_IMGFEL/CAM/BEAMVIEW:*', '*')])
    del strict_aliases['mcp-trace']
    del strict_aliases['imgfel-frames']

    # Without strict alias selection and a bogus alias.
    subrun = run.only_aliases(mock_sa3_control_aliases,
                              require_all=True, strict=False)
    np.testing.assert_array_equal(subrun.train_ids, run.train_ids[1::2])

    # With strict alias selection.
    subrun = run.only_aliases(strict_aliases, require_all=True, strict=True)
    np.testing.assert_array_equal(subrun.train_ids, run.train_ids[1::2])

    # Test that alias file paths are preserved
    subrun = H5File(mock_sa3_control_data).only_aliases(mock_sa3_control_aliases_yaml)
    assert subrun._alias_files == [mock_sa3_control_aliases_yaml]


def test_preserve_aliases(mock_sa3_control_data, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)

    # Test whether selection operations preserve aliases.
    for x in (by_index[:5], by_id[run.train_ids[:5]]):
        assert run.select_trains(x)._aliases == run._aliases
        assert run.select_trains(x)._alias_files == run._alias_files

    assert run.select('*')._aliases == run._aliases
    assert run.select('*')._alias_files == run._alias_files
    assert run.deselect('*XGM*')._aliases == run._aliases
    assert run.deselect('*XGM*')._alias_files == run._alias_files
    assert all([subrun._aliases == run._aliases
                for subrun in run.split_trains(parts=5)])
    assert all([subrun._alias_files == run._alias_files
                for subrun in run.split_trains(parts=5)])


def test_aliases_union(mock_sa3_control_data, mock_sa3_control_aliases, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)

    # Split the aliases into two halves and test the union.
    run1 = run.with_aliases(dict(islice(mock_sa3_control_aliases.items(), 0, None, 2)))
    run2 = run.with_aliases(dict(islice(mock_sa3_control_aliases.items(), 1, None, 2)))
    union = run1.union(run2)
    assert union._aliases == mock_sa3_control_aliases
    assert union._alias_files == [mock_sa3_control_aliases_yaml]

    # Split the run into two.
    even_run = run.select_trains(by_id[run.train_ids[0::2]])
    odd_run = run.select_trains(by_id[run.train_ids[1::2]])

    # Test overlapping aliases with no conflict.
    even_run.union(odd_run)

    # Test conflicting aliases.
    conflicting_aliases = mock_sa3_control_aliases.copy()
    conflicting_aliases['hv'] = ('SA3_XTD10_XGM/XGM/DOOCS', 'pressure.pressure1')
    with pytest.raises(ValueError):
        even_run.union(odd_run.with_aliases(conflicting_aliases))


def test_alias_select(mock_sa3_control_data, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)

    # Only source alias.
    subrun = run.alias.select('sa3-xgm')
    assert subrun.all_sources == {'SA3_XTD10_XGM/XGM/DOOCS'}
    assert subrun.alias['sa3-xgm'].keys() == run.alias['sa3-xgm'].keys()

    # Source alias and key glob.
    subrun = run.alias.select('sa3-xgm', 'pressure.pressure*.value')
    assert subrun.all_sources == {'SA3_XTD10_XGM/XGM/DOOCS'}
    assert subrun.alias['sa3-xgm'].keys() == {
        'pressure.pressure1.value', 'pressure.pressureFiltered.value'}

    # Iterable of aliases and/or with key globs.
    subrun = run.alias.select([('sa3-xgm', 'pressure.pressure*.value'),
                               'beam-x', 'mcp-voltage'])
    assert subrun.all_sources == {'SA3_XTD10_XGM/XGM/DOOCS', 'SA3_XTD10_MCP/MCPS/MPOD'}
    assert subrun.alias['sa3-xgm'].keys() == {
        'pressure.pressure1.value', 'pressure.pressureFiltered.value',
        'beamPosition.ixPos.value'}
    assert subrun.alias['mcp-mpod'].keys() == {'channels.U3.voltage.value'}

    # Dictionary
    subrun = run.alias.select({'sa3-xgm': None, 'mcp-mpod': {'channels.U1.voltage'}})
    assert subrun.all_sources == {'SA3_XTD10_XGM/XGM/DOOCS', 'SA3_XTD10_MCP/MCPS/MPOD'}
    assert subrun.alias['sa3-xgm'].keys() == run.alias['sa3-xgm'].keys()
    assert subrun.alias['mcp-mpod'].keys() == {'channels.U1.voltage.value'}

    # Test that file paths are preserved
    assert run.alias.select("sa3-xgm")._alias_files == [mock_sa3_control_aliases_yaml]


def test_alias_deselect(mock_sa3_control_data, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)

    # De-select via alias.
    subrun = run.alias.deselect([
        ('sa3-xgm', 'pressure.*'), ('sa3-xgm', 'current.*'),
        ('sa3-xgm', 'gasDosing.*'), ('sa3-xgm', 'gasSupply.*'),
        ('sa3-xgm', 'pressure.*'), ('sa3-xgm', 'pulseEnergy.*'),
        ('sa3-xgm', 'signalAdaption.*')
    ])
    assert subrun.all_sources == run.all_sources
    assert subrun.alias['sa3-xgm'].keys() == {
        'state.value', 'state.timestamp',
        'beamPosition.ixPos.value', 'beamPosition.ixPos.timestamp',
        'beamPosition.iyPos.value', 'beamPosition.iyPos.timestamp',
        'pollingInterval.value', 'pollingInterval.timestamp'}
    assert subrun._alias_files == [mock_sa3_control_aliases_yaml]

def test_alias_jhub_links(mock_sa3_control_data, mock_sa3_control_aliases_yaml):
    run = H5File(mock_sa3_control_data).with_aliases(mock_sa3_control_aliases_yaml)

    # A link for the current alias file should not be generated since it isn't under /gpfs
    assert run.alias.jhub_links() == { }

    # Test that the /gpfs root is replaced with the ~/GPFS symlink
    run._alias_files[0] = "/gpfs/foo.yaml"
    assert run.alias.jhub_links() == {"/gpfs/foo.yaml": "https://max-jhub.desy.de/hub/user-redirect/lab/tree/GPFS/foo.yaml"}
