import numpy as np

from furax.mapmaking import ObservationBufferShape as Shape
from furax.mapmaking._catalog import ObservationCatalog, detector_table, gather_rows
from furax.mapmaking._reader import ObservationRecord


def test_observation_catalog_keeps_the_probes_and_replaces_the_reads():
    probes = {0: Shape(2, 100), 1: 'RuntimeError: no file', 2: Shape(3, 50)}
    catalog = ObservationCatalog(['a', 'b', 'c'], probes)
    assert catalog.probe_failed.tolist() == [False, True, False]

    reads = {0: ObservationRecord('a0', ['d0', 'd1'], n_samples=90, load_seconds=1.5), 2: 'OSError'}
    catalog.reads = reads
    table = catalog.to_table()
    assert table['name'].to_list() == ['a0', 'b', 'c']  # as read, or as given
    assert table['loaded'].to_list() == [True, False, False]
    assert table['n_samples'].to_list() == [90, 0, 0]
    assert table['probed_samples'].to_list() == [100, 0, 50]
    assert table['load_seconds'].to_list() == [1.5, None, None]
    assert table['valid_sample_fraction'].null_count() == 3
    assert table['error'].to_list() == [
        None,
        'probe failed: RuntimeError: no file',
        'load failed: OSError',
    ]
    assert catalog.n_failed == 2

    # a new pass replaces the reads of the previous one
    catalog.reads = {}
    assert catalog.to_table()['loaded'].to_list() == [False, False, False]


def test_detector_frame_follows_the_detector_axis():
    # 2 and 1 detectors read on 3 padded slots, and an observation not read, on a QU TOD; only
    # the second has its quaternions read, its detector at the centre of the focal plane
    reads = {
        0: ObservationRecord('a', ['d0', 'd1'], n_samples=10),
        1: ObservationRecord('b', ['d2'], quaternions=np.array([[1.0, 0, 0, 0]]), n_samples=5),
        2: 'OSError',
    }
    sigma = np.arange(3 * 2 * 3, dtype=float).reshape(3, 2, 3)  # (obs, stream, slot)
    table = detector_table(reads, {'sigma': sigma}, 'QU')
    assert table['observation'].to_list() == [0, 0, 1]
    assert table['detector'].to_list() == [0, 1, 0]
    assert table['name'].to_list() == ['d0', 'd1', 'd2']
    assert table['sigma_q'].to_list() == [sigma[0, 0, 0], sigma[0, 0, 1], sigma[1, 0, 0]]
    assert table['sigma_u'].to_list() == [sigma[0, 1, 0], sigma[0, 1, 1], sigma[1, 1, 0]]
    assert table['xi'].to_list() == [None, None, 0.0]
    assert table['eta'].to_list() == [None, None, 0.0]


def test_detector_frame_of_nothing_read():
    table = detector_table({0: 'OSError'})
    assert table.height == 0
    assert table['observation'].dtype == table['detector'].dtype


def test_gather_rows_on_a_single_process():
    rows = {2: 'b', 0: Shape(1, 2)}
    assert gather_rows(rows) == rows
    assert gather_rows({}) == {}
