import numpy as np
import pytest


def test_read_tum_trajectory(tmp_path):
    from robust_loop_verifier.io import read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text(
        "1.0 0 0 0 0 0 0 1\n"
        "1.1 1 2 3 0 0 0 1\n",
        encoding="utf-8",
    )

    records = read_tum_trajectory(path)

    assert [record.timestamp for record in records] == [1.0, 1.1]
    np.testing.assert_allclose(records[1].pose[:3, 3], [1, 2, 3])


def test_read_tum_trajectory_skips_blank_comments_and_sorts(tmp_path):
    from robust_loop_verifier.io import read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text(
        "\n"
        "# timestamp tx ty tz qx qy qz qw\n"
        "2.0 2 0 0 0 0 0 1\n"
        "   \n"
        "1.0 1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    records = read_tum_trajectory(path)

    assert [record.timestamp for record in records] == [1.0, 2.0]
    np.testing.assert_allclose(records[0].pose[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(records[1].pose[:3, 3], [2, 0, 0])


def test_read_tum_trajectory_rejects_bad_field_count(tmp_path):
    from robust_loop_verifier.io import read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text("1.0 0 0 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TUM row 1 must have 8 fields"):
        read_tum_trajectory(path)


def test_read_tum_trajectory_rejects_non_finite_values(tmp_path):
    from robust_loop_verifier.io import read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text("1.0 0 nan 0 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TUM row 1"):
        read_tum_trajectory(path)


def test_associate_tum_by_timestamp_nearest_with_tolerance(tmp_path):
    from robust_loop_verifier.io import associate_tum_by_timestamp, read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text(
        "10.00 0 0 0 0 0 0 1\n"
        "10.05 5 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    records = read_tum_trajectory(path)

    record = associate_tum_by_timestamp(records, 10.049, max_delta_sec=0.01)

    np.testing.assert_allclose(record.pose[:3, 3], [5, 0, 0])


def test_associate_tum_by_timestamp_rejects_tolerance_miss(tmp_path):
    from robust_loop_verifier.io import associate_tum_by_timestamp, read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text("10.00 0 0 0 0 0 0 1\n", encoding="utf-8")
    records = read_tum_trajectory(path)

    with pytest.raises(ValueError, match="max_delta_sec|exceeds|tolerance"):
        associate_tum_by_timestamp(records, 10.05, max_delta_sec=0.01)


def test_associate_tum_by_timestamp_rejects_invalid_query_or_tolerance(tmp_path):
    from robust_loop_verifier.io import associate_tum_by_timestamp, read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text("10.00 0 0 0 0 0 0 1\n", encoding="utf-8")
    records = read_tum_trajectory(path)

    with pytest.raises(ValueError, match="timestamp|finite"):
        associate_tum_by_timestamp(records, np.nan, max_delta_sec=0.01)

    with pytest.raises(ValueError, match="max_delta_sec|finite"):
        associate_tum_by_timestamp(records, 10.0, max_delta_sec=np.nan)

    with pytest.raises(ValueError, match="max_delta_sec|negative"):
        associate_tum_by_timestamp(records, 10.0, max_delta_sec=-1.0)


def test_associate_tum_by_timestamp_rejects_empty_records():
    from robust_loop_verifier.io import associate_tum_by_timestamp

    with pytest.raises(ValueError, match="empty TUM trajectory"):
        associate_tum_by_timestamp([], 10.0, max_delta_sec=0.01)


def test_associate_tum_by_timestamp_sorts_unsorted_records(tmp_path):
    from robust_loop_verifier.io import associate_tum_by_timestamp, read_tum_trajectory

    path = tmp_path / "trajectory.txt"
    path.write_text(
        "10.00 0 0 0 0 0 0 1\n"
        "10.05 5 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    records = read_tum_trajectory(path)

    record = associate_tum_by_timestamp(list(reversed(records)), 10.049, max_delta_sec=0.01)

    np.testing.assert_allclose(record.pose[:3, 3], [5, 0, 0])
