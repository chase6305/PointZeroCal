import numpy as np
import pytest

from point_zero_cal import SampleTable, load_samples_csv


def test_numeric_csv_preserves_first_sample_and_physical_rows(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_bytes(
        '\ufeff# recording\r\n\r\n"1", 2 # first pose\r\n# next pose\r\n3,4\r\n'.encode(
            "utf-8"
        )
    )

    table = load_samples_csv(path, ["joint1", "joint2"])

    np.testing.assert_array_equal(table.values, [[1, 2], [3, 4]])
    assert table.joint_names == ("joint1", "joint2")
    assert table.csv_rows == (3, 5)
    assert not table.values.flags.writeable


def test_named_csv_reorders_columns_and_retains_source_rows(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "# recording\n joint2, joint1\n90,0\n\n-90,90 # second pose\n",
        encoding="utf-8",
    )

    table = load_samples_csv(path, ("joint1", "joint2"), header=True)

    np.testing.assert_array_equal(table.values, [[0, 90], [90, -90]])
    assert table.csv_rows == (3, 5)


def test_quoted_joint_names_preserve_hashes_commas_and_escaped_quotes(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text('"j#1", "j,""2" # header\n1,2\n3,4\n', encoding="utf-8")

    table = load_samples_csv(path, ('j,"2', "j#1"), header=True)

    np.testing.assert_array_equal(table.values, [[2, 1], [4, 3]])


@pytest.mark.parametrize(
    "contents, header, message",
    [
        ("joint1,joint2\n0,1\n1,0\n", False, "line 1, column 1.*convert"),
        ("broken,1\n0,1\n1,0\n", False, "line 1, column 1.*convert"),
        ("0,1\n1,0\n", True, "missing joint names.*unknown joint names"),
        ("joint1,joint1\n0,1\n1,0\n", True, "duplicate joint names"),
        ("joint2\n0\n1\n", True, "missing joint names:.*joint1"),
        ("joint1,joint2,time\n0,1,0\n1,0,1\n", True, "unknown joint names:.*time"),
        ("joint1,\n0,1\n1,0\n", True, "empty joint name"),
        ("0,1\n2,3,4\n", False, "line 2: expected 2 joint columns, got 3"),
        ('0,1\n"2,3\n', False, "line 2: invalid CSV"),
        ("", False, "found 0 sample rows"),
        ("# no samples\n\n", False, "found 0 sample rows"),
        ("joint1,joint2\n", True, "found 0 sample rows"),
        ("0,1\n", False, "found 1 sample rows"),
    ],
)
def test_csv_rejects_ambiguous_or_malformed_tables(tmp_path, contents, header, message):
    path = tmp_path / "samples.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_samples_csv(path, ("joint1", "joint2"), header=header)


@pytest.mark.parametrize(
    "value", ["", "invalid", "1+2j", "nan", "inf", "-inf", "1e999"]
)
def test_invalid_values_identify_original_column_and_joint(tmp_path, value):
    path = tmp_path / "samples.csv"
    path.write_text(f"joint2,joint1\n0,1\n# comment\n{value},2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 4, column 1 \\('joint2'\\)") as error:
        load_samples_csv(path, ("joint1", "joint2"), header=True)

    assert str(path) in str(error.value)


def test_single_joint_csv_stays_two_dimensional(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("joint1\n1\n2\n", encoding="utf-8")

    table = load_samples_csv(path, ("joint1",), header=True)

    assert table.values.shape == (2, 1)
    assert table.csv_rows == (2, 3)


def test_csv_reports_invalid_encoding(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_bytes(b"0,1\n2,\xff\n")

    with pytest.raises(ValueError, match="UTF-8"):
        load_samples_csv(path, ("joint1", "joint2"))


def test_sample_table_copies_values_and_metadata():
    values = np.array([[1.0], [2.0]])
    names = ["joint1"]
    rows = [2, 4]
    table = SampleTable(values, names, rows)
    values[:] = 0
    names[0] = "changed"
    rows[0] = 1

    np.testing.assert_array_equal(table.values, [[1], [2]])
    assert table.joint_names == ("joint1",)
    assert table.csv_rows == (2, 4)
    with pytest.raises(ValueError, match="read-only"):
        table.values[0, 0] = 0


@pytest.mark.parametrize("rows", [(0, 1), (2, 2), (3, 2), (True, 2), (1.5, 2), (1,)])
def test_sample_table_rejects_invalid_source_rows(rows):
    with pytest.raises(ValueError, match="csv_rows"):
        SampleTable(np.array([[1.0], [2.0]]), ("joint1",), rows)


@pytest.mark.parametrize("names", ["joint1", [], [""], ["joint1", "joint1"], [None]])
def test_csv_requires_unique_model_joint_names(tmp_path, names):
    with pytest.raises(ValueError, match="joint_names"):
        load_samples_csv(tmp_path / "unused.csv", names)
