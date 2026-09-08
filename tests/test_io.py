import stat

from point_zero_cal._io import write_text_atomic


def test_atomic_write_preserves_symlink_target_and_existing_permissions(tmp_path):
    destination = tmp_path / "report.json"
    destination.write_text("old contents")
    destination.chmod(0o640)
    link = tmp_path / "link.json"
    link.symlink_to(destination)

    write_text_atomic(link, '{"status": "标定完成"}\n')

    assert link.is_symlink()
    assert destination.read_text() == '{"status": "标定完成"}\n'
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(".report.json.*.tmp"))
