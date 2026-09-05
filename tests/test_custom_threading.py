from sidescantools.custom_threading import EGNTableBuilder, GeoreferencerManager


def test_georeferencer_cleanup_only_removes_its_blockmedian_temp_files(qtbot, tmp_path):
    generated_temp = tmp_path / "outmedian_line_ch0.xyz"
    user_xyz = tmp_path / "survey_points.xyz"
    user_xml = tmp_path / "metadata.xml"
    for path in (generated_temp, user_xyz, user_xml):
        path.write_text("test", encoding="utf-8")

    manager = GeoreferencerManager()
    qtbot.addWidget(manager)
    manager.output_folder = tmp_path

    manager.cleanup()

    assert not generated_temp.exists()
    assert user_xyz.exists()
    assert user_xml.exists()


def test_egn_builder_reports_when_no_files_can_be_processed(qtbot, tmp_path):
    builder = EGNTableBuilder(tmp_path / "table.npz")
    qtbot.addWidget(builder)
    messages = []
    builder.aborted_signal.connect(messages.append)

    builder.build_egn_table(
        [],
        tmp_path,
        nadir_angle=0,
        active_intern_depth=False,
        chunk_size=100,
        active_downsampling=False,
        egn_table_parameters=[360, 2],
    )

    assert messages == ["No sonar files with usable bottom information were found."]
