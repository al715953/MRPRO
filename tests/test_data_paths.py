from pathlib import Path

from src.data_access import config


def test_source_data_directory_is_at_project_root():
    project_root = Path(__file__).resolve().parents[1]

    assert config.PROJECT_ROOT == project_root
    assert config.DATA_FOLDER_PATH == project_root / "data"
    assert Path(config.DATA_FOLDER) == project_root / "data"
    assert not (project_root / "src" / "data").exists()


def test_all_persistent_paths_share_the_new_data_directory():
    paths = (
        config.CSV_FILE_PATH,
        config.FILE_APUESTAS,
        config.FILE_CARTERAS_SOMBRA,
        config.FILE_TABLERO_SOMBRA,
        config.MASTER_LOG_PATH,
        config.MODEL_FILE_PATH,
        config.BACKTEST_MODEL_FILE_PATH,
        config.NUMBER_MODEL_FILE_PATH,
        config.BACKTEST_NUMBER_MODEL_FILE_PATH,
        config.BACKTEST_MODEL_CACHE_PATH,
    )

    assert all(Path(path).parent == config.DATA_FOLDER_PATH for path in paths)


def test_legacy_migration_moves_missing_files_without_overwriting(tmp_path):
    legacy = tmp_path / "src" / "data"
    destination = tmp_path / "data"
    legacy.mkdir(parents=True)
    destination.mkdir()
    (legacy / "history.csv").write_text("legacy-history", encoding="utf-8")
    (legacy / "model.json").write_text("legacy-model", encoding="utf-8")
    (destination / "model.json").write_text("current-model", encoding="utf-8")

    moved = config.migrate_legacy_data(legacy, destination)

    assert moved == ["history.csv"]
    assert (destination / "history.csv").read_text(encoding="utf-8") == "legacy-history"
    assert (destination / "model.json").read_text(encoding="utf-8") == "current-model"
    assert (legacy / "model.json").read_text(encoding="utf-8") == "legacy-model"


def test_user_data_directory_uses_windows_location(monkeypatch, tmp_path):
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert config._user_data_directory() == tmp_path / "MRPRO" / "data"


def test_user_data_directory_uses_apple_application_support(monkeypatch, tmp_path):
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    expected = tmp_path / "Library" / "Application Support" / "MRPRO" / "data"

    for platform in ("darwin", "ios"):
        monkeypatch.setattr(config.sys, "platform", platform)
        assert config._user_data_directory() == expected
