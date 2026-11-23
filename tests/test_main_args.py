from pathlib import Path


def test_main_args_include_minibench_and_dry_run():
    main_text = (Path(__file__).resolve().parent.parent / "main.py").read_text(
        encoding="utf-8"
    )
    assert '"minibench"' in main_text
    assert '"--dry-run"' in main_text
