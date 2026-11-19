# Changelog

## Unreleased

- **Added** `runner/`: a generic resumable shard runner (`shard_runner.py`), a launcher that restarts it under the watchdog (`run_sharded.sh`), a toy task and an example config. Tested natively on macOS (M3 Pro, mlx-lm 0.32.2) with the toy task and with Gemma 4 E4B instruct 8-bit: kill and resume, launcher restarts and canary all pass.
- **Added** `tests/test_shard_runner.py` and `tests/test_run_guarded.py`: GPU-free tests (kill and resume, torn line, canary, pause, recycle, design change; watchdog with fake `vm_stat` and `footprint`).
- **Changed** `scripts/run_guarded.py`: earlier logs and guard records with the same `--name` are kept under a time-stamped name instead of overwritten; the guard record includes the last five log lines; new `--control-file`, `--block-on`, `--mlx-pattern` and `--log-dir` options; exit code 10 when refused because of the control file. Defaults and the kill rule are unchanged.
- **Added** `FRAMEWORK_KNOWLEDGE_CARD.md`: a generic guide to LoRA training and evaluation on Apple Silicon with mlx-lm.

- **Fixed** `tests/test_run_guarded.py`: the competing-process check now also works on macOS, where `sh` replaces itself with a single command and drops the marker.

## 0.1.0

Initial release: memory-safe LoRA recipe for Apple Silicon (mlx-lm), agent runbook and measured findings.
