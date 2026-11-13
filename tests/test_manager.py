"""Integration tests for the JobManager rules."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import xarray as xr

from app.manager import JobManager, LimitExceededError, ServiceShuttingDownError
from app.models import BaselineStatus, JobState
from app.netcdf_io import DatasetError
from tests.conftest import wait_until


def _payload(path: Path) -> bytes:
    return path.read_bytes()


async def _start(settings) -> JobManager:
    manager = JobManager(settings)
    await manager.startup()
    return manager


async def _submit(manager: JobManager, target_file: Path, name: str = "t.nc"):
    return await manager.submit(name, _payload(target_file))


async def _states(manager: JobManager):
    return {jid: job.state for jid, job in manager._jobs.items()}


# --------------------------------------------------------------------- happy
async def test_end_to_end_done_and_downloadable(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        job = await _submit(manager, target_file)
        await wait_until(lambda: job.terminal, timeout=40)
        assert job.state is JobState.DONE
        assert job.output_path.is_file()
        with xr.open_dataset(job.output_path, engine="h5netcdf") as out:
            assert tuple(out["tas"].dims) == ("time", "lat", "lon")
            assert out.sizes["time"] == 4
    finally:
        await manager.shutdown()


async def test_climatology_is_reused_by_second_job(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        first = await _submit(manager, target_file, "a.nc")
        await wait_until(lambda: first.state is JobState.DONE, timeout=40)
        assert settings.baseline_work_path.is_file()

        second = await _submit(manager, target_file, "b.nc")
        # No active baseline run should be needed anymore.
        await wait_until(lambda: second.terminal, timeout=40)
        assert second.state is JobState.DONE
        assert manager._baseline_run is None or (
            manager._baseline_run.status is BaselineStatus.DONE
        )
    finally:
        await manager.shutdown()


# ------------------------------------------------------------- concurrency
async def test_max_two_active_jobs_rejected_with_409(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        jobs = []
        jobs.append(await _submit(manager, target_file))
        jobs.append(await _submit(manager, target_file))
        with pytest.raises(LimitExceededError):
            await _submit(manager, target_file)

        # Terminal jobs free capacity again.
        await wait_until(
            lambda: all(j.state is JobState.DONE for j in jobs), timeout=60
        )
        third = await _submit(manager, target_file)
        await wait_until(lambda: third.terminal, timeout=40)
        assert third.state is JobState.DONE
    finally:
        await manager.shutdown()


async def test_invalid_grid_upload_is_rejected_and_not_counted(
    settings, baseline_file
):
    import pandas as pd

    from app.compute import make_synthetic_dataset

    manager = await _start(settings)
    try:
        times = pd.date_range("2021-01-01", periods=2, freq="MS")
        bad = make_synthetic_dataset(times, 5, 8, seed=99)
        bad_path = settings.samples_dir / "bad-grid.nc"
        bad.to_netcdf(bad_path, engine="h5netcdf")
        bad.close()
        with pytest.raises(DatasetError):
            await manager.submit("bad.nc", bad_path.read_bytes())
        assert manager._jobs == {}
    finally:
        await manager.shutdown()


# ----------------------------------------------------------- shared baseline
async def test_two_jobs_share_one_baseline_run(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")

        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run
        assert run.waiters == {a.id, b.id}
        assert run.proc is not None and run.proc.pid  # exactly one process

        await wait_until(
            lambda: a.state is JobState.DONE and b.state is JobState.DONE,
            timeout=60,
        )
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_one_waiter_does_not_affect_other(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run

        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)

        # The shared run survives because b is still waiting.
        assert run.status is BaselineStatus.RUNNING
        assert run.waiters == {b.id}
        assert run.proc.returncode is None
        assert not a.target_path.parent.exists()  # cancelled files cleaned

        await wait_until(lambda: b.state is JobState.DONE, timeout=60)
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_last_waiter_terminates_baseline(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run
        pid = run.proc.pid

        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)
        await manager.cancel(b.id)

        # Process must be reaped, run terminated, artifacts cleaned.
        await wait_until(
            lambda: run.status is BaselineStatus.TERMINATED, timeout=15
        )
        assert run.proc.returncode is not None
        assert run.proc.pid == pid
        assert not settings.baseline_work_path.exists()
        assert not run.part_path.exists()
        assert b.state is JobState.CANCELLED

        # A later submission must NOT attach to the terminated/draining run.
        # It may briefly observe the draining slot before spawning a fresh
        # run, so assert on the outcome: the job completes and never joins the
        # old run's waiter set.
        old_run = run
        c = await _submit(manager, target_file, "c.nc")
        assert c.id not in old_run.waiters

        async def _fresh_run_or_done():
            current = manager._baseline_run
            return c.state is JobState.DONE or (
                current is not None
                and current is not old_run
                and current.status
                in (BaselineStatus.STARTING, BaselineStatus.RUNNING)
            )

        await wait_until(_fresh_run_or_done, timeout=15)
        await wait_until(lambda: c.state is JobState.DONE, timeout=60)
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_queued_job_before_worker_starts(
    settings, baseline_file, target_file
):
    # Fill both slots first, then cancel a queued one (it has no process).
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None, timeout=10
        )
        await manager.cancel(b.id)
        await wait_until(lambda: b.state is JobState.CANCELLED, timeout=10)
        assert b.proc is None
        await wait_until(lambda: a.state is JobState.DONE, timeout=60)
    finally:
        await manager.shutdown()


# ------------------------------------------------------------- own-process
async def test_cancel_running_anomaly_reaps_and_reports_cancelled(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        seed = await _submit(manager, target_file, "seed.nc")
        await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

        job = await _submit(manager, target_file, "late.nc")
        await wait_until(lambda: job.proc is not None, timeout=15)
        proc = job.proc
        await manager.cancel(job.id)
        assert job.state is JobState.CANCELLING  # not cancelled yet
        await wait_until(lambda: job.state is JobState.CANCELLED, timeout=20)
        assert proc.returncode is not None  # actually reaped
        assert not job.target_path.parent.exists()
        assert not job.output_path.exists()
    finally:
        await manager.shutdown()


async def test_cancel_racing_completion_never_publishes_late_result(
    settings, baseline_file, target_file
):
    # Hammer cancellation while the anomaly worker is on its way out.
    manager = await _start(settings)
    try:
        seed = await _submit(manager, target_file, "seed.nc")
        await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

        for attempt in range(3):
            job = await _submit(manager, target_file, f"race{attempt}.nc")
            await wait_until(lambda: job.proc is not None, timeout=15)
            # Cancel immediately: either the worker exits first or publish is
            # suppressed; both outcomes must remain internally consistent.
            await manager.cancel(job.id)
            await wait_until(lambda: job.terminal, timeout=30)
            if job.state is JobState.DONE:
                assert job.output_path.is_file()
            else:
                assert job.state is JobState.CANCELLED
                assert not job.output_path.exists()
    finally:
        await manager.shutdown()


async def test_baseline_worker_failure_marks_waiters_error(
    settings, target_file, monkeypatch
):
    # Break the baseline input after manager startup validation by pointing the
    # worker at an unreadable file: simulate via a corrupt replacement.
    manager = await _start(settings)
    try:
        good = settings.baseline_input_path
        backup = good.with_suffix(".nc.bak")
        good.rename(backup)
        good.write_bytes(b"not an hdf5 file at all")
        job = await _submit(manager, target_file, "x.nc")
        await wait_until(lambda: job.terminal, timeout=20)
        assert job.state is JobState.ERROR
        assert "baseline" in job.message.lower()
        good.unlink()
        backup.rename(good)
    finally:
        await manager.shutdown()


async def test_cancel_is_idempotent_and_terminal_jobs_unchanged(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        job = await _submit(manager, target_file, "a.nc")
        await wait_until(lambda: job.terminal, timeout=40)
        before = job.state
        again = await manager.cancel(job.id)
        assert again.state is before
    finally:
        await manager.shutdown()


async def test_shutdown_terminates_all_workers(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    a = await _submit(manager, target_file, "a.nc")
    b = await _submit(manager, target_file, "b.nc")
    await wait_until(
        lambda: manager._baseline_run is not None
        and manager._baseline_run.proc is not None,
        timeout=10,
    )
    base_proc = manager._baseline_run.proc
    await manager.shutdown()
    assert base_proc.returncode is not None
    for job in (a, b):
        assert job.proc is None or job.proc.returncode is not None


# ---------------------------------------------------------------- shutdown
async def test_shutdown_rejects_new_submissions(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    await manager.shutdown()
    with pytest.raises(ServiceShuttingDownError):
        await _submit(manager, target_file, "late.nc")
    # Nothing was created for the refused submission.
    assert manager._jobs == {}
    assert not any(settings.jobs_dir.iterdir())


async def test_shutdown_is_idempotent_and_coalesced(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    a = await _submit(manager, target_file, "a.nc")
    b = await _submit(manager, target_file, "b.nc")
    await wait_until(
        lambda: manager._baseline_run is not None
        and manager._baseline_run.proc is not None,
        timeout=10,
    )
    base_proc = manager._baseline_run.proc

    # Concurrent and repeated shutdown calls share one teardown.
    results = await asyncio.gather(
        manager.shutdown(), manager.shutdown(), return_exceptions=True
    )
    assert results == [None, None]
    await manager.shutdown()  # again, after completion
    assert manager._shutdown_task is not None
    assert manager._shutdown_task.done()
    assert base_proc.returncode is not None
    for job in (a, b):
        assert job.state is JobState.CANCELLED
        assert not job.target_path.parent.exists()


async def test_shutdown_cancelled_await_still_reaps_everything(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    a = await _submit(manager, target_file, "a.nc")
    await wait_until(
        lambda: manager._baseline_run is not None
        and manager._baseline_run.proc is not None,
        timeout=10,
    )
    base_proc = manager._baseline_run.proc

    outer = asyncio.create_task(manager.shutdown())
    # Cancel the awaiting task while the teardown is in flight.
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    # The shielded teardown must still run to completion: no live process,
    # no unfinished job, no leftover directory.
    await wait_until(lambda: manager._shutdown_task.done(), timeout=15)
    assert base_proc.returncode is not None
    assert a.state is JobState.CANCELLED
    assert not a.target_path.parent.exists()
    # And a later, uncancelled shutdown call returns cleanly (same finish).
    await manager.shutdown()


async def test_shutdown_during_running_anomaly_reaps_worker_and_publishes_nothing(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    seed = await _submit(manager, target_file, "seed.nc")
    await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

    job = await _submit(manager, target_file, "anom.nc")
    await wait_until(lambda: job.proc is not None, timeout=15)
    proc = job.proc
    assert proc.returncode is None

    await manager.shutdown()
    assert proc.returncode is not None  # actually reaped
    assert job.state is JobState.CANCELLED
    assert not job.output_path.exists()
    assert not job.target_path.parent.exists()
    # The already completed job keeps its result on disk.
    assert seed.state is JobState.DONE
    assert seed.output_path.is_file()


async def test_shutdown_racing_baseline_completion_never_starts_anomaly(
    settings, baseline_file, target_file, monkeypatch
):
    manager = await _start(settings)
    job = await _submit(manager, target_file, "race.nc")
    launch_called = asyncio.Event()
    original_launch = manager._launch_anomaly

    async def _guarded_launch(record):
        # Simulate the exact handoff described in the incident: the shared
        # baseline has just completed and the job is about to start its
        # anomaly worker when the shutdown begins.
        launch_called.set()
        await asyncio.sleep(0.05)
        return await original_launch(record)

    monkeypatch.setattr(manager, "_launch_anomaly", _guarded_launch)

    await wait_until(launch_called.wait, timeout=20)
    await manager.shutdown()

    assert job.state is JobState.CANCELLED
    assert job.proc is None  # the anomaly process was never started
    assert not job.output_path.exists()
    assert not job.target_path.parent.exists()


async def test_late_created_anomaly_process_is_stopped_and_reaped(
    settings, baseline_file, target_file, monkeypatch
):
    # Reproduce the incident deterministically: the anomaly process is created
    # by the baseline->anomaly handoff while the shutdown flag is being set,
    # so it is absent from phase-1's snapshot.  The phase-4 rescan must still
    # terminate and reap it.
    manager = await _start(settings)
    seed = await _submit(manager, target_file, "seed.nc")
    await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

    job = await _submit(manager, target_file, "late-spawn.nc")
    await wait_until(lambda: job.proc is not None, timeout=15)
    hidden_proc = job.proc

    real_terminate = manager._terminate
    phase_two_seen = asyncio.Event()
    release = asyncio.Event()

    async def _hide_from_phase_two(proc, cancel_file, label):
        # The first terminate attempt for this process is a no-op that only
        # unblocks the test, emulating a process missing from the phase-1
        # snapshot.  A later (phase-4) attempt performs the real reaping.
        if proc is hidden_proc and not phase_two_seen.is_set():
            phase_two_seen.set()
            await release.wait()
            return None
        return await real_terminate(proc, cancel_file, label)

    monkeypatch.setattr(manager, "_terminate", _hide_from_phase_two)
    shutdown_task = asyncio.create_task(manager.shutdown())

    # Phase 2 "misses" the hidden process and parks on the release gate, which
    # also keeps its supervisor blocked inside the phase-3 gather.
    await wait_until(phase_two_seen.wait, timeout=10)
    assert hidden_proc.returncode is None
    release.set()  # let teardown proceed into the phase-4 rescan

    await shutdown_task
    assert hidden_proc.returncode is not None
    assert job.proc.returncode is not None
    assert job.state is JobState.CANCELLED
    assert not job.target_path.parent.exists()


async def test_done_result_survives_shutdown_and_unfinished_dirs_removed(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    done_job = await _submit(manager, target_file, "done.nc")
    await wait_until(lambda: done_job.state is JobState.DONE, timeout=40)

    # A second, still-unfinished job when shutdown arrives.
    running = await _submit(manager, target_file, "running.nc")
    await wait_until(lambda: running.proc is not None, timeout=15)
    proc = running.proc

    await manager.shutdown()

    assert done_job.state is JobState.DONE
    assert done_job.output_path.is_file()  # pre-shutdown success retained
    assert running.state is JobState.CANCELLED
    assert proc.returncode is not None
    assert not running.target_path.parent.exists()
    # Only the finished job's directory remains under jobs/.
    remaining = [p for p in settings.jobs_dir.iterdir() if p.is_dir()]
    assert remaining == [done_job.target_path.parent]


async def test_cancel_request_racing_shutdown_is_harmless(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    a = await _submit(manager, target_file, "a.nc")
    b = await _submit(manager, target_file, "b.nc")
    await wait_until(
        lambda: manager._baseline_run is not None
        and manager._baseline_run.proc is not None,
        timeout=10,
    )
    base_proc = manager._baseline_run.proc

    # A user cancel racing the shutdown must not corrupt the teardown.
    shutdown_task = asyncio.create_task(manager.shutdown())
    cancel_result = await manager.cancel(a.id)
    await shutdown_task

    assert cancel_result.state in (
        JobState.CANCELLING,
        JobState.CANCELLED,
    )
    assert a.state is JobState.CANCELLED
    assert b.state is JobState.CANCELLED
    assert base_proc.returncode is not None
    assert not a.target_path.parent.exists()
    assert not b.target_path.parent.exists()


async def test_restart_cleans_leftover_job_dirs_with_original_rule(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    job = await _submit(manager, target_file, "a.nc")
    await wait_until(lambda: job.state is JobState.DONE, timeout=40)
    leftover = settings.jobs_dir / "orphan"
    leftover.mkdir()
    (leftover / "partial.nc.part").write_bytes(b"x")

    # A brand-new manager over the same data dir applies the original
    # startup rule: wipe the whole jobs directory.
    manager_two = JobManager(settings)
    await manager_two.startup()
    try:
        assert not settings.jobs_dir.exists() or not any(
            settings.jobs_dir.iterdir()
        )
        # The published climatology is untouched by startup cleanup.
        assert settings.baseline_work_path.is_file()
    finally:
        await manager_two.shutdown()
