#!/usr/bin/env python3
"""Controllable shutdown-handoff demonstration for the tas anomaly service.

Spawns a real uvicorn server (fresh data directory, deliberately slow
interruptible workers) and drives the two incident interleavings plus the
result-retention guarantee through the public HTTP API:

  Act 1 - shutdown interlaced with shared-baseline completion:
          queued jobs must never start anomaly workers after shutdown begins.
  Act 2 - shutdown while an anomaly worker is running:
          the worker must be terminated/reaped and no result may be published.
  Act 3 - a job completed before shutdown keeps its downloadable result,
          while unfinished job directories are removed.

On hosts whose seccomp policy forbids kill(2) the service is started with
ENABLE_SHUTDOWN_ENDPOINT=1 and the demo triggers the same lifespan shutdown
through ``POST /admin/shutdown``; otherwise SIGTERM is used, exactly like
``docker compose stop``.

Usage::

    python -I -m scripts.demo_shutdown [--host 127.0.0.1] [--port 8011]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLOW_STEP_DELAY = "0.35"
SLOW_STEPS = "12"


# --------------------------------------------------------------------- helpers
def log(act: str, msg: str) -> None:
    print(f"[{act}] {msg}", flush=True)


def http(method: str, url: str, data: bytes | None = None,
         content_type: str | None = None) -> tuple[int, dict | str]:
    """Return (status, body). Connection failures surface as status 0 so the
    demo can treat an exited service as an expected, checkable outcome."""
    req = urllib.request.Request(url, data=data, method=method)
    if content_type:
        req.add_header("content-type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                return resp.status, json.loads(body)
            return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:  # noqa: BLE001
            return exc.code, {}
    except (urllib.error.URLError, ConnectionError, OSError):
        return 0, {}


def wait_health(base: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = http("GET", f"{base}/healthz")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


def wait_predicate(desc: str, predicate, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for: {desc} (last={last!r})")


def upload(base: str, sample: Path) -> dict:
    boundary = "----tasdemo"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{sample.name}"\r\n'
        "Content-Type: application/x-netcdf\r\n\r\n"
    ).encode()
    payload += sample.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status, body = http(
        "POST", f"{base}/api/jobs", payload,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    assert status == 201, f"upload failed: {status} {body}"
    return body


def job_state(base: str, job_id: str) -> dict:
    status, body = http("GET", f"{base}/api/jobs/{job_id}")
    if status == 0:
        return {"state": "<server exited>"}
    assert status == 200, body
    return body


def list_worker_pids() -> set[int]:
    """PIDs of ``python -m app.worker`` processes still alive on the host."""
    pids: set[int] = set()
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        if "app.worker" in line and "demo_shutdown" not in line:
            try:
                pids.add(int(line.strip().split(None, 1)[0]))
            except (ValueError, IndexError):
                continue
    return pids


def signals_allowed() -> bool:
    probe = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    time.sleep(0.15)
    try:
        os.kill(probe.pid, signal.SIGTERM)
        probe.wait(timeout=5)
        return True
    except (PermissionError, ProcessLookupError, OSError):
        try:
            probe.wait(timeout=1)
        except Exception:  # noqa: BLE001
            pass
        return False


# ----------------------------------------------------------------------- acts
def act_baseline_interleave(base: str, samples: list[Path],
                            use_endpoint: bool, server: subprocess.Popen) -> None:
    act = "ACT1 基准完成×停机交错"
    log(act, "连续上传两份任务，触发共享基准计算 …")
    j1 = upload(base, samples[0])
    j2 = upload(base, samples[1])
    log(act, f"任务 {j1['id'][:8]} / {j2['id'][:8]} 已受理，等待共享基准进入 running")

    def baseline_running() -> dict | None:
        status, body = http("GET", f"{base}/api/baseline")
        run = body.get("active_run") if status == 200 else None
        if run and run["status"] in ("starting", "running"):
            return run
        return None

    run = wait_predicate("baseline starting/running", baseline_running, timeout=20)
    log(act, f"共享基准 run={run['run_id'][:8]} status={run['status']} "
             f"等待者={[w[:8] for w in run['waiters']]}")

    before = list_worker_pids()
    log(act, f"停机开始时存活 worker PID={sorted(before)}；"
             "基准即将完成，等待中的任务不得再启动距平进程")
    trigger_shutdown(act, base, use_endpoint, server)

    def jobs_settled() -> bool:
        states = {
            job_state(base, j1["id"])["state"],
            job_state(base, j2["id"])["state"],
        }
        return states <= {"cancelled", "done", "<server exited>"}

    wait_predicate("jobs reach terminal states", jobs_settled, timeout=40)
    s1 = job_state(base, j1["id"])
    s2 = job_state(base, j2["id"])
    log(act, f"停机后任务状态：{s1['state']} / {s2['state']}")

    # New work must be refused: 503 while the handoff is still observable, or a
    # refused connection once the service process has left — both prove that
    # no new task can be accepted after shutdown began.
    boundary = "----tasdemo"
    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="x.nc"\r\n'
        "Content-Type: application/x-netcdf\r\n\r\nx\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    status, body = http(
        "POST", f"{base}/api/jobs", payload,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    if status == 0:
        log(act, "服务进程已退出，新连接被拒绝（无法再受理任务），符合预期")
    else:
        log(act, f"停机后再上传 → HTTP {status}（预期 503）")
        assert status == 503, f"expected 503 while shutting down, got {status}: {body}"

    assert_server_and_workers_reaped(act, server, before)


def act_anomaly_running(base: str, samples: list[Path],
                        use_endpoint: bool, server: subprocess.Popen,
                        data_dir: Path) -> str:
    act = "ACT2 距平运行中停机"
    # First produce the climatology with a throwaway job so the next job goes
    # straight into its own anomaly subprocess.
    seed = upload(base, samples[0])
    log(act, f"先完成种子任务 {seed['id'][:8]} 以发布共享气候态 …")
    wait_predicate(
        "seed done",
        lambda: job_state(base, seed["id"])["state"] == "done",
        timeout=90,
    )
    log(act, "气候态已发布；上传新任务，其距平子进程立即启动")
    job = upload(base, samples[1])

    def proc_running() -> bool:
        body = job_state(base, job["id"])
        return body["state"] == "running" and body["stage"] == "anomaly"

    wait_predicate("anomaly worker running", proc_running, timeout=30)
    before = list_worker_pids()
    log(act, f"距平 worker 运行中 PID={sorted(before)}，现在停机 …")
    trigger_shutdown(act, base, use_endpoint, server)

    # Observe the cancelled terminal state while the handoff is still visible.
    def cancelled_or_exiting() -> bool:
        body = job_state(base, job["id"])
        return body["state"] in ("cancelled", "<server exited>")

    wait_predicate("job cancelled after reaping", cancelled_or_exiting, timeout=40)
    body = job_state(base, job["id"])
    log(act, f"任务终态={body['state']} 消息={body.get('message', '-')}")

    # No result may have been published for the interrupted job.
    status, _ = http("GET", f"{base}/api/jobs/{job['id']}/download")
    log(act, f"下载未完成任务 → HTTP {status}（预期 409，或进程已退出连接拒绝）")
    assert status in (0, 409, 503)
    job_dir = data_dir / "jobs" / job["id"]
    assert not job_dir.exists(), f"unfinished job directory leaked: {job_dir}"
    assert_server_and_workers_reaped(act, server, before)
    return seed["id"]


def act_retain_done_result(base: str, samples: list[Path],
                           use_endpoint: bool, server: subprocess.Popen,
                           data_dir: Path) -> None:
    act = "ACT3 已完成结果保留"
    done = upload(base, samples[0])
    wait_predicate(
        "done job finishes",
        lambda: job_state(base, done["id"])["state"] == "done",
        timeout=90,
    )
    running = upload(base, samples[1])
    wait_predicate(
        "second job starts",
        lambda: job_state(base, running["id"])["state"] in (
            "queued", "running", "cancelling"),
        timeout=30,
    )
    before = list_worker_pids()
    log(act, f"done={done['id'][:8]} 保留验证；running={running['id'][:8]} 将被回收")
    trigger_shutdown(act, base, use_endpoint, server)
    assert_server_and_workers_reaped(act, server, before)

    done_dir = data_dir / "jobs" / done["id"]
    running_dir = data_dir / "jobs" / running["id"]
    assert done_dir.is_dir() and (done_dir / "anomaly.nc").is_file(), \
        "pre-shutdown successful result must be retained"
    assert not running_dir.exists(), "unfinished directory must be cleaned"
    log(act, f"done 任务目录保留：{done_dir}/anomaly.nc；"
             f"未完成目录已删除：{running_dir}")


def trigger_shutdown(act: str, base: str, use_endpoint: bool,
                     server: subprocess.Popen) -> None:
    if use_endpoint:
        log(act, "POST /admin/shutdown（受限沙箱：进程内触发生命周期停机）")
        status, body = http("POST", f"{base}/admin/shutdown")
        assert status == 200, body
    else:
        log(act, f"kill -TERM {server.pid}（与 docker compose stop 同一路径）")
        server.terminate()


def assert_server_and_workers_reaped(act: str, server: subprocess.Popen,
                                     before: set[int]) -> None:
    # The service process must exit after the lifespan shutdown completes.
    try:
        rc = server.wait(timeout=60)
    except subprocess.TimeoutExpired as exc:
        server.kill()
        raise AssertionError("server process survived shutdown") from exc
    log(act, f"服务进程已退出，returncode={rc}")

    deadline = time.monotonic() + 10.0
    leftover: set[int] = set()
    while time.monotonic() < deadline:
        leftover = list_worker_pids() & before
        if not leftover:
            break
        time.sleep(0.2)
    assert not leftover, f"worker processes leaked after shutdown: {leftover}"
    log(act, f"停机前所有 worker（{sorted(before)}）均已终止并回收，无残留进程")


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--acts", default="1,2,3",
        help="comma-separated acts to run (1=baseline interleave, "
             "2=anomaly running, 3=retain done result)",
    )
    args = parser.parse_args()
    acts = {int(x) for x in args.acts.split(",") if x}

    use_endpoint = not signals_allowed()
    if use_endpoint:
        print("[demo] 本环境禁止 kill(2)，改用 /admin/shutdown 触发同一停机逻辑")
    else:
        print("[demo] 信号可用，使用 SIGTERM（等同 docker compose stop）")

    overall_rc = 0
    for act_no in sorted(acts):
        data_dir = Path(tempfile.mkdtemp(prefix=f"tas-demo-act{act_no}-"))
        try:
            # Generate the fixed baseline and demo samples up front.
            subprocess.run(
                [sys.executable, "-I", "-m", "scripts.generate_samples",
                 "--data-dir", str(data_dir)],
                cwd=REPO_ROOT, check=True, capture_output=True, text=True,
            )
            samples = sorted((data_dir / "samples").glob("*.nc"))
            base = f"http://{args.host}:{args.port}"
            env = os.environ.copy()
            env.update(
                DATA_DIR=str(data_dir),
                SLOW_STEP_DELAY=SLOW_STEP_DELAY,
                SLOW_STEPS=SLOW_STEPS,
                MAX_ACTIVE_JOBS="2",
                WORKER_TERM_TIMEOUT="8",
            )
            if use_endpoint:
                env["ENABLE_SHUTDOWN_ENDPOINT"] = "1"
            log(f"ACT{act_no}", f"启动 uvicorn（DATA_DIR={data_dir}）")
            server = subprocess.Popen(
                [sys.executable, "-I", "-m", "uvicorn",
                 "app.main:app", "--host", args.host,
                 "--port", str(args.port)],
                cwd=REPO_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_health(base)
                if act_no == 1:
                    act_baseline_interleave(
                        base, samples, use_endpoint, server)
                elif act_no == 2:
                    act_anomaly_running(
                        base, samples, use_endpoint, server, data_dir)
                elif act_no == 3:
                    act_retain_done_result(
                        base, samples, use_endpoint, server, data_dir)
                print(f"[ACT{act_no}] ✅ 通过\n", flush=True)
            finally:
                if server.poll() is None:
                    server.kill()
                    server.wait(timeout=10)
                # Surface the server log tail to make the handoff auditable.
                output = server.stdout.read() if server.stdout else ""
                tail = "\n".join(output.splitlines()[-12:])
                print(f"---- ACT{act_no} 服务日志尾部 ----\n{tail}\n",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback

            overall_rc = 1
            print(f"[ACT{act_no}] ❌ 失败: {exc!r}", flush=True)
            traceback.print_exc()
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

    print("演示全部通过" if overall_rc == 0 else "演示存在失败项")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
