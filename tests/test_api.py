"""End-to-end HTTP tests through the real FastAPI app (in-process ASGI)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from tests.conftest import wait_until


@pytest.fixture()
async def client(settings: Settings, baseline_file: Path):
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as ac:
        # lifespan startup runs on ASGI requests? No — ASGITransport does not
        # run lifespan by default; trigger it explicitly.
        async with _lifespan(app):
            yield ac


class _lifespan:
    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        self.ctx = self.app.router.lifespan_context(self.app)
        await self.ctx.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.ctx.__aexit__(*exc)


async def test_health_baseline_and_samples(client: httpx.AsyncClient):
    r = await client.get("/healthz")
    assert r.status_code == 200
    r = await client.get("/api/baseline")
    assert r.status_code == 200
    body = r.json()
    assert body["climatology_ready"] is False
    assert body["grid"]["lat_size"] == 6


async def test_submit_status_cancel_download_flow(
    client: httpx.AsyncClient, target_file: Path
):
    with target_file.open("rb") as fh:
        r = await client.post(
            "/api/jobs", files={"file": ("target.nc", fh, "application/x-netcdf")}
        )
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]

    r = await client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["state"] in ("queued", "running", "cancelling", "done")

    # Wait for completion via the same polling loop the browser uses.
    async def _done():
        poll = await client.get(f"/api/jobs/{job_id}")
        return poll.json()["state"] == "done"

    await wait_until(_done, timeout=40)

    r = await client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-netcdf")
    assert len(r.content) > 0


async def test_upload_garbage_returns_400(
    client: httpx.AsyncClient
):
    r = await client.post(
        "/api/jobs",
        files={"file": ("bad.nc", b"definitely not netcdf", "application/octet-stream")},
    )
    assert r.status_code == 400


async def test_cancel_flow_over_http(
    client: httpx.AsyncClient, target_file: Path
):
    with target_file.open("rb") as fh:
        r = await client.post(
            "/api/jobs", files={"file": ("a.nc", fh, "application/x-netcdf")}
        )
    job_id = r.json()["id"]
    r = await client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert r.json()["state"] in ("cancelling", "cancelled", "done")

    async def _terminal():
        poll = await client.get(f"/api/jobs/{job_id}")
        return poll.json()["state"] in ("cancelled", "done", "error")

    await wait_until(_terminal, timeout=30)

    r = await client.get("/api/jobs/unknown-id")
    assert r.status_code == 404


async def test_limit_returns_409(client: httpx.AsyncClient, target_file: Path):
    ids = []
    for _ in range(2):
        with target_file.open("rb") as fh:
            r = await client.post(
                "/api/jobs",
                files={"file": ("t.nc", fh, "application/x-netcdf")},
            )
        assert r.status_code == 201
        ids.append(r.json()["id"])
    with target_file.open("rb") as fh:
        r = await client.post(
            "/api/jobs",
            files={"file": ("t.nc", fh, "application/x-netcdf")},
        )
    assert r.status_code == 409


async def test_submission_after_shutdown_returns_503(
    settings: Settings, baseline_file: Path, target_file: Path, monkeypatch
):
    monkeypatch.setenv("ENABLE_SHUTDOWN_ENDPOINT", "1")
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as ac:
        async with _lifespan(app):
            with target_file.open("rb") as fh:
                ok = await ac.post(
                    "/api/jobs",
                    files={"file": ("t.nc", fh, "application/x-netcdf")},
                )
            assert ok.status_code == 201

            # Drive the manager shutdown directly (the HTTP endpoint would call
            # os._exit, which cannot run inside the in-process ASGI test).
            await app.state.manager.shutdown()

            with target_file.open("rb") as fh:
                refused = await ac.post(
                    "/api/jobs",
                    files={"file": ("t.nc", fh, "application/x-netcdf")},
                )
            assert refused.status_code == 503
            assert "shutting down" in refused.json()["detail"]

            # Read-only endpoints still answer during/after the handoff.
            assert (await ac.get("/healthz")).status_code == 200
            assert (await ac.get("/api/jobs")).status_code == 200


async def test_shutdown_endpoint_disabled_by_default(
    settings: Settings, baseline_file: Path
):
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=30
    ) as ac:
        async with _lifespan(app):
            r = await ac.post("/admin/shutdown")
            assert r.status_code == 404
        await app.state.manager.shutdown()
