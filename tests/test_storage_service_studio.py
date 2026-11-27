import subprocess
from service_test_client import OperatorClient as TestClient

from baloot_platform.service import api
from baloot_platform.service.api import app
from baloot_platform.storage.jobs import JobStore


def test_job_leases_idempotency_and_cancel(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    first = store.create("a", {"x": 1}, idempotency_key="same")
    second = store.create("b", {"x": 2}, idempotency_key="same")
    assert first == second == "a"
    claimed = store.claim("worker-1")
    assert claimed["jobId"] == "a" and claimed["status"] == "running"
    assert store.heartbeat("a", "worker-1")
    assert store.request_cancel("a") and store.cancelled("a")
    assert any(e["type"] == "cancel_requested" for e in store.events("a"))


def test_job_listing_is_filtered_bounded_and_newest_first(tmp_path):
    store = JobStore(tmp_path / "listed-jobs.sqlite3")
    store.create("older", {"x": 1}, kind="foundry_deep_analysis_v1")
    store.create("newer", {"x": 2}, kind="foundry_deep_analysis_v1")
    store.create("other", {"x": 3}, kind="video")

    items = store.list_jobs(kinds=("foundry_deep_analysis_v1",), limit=1)
    assert [item["jobId"] for item in items] == ["newer"]
    assert store.list_jobs(kinds=("video",), status="queued")[0]["jobId"] == "other"


def test_completed_retry_clears_stale_error(tmp_path):
    store = JobStore(tmp_path / "retried-jobs.sqlite3")
    store.create("retry", {"x": 1}, max_attempts=2)
    first = store.claim("worker-1")
    assert store.update(
        "retry",
        status="queued",
        error="transient",
        worker_id="worker-1",
        fencing_token=first["attempts"],
    )
    second = store.claim("worker-2")
    assert store.update(
        "retry",
        status="completed",
        result={"ok": True},
        worker_id="worker-2",
        fencing_token=second["attempts"],
    )
    completed = store.get("retry")
    assert completed["status"] == "completed"
    assert completed["error"] is None


def test_api_bundle_plugins_config_validation_and_security_headers():
    c = TestClient(app)
    r = c.get("/api/v1/bundles/current")
    assert r.status_code == 200 and len(r.json()["sha256"]) == 64
    assert c.post("/api/v1/auth/guest", json={"displayName": "test"}).status_code in {
        200,
        201,
    }
    plugins = c.get("/api/v1/plugins")
    assert plugins.status_code == 200
    assert any(p["id"] == "premium_video" for p in plugins.json()["items"])
    assert plugins.json()["externalInProcessAllowed"] is False
    payload = {
        "request": {
            "scope": "round",
            "contract": "sun",
            "rulesProfile": "sun_only",
            "projectsEnabled": False,
        }
    }
    response = c.post("/api/v1/configuration:validate", json=payload)
    assert response.status_code == 200 and response.json()["resolved"]["contracts"] == [
        "sun"
    ]
    assert (
        r.headers["x-content-type-options"] == "nosniff"
        and r.headers["x-frame-options"] == "DENY"
    )


def test_api_job_cancellation_and_events(monkeypatch, tmp_path):
    # Existing stored legacy jobs remain readable and cancellable after the
    # direct generator becomes a Research V4 compatibility adapter.
    store = JobStore(tmp_path / "legacy-video-jobs.sqlite3")
    monkeypatch.setattr(api, "STORE", store)
    job = "storedlegacyjob1"
    store.create(job, {"seed": 991}, kind="video")
    c = TestClient(app)
    cancelled = c.post(f"/api/v1/video-jobs/{job}:cancel")
    assert cancelled.status_code == 200
    assert cancelled.headers["deprecation"] == "true"
    read = c.get(f"/api/v1/video-jobs/{job}")
    assert read.headers["deprecation"] == "true"
    data = read.json()
    assert data["status"] == "cancelled"
    events = c.get(f"/api/v1/video-jobs/{job}/events")
    assert events.headers["deprecation"] == "true"
    assert events.json()["events"]


def test_studio_is_arabic_guided_and_json_only_advanced(root):
    text = (root / "src/baloot_platform/service/static/studio.html").read_text(
        encoding="utf-8"
    )
    assert '<html lang="ar" dir="rtl">' in text
    assert "إعداد سريع" in text and "ضبط متقدم" in text and "للمطورين" in text
    # Raw JSON editor must be inside the advanced tab only.
    assert text.index('id="advanced"') < text.index('id="advancedJson"')
    assert "premium_pillow_v4" not in text


def test_studio_javascript_syntax(root, tmp_path):
    text = (root / "src/baloot_platform/service/static/studio.html").read_text(
        encoding="utf-8"
    )
    script = text.split("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "studio.js"
    path.write_text(script, encoding="utf-8")
    subprocess.run(["node", "--check", str(path)], check=True, capture_output=True)
