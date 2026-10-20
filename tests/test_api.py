import pytest
from fastapi.testclient import TestClient

from experimentation_platform.serving import build_demo_app, create_app
from experimentation_platform.registry import ExperimentRegistry
from experimentation_platform.events import EventStore


@pytest.fixture()
def client():
    return TestClient(create_app(ExperimentRegistry(), EventStore()))


@pytest.fixture()
def demo_client():
    return TestClient(build_demo_app())


EXPERIMENT = {
    "id": "api-exp",
    "name": "API experiment",
    "hypothesis": "h",
    "variants": [
        {"name": "control", "weight": 0.5, "is_control": True},
        {"name": "treatment", "weight": 0.5},
    ],
    "metrics": [
        {"name": "signup", "kind": "conversion", "is_primary": True},
        {"name": "errors", "kind": "conversion", "is_guardrail": True,
         "higher_is_better": False, "guardrail_tolerance": 0.005},
    ],
}


def test_health(demo_client):
    r = demo_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "checkout-flow-v2" in r.json()["experiments"]


def test_create_and_get_experiment(client):
    r = client.post("/experiments", json=EXPERIMENT)
    assert r.status_code == 201
    assert r.json()["status"] == "draft"
    r = client.get("/experiments/api-exp")
    assert r.status_code == 200
    assert r.json()["name"] == "API experiment"


def test_create_invalid_experiment_rejected(client):
    bad = dict(EXPERIMENT, id="bad")
    bad["variants"] = [
        {"name": "control", "weight": 0.4, "is_control": True},
        {"name": "treatment", "weight": 0.4},
    ]
    assert client.post("/experiments", json=bad).status_code == 422


def test_missing_experiment_404(client):
    assert client.get("/experiments/nope").status_code == 404


def test_transition_flow(client):
    client.post("/experiments", json=EXPERIMENT)
    r = client.post("/experiments/api-exp/transition", json={"target": "running"})
    assert r.status_code == 200 and r.json()["status"] == "running"
    r = client.post("/experiments/api-exp/transition", json={"target": "draft"})
    assert r.status_code == 422


def test_assign_is_deterministic(client):
    client.post("/experiments", json=EXPERIMENT)
    a = client.post("/experiments/api-exp/assign",
                    json={"unit_id": "u-1", "attributes": {}}).json()
    b = client.post("/experiments/api-exp/assign",
                    json={"unit_id": "u-1", "attributes": {}}).json()
    assert a == b and a["variant"] in ("control", "treatment")


def test_ingest_results_and_readout_flow(client):
    client.post("/experiments", json=EXPERIMENT)
    client.post("/experiments/api-exp/transition", json={"target": "running"})
    seq = 0
    for i in range(200):
        variant = "control" if i % 2 == 0 else "treatment"
        seq += 1
        r = client.post("/experiments/api-exp/events", json={
            "event_id": f"e-{seq}", "unit_id": f"u-{i}", "variant": variant,
            "event_name": "exposure",
        })
        assert r.json()["accepted"]
        if variant == "treatment" and i % 4 == 1:
            seq += 1
            client.post("/experiments/api-exp/events", json={
                "event_id": f"e-{seq}", "unit_id": f"u-{i}",
                "variant": variant, "event_name": "signup",
            })
    # duplicate event id is reported, not double-counted
    r = client.post("/experiments/api-exp/events", json={
        "event_id": "e-1", "unit_id": "u-0", "variant": "control",
        "event_name": "exposure",
    })
    assert r.json()["duplicate"] is True

    results = client.get("/experiments/api-exp/results").json()["results"]
    assert results["signup"]["per_variant"]["control"]["n_exposed"] == 100
    assert results["signup"]["per_variant"]["treatment"]["n_converted"] == 50

    readout = client.get("/experiments/api-exp/readout").json()
    assert readout["experiment_id"] == "api-exp"
    assert readout["decision"] in ("ship", "no_ship", "inconclusive")
    assert readout["srm"]["passed"]
