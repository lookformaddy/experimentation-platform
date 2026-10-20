from datetime import timedelta

from experimentation_platform.events import EventStore

from conftest import BASE, event


def test_append_and_count(store):
    assert store.append(event(1, "u-1", "control")).accepted
    assert len(store) == 1


def test_duplicate_event_id_dropped(store):
    e = event(1, "u-1", "control")
    assert store.append(e).accepted
    result = store.append(e)
    assert not result.accepted and result.duplicate
    assert len(store) == 1
    assert store.duplicates_dropped == 1


def test_duplicate_exposure_counted_once(store):
    store.append(event(1, "u-1", "control"))
    store.append(event(2, "u-1", "control"))
    assert store.duplicate_exposures == 1
    assert store.exposed_units("exp-1") == {"control": {"u-1"}}


def test_first_exposure_variant_wins(store):
    store.append(event(1, "u-1", "control"))
    store.append(event(2, "u-1", "treatment"))
    assert store.exposed_units("exp-1")["control"] == {"u-1"}
    assert "treatment" not in store.exposed_units("exp-1")


def test_late_event_flagged(store):
    store.append(event(1, "u-1", "control", when=BASE))
    store.append(event(2, "u-2", "control", when=BASE + timedelta(hours=5)))
    # 5h after the first event but 4h behind the watermark: late (>1h window)
    late = event(3, "u-3", "control", when=BASE + timedelta(hours=1))
    result = store.append(late)
    assert result.accepted and result.late
    assert store.is_late(late)
    assert store.late_count == 1


def test_events_for_can_exclude_late(store):
    store.append(event(1, "u-1", "control", when=BASE))
    store.append(event(2, "u-2", "control", when=BASE + timedelta(hours=5)))
    store.append(event(3, "u-3", "control", when=BASE + timedelta(hours=1)))
    assert len(store.events_for("exp-1")) == 3
    assert len(store.events_for("exp-1", include_late=False)) == 2


def test_jsonl_roundtrip(store, tmp_path):
    store.extend([
        event(1, "u-1", "control"),
        event(2, "u-1", "control", name="signup"),
        event(3, "u-2", "treatment", name="revenue", value=12.5),
    ])
    path = tmp_path / "events.jsonl"
    assert store.save_jsonl(path) == 3
    loaded = EventStore.load_jsonl(path)
    assert len(loaded) == 3
    assert loaded.all_events() == store.all_events()
