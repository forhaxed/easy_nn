"""Credit accounting -- the mechanism that keeps the executor's memory bounded."""

import threading

import pytest

from easy_nn.server.blobq import WorkQueue


def test_credit_is_counted_in_units_of_work_not_messages():
    returned = []
    feed = WorkQueue(on_consume=returned.append)
    feed.put("first message", units=3)
    feed.put("second message", units=2)
    feed.put_end()

    stream = feed.stream()
    payload, units = next(stream)
    assert (payload, units) == ("first message", 3)
    assert returned == [], "nothing consumed yet"

    # The trainer credits back per sample as it works through them.
    feed.consumed(1)
    feed.consumed(1)
    assert returned == [1, 1]
    assert feed.depth == 3, "3 of 5 units still outstanding"

    feed.consumed(1)
    payload, units = next(stream)
    assert (payload, units) == ("second message", 2)
    feed.consumed(2)
    assert feed.depth == 0

    with pytest.raises(StopIteration):
        next(stream)


def test_depth_tracks_received_minus_consumed():
    feed = WorkQueue()
    assert feed.depth == 0
    feed.put("a", units=8)
    assert feed.depth == 8
    feed.consumed(5)
    assert feed.depth == 3


def test_zero_or_negative_credit_is_ignored():
    returned = []
    feed = WorkQueue(on_consume=returned.append)
    feed.consumed(0)
    feed.consumed(-4)
    assert returned == []
    assert feed.depth == 0


def test_time_spent_waiting_for_work_is_measured():
    """The one number that says whether the network is the bottleneck."""
    feed = WorkQueue()
    collected = []

    def consume():
        collected.extend(feed.stream())

    reader = threading.Thread(target=consume)
    reader.start()
    time_before = feed.wait_seconds

    threading.Event().wait(0.3)  # the queue is empty; the consumer is blocked
    feed.put("late", units=1)
    feed.put_end()
    reader.join(timeout=5)

    assert collected == [("late", 1)]
    assert feed.wait_seconds > time_before + 0.2


def test_eval_blobs_wait_until_the_set_is_complete():
    feed = WorkQueue()
    collected = []

    reader = threading.Thread(target=lambda: collected.extend(feed.eval_blobs()))
    reader.start()
    assert reader.is_alive(), "eval must not be run against a half-sent set"

    feed.put_eval("one", 10)
    feed.put_eval("two", 10)
    feed.eval_done()
    reader.join(timeout=5)

    assert collected == ["one", "two"]


def test_oversized_eval_set_is_refused_rather_than_swallowed():
    feed = WorkQueue(eval_memory_limit=100)
    feed.put_eval("small", 60)
    with pytest.raises(MemoryError, match="evaluation data exceeds"):
        feed.put_eval("too big", 60)


def test_a_broken_stream_wakes_a_blocked_consumer():
    feed = WorkQueue()
    stream = feed.stream()
    feed.fail(ConnectionError("client went away"))

    with pytest.raises(ConnectionError, match="client went away"):
        next(stream)
