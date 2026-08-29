"""Credit accounting -- the mechanism that keeps the executor's memory bounded."""

import threading

import pytest

from easy_nn.server.blobq import BlobFeed


def test_credit_is_returned_only_after_the_consumer_moves_on():
    returned = []
    feed = BlobFeed(on_consume=returned.append)
    feed.put_blob("a")
    feed.put_blob("b")
    feed.put_end()

    stream = feed.train_stream()

    kind, payload = next(stream)
    assert (kind, payload) == ("blob", "a")
    # Still working on "a": no credit yet, so the client cannot send a third.
    assert returned == []

    kind, payload = next(stream)
    assert (kind, payload) == ("blob", "b")
    assert returned == [1], "credit for 'a' is due once its batches are consumed"

    assert next(stream) == ("end", None)
    assert returned == [1, 1], "credit for 'b' is due before the stream ends"

    with pytest.raises(StopIteration):
        next(stream)


def test_epoch_markers_do_not_consume_credit():
    returned = []
    feed = BlobFeed(on_consume=returned.append)
    feed.put_epoch_end(0)
    feed.put_blob("a")
    feed.put_end()

    items = list(feed.train_stream())
    assert items == [("epoch_end", 0), ("blob", "a"), ("end", None)]
    assert returned == [1]


def test_eval_blobs_wait_until_the_set_is_complete():
    feed = BlobFeed()
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
    feed = BlobFeed(eval_memory_limit=100)
    feed.put_eval("small", 60)
    with pytest.raises(MemoryError, match="evaluation data exceeds"):
        feed.put_eval("too big", 60)


def test_a_broken_stream_wakes_a_blocked_consumer():
    feed = BlobFeed()
    stream = feed.train_stream()
    feed.fail(ConnectionError("client went away"))

    with pytest.raises(ConnectionError, match="client went away"):
        next(stream)
