import asyncio
from asyncio.futures import Future
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, cast

import pytest
from pytest_mock import MockerFixture

from strawberry.dataloader import AbstractCache, DataLoader
from strawberry.exceptions import WrongNumberOfResultsReturned

IDXType = Callable[[List[int]], Awaitable[List[int]]]


async def idx(keys: List[int]) -> List[int]:
    return keys


@pytest.mark.asyncio
async def test_loading():
    loader = DataLoader(load_fn=idx)

    value_a = await loader.load(1)
    value_b = await loader.load(2)
    value_c = await loader.load(3)

    assert value_a == 1
    assert value_b == 2
    assert value_c == 3

    values = await loader.load_many([1, 2, 3, 4, 5, 6])

    assert values == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_gathering(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader))

    [value_a, value_b, value_c] = await asyncio.gather(
        loader.load(1),
        loader.load(2),
        loader.load(3),
    )

    mock_loader.assert_called_once_with([1, 2, 3])

    assert value_a == 1
    assert value_b == 2
    assert value_c == 3


@pytest.mark.asyncio
async def test_max_batch_size(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader), max_batch_size=2)

    [value_a, value_b, value_c] = await asyncio.gather(
        loader.load(1),
        loader.load(2),
        loader.load(3),
    )

    mock_loader.assert_has_calls([mocker.call([1, 2]), mocker.call([3])])  # type: ignore

    assert value_a == 1
    assert value_b == 2
    assert value_c == 3


@pytest.mark.asyncio
async def test_error():
    async def idx(keys: List[int]) -> List[Union[int, ValueError]]:
        return [ValueError()]

    loader = DataLoader(load_fn=idx)

    with pytest.raises(ValueError):
        await loader.load(1)


@pytest.mark.asyncio
async def test_error_and_values():
    async def idx(keys: List[int]) -> List[Union[int, ValueError]]:
        return [2] if keys == [2] else [ValueError()]

    loader = DataLoader(load_fn=idx)

    with pytest.raises(ValueError):
        await loader.load(1)

    assert await loader.load(2) == 2


@pytest.mark.asyncio
async def test_when_raising_error_in_loader():
    async def idx(keys: List[int]) -> List[Union[int, ValueError]]:
        raise ValueError

    loader = DataLoader(load_fn=idx)

    with pytest.raises(ValueError):
        await loader.load(1)

    a = loader.load(1)
    a.cancel()
    with pytest.raises(ValueError):
        await asyncio.gather(
            loader.load(1),
            loader.load(2),
            loader.load(3),
        )


@pytest.mark.asyncio
async def test_returning_wrong_number_of_results():
    async def idx(keys: List[int]) -> List[int]:
        return [1, 2]

    loader = DataLoader(load_fn=idx)

    with pytest.raises(
        WrongNumberOfResultsReturned,
        match=(
            "Received wrong number of results in dataloader, "
            "expected: 1, received: 2"
        ),
    ):
        await loader.load(1)


@pytest.mark.asyncio
async def test_caches_by_id(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader), cache=True)

    a = loader.load(1)
    b = loader.load(1)

    assert a == b

    assert await a == 1
    assert await b == 1

    mock_loader.assert_called_once_with([1])


@pytest.mark.asyncio
async def test_caches_by_id_when_loading_many(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader), cache=True)

    a = loader.load(1)
    b = loader.load(1)

    assert a == b

    assert [1, 1] == await asyncio.gather(a, b)

    mock_loader.assert_called_once_with([1])


@pytest.mark.asyncio
async def test_cache_disabled(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader), cache=False)

    a = loader.load(1)
    b = loader.load(1)

    assert a != b

    assert await a == 1
    assert await b == 1

    mock_loader.assert_has_calls([mocker.call([1, 1])])  # type: ignore


@pytest.mark.asyncio
async def test_cache_disabled_immediate_await(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)

    loader = DataLoader(load_fn=cast(IDXType, mock_loader), cache=False)

    a = await loader.load(1)
    b = await loader.load(1)

    assert a == b

    mock_loader.assert_has_calls([mocker.call([1]), mocker.call([1])])  # type: ignore


@pytest.mark.asyncio
async def test_prime():
    async def idx(keys: List[Union[int, float]]) -> List[Union[int, float]]:
        assert keys, "At least one key must be specified"
        return keys

    loader = DataLoader(load_fn=idx)

    # Basic behavior intact
    a1 = loader.load(1)
    assert await a1 == 1

    # Prime doesn't overrides value
    loader.prime(1, 1.1)
    loader.prime(2, 2.1)
    b1 = loader.load(1)
    b2 = loader.load(2)
    assert await b1 == 1
    assert await b2 == 2.1

    # Unless you tell it to
    loader.prime(1, 1.2, force=True)
    loader.prime(2, 2.2, force=True)
    b1 = loader.load(1)
    b2 = loader.load(2)
    assert await b1 == 1.2
    assert await b2 == 2.2

    # Preset will override pending values, but not cached values
    c2 = loader.load(2)  # This is in cache
    c3 = loader.load(3)  # This is pending
    loader.prime_many({2: 2.3, 3: 3.3}, force=True)
    assert await c2 == 2.2
    assert await c3 == 3.3

    # If we prime all keys in a batch, the load_fn is never called
    # (See assertion in idx)
    c4 = loader.load(4)
    loader.prime_many({4: 4.4})
    assert await c4 == 4.4

    # Yield to ensure the last batch has been dispatched,
    # despite all values being primed
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_prime_nocache():
    async def idx(keys: List[Union[int, float]]) -> List[Union[int, float]]:
        assert keys, "At least one key must be specified"
        return keys

    loader = DataLoader(load_fn=idx, cache=False)

    # Primed value is ignored
    loader.prime(1, 1.1)
    a1 = loader.load(1)
    assert await a1 == 1

    # Unless it affects pending value in the current batch
    b1 = loader.load(2)
    loader.prime(2, 2.2)
    assert await b1 == 2.2

    # Yield to ensure the last batch has been dispatched,
    # despite all values being primed
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_clear():
    batch_num = 0

    async def idx(keys: List[int]) -> List[Tuple[int, int]]:
        """Maps key => (key, batch_num)"""
        nonlocal batch_num
        batch_num += 1
        return [(key, batch_num) for key in keys]

    loader = DataLoader(load_fn=idx)

    assert await loader.load_many([1, 2, 3]) == [(1, 1), (2, 1), (3, 1)]

    loader.clear(1)

    assert await loader.load_many([1, 2, 3]) == [(1, 2), (2, 1), (3, 1)]

    loader.clear_many([1, 2])

    assert await loader.load_many([1, 2, 3]) == [(1, 3), (2, 3), (3, 1)]

    loader.clear_all()

    assert await loader.load_many([1, 2, 3]) == [(1, 4), (2, 4), (3, 4)]


@pytest.mark.asyncio
async def test_clear_nocache():
    batch_num = 0

    async def idx(keys: List[int]) -> List[Tuple[int, int]]:
        """Maps key => (key, batch_num)"""
        nonlocal batch_num
        batch_num += 1
        return [(key, batch_num) for key in keys]

    loader = DataLoader(load_fn=idx, cache=False)

    assert await loader.load_many([1, 2, 3]) == [(1, 1), (2, 1), (3, 1)]

    loader.clear(1)

    assert await loader.load_many([1, 2, 3]) == [(1, 2), (2, 2), (3, 2)]

    loader.clear_many([1, 2])

    assert await loader.load_many([1, 2, 3]) == [(1, 3), (2, 3), (3, 3)]

    loader.clear_all()

    assert await loader.load_many([1, 2, 3]) == [(1, 4), (2, 4), (3, 4)]


@pytest.mark.asyncio
async def test_dont_dispatch_cancelled():
    async def idx(keys: List[int]) -> List[int]:
        await asyncio.sleep(0.2)
        return keys

    loader = DataLoader(load_fn=idx)

    value_a = await loader.load(1)
    # value_b will be cancelled by hand
    value_b = cast("Future[Any]", loader.load(2))
    value_b.cancel()
    # value_c will be cancelled by the timeout
    with pytest.raises(asyncio.TimeoutError):
        value_c = cast("Future[Any]", loader.load(3))
        await asyncio.wait_for(value_c, 0.1)
    value_d = await loader.load(4)

    assert value_a == 1
    assert value_d == 4

    # 2 can still be used here because a new future will be created for it
    values = await loader.load_many([1, 2, 3, 4, 5, 6])
    assert values == [1, 2, 3, 4, 5, 6]

    with pytest.raises(asyncio.CancelledError):
        value_b.result()
    with pytest.raises(asyncio.CancelledError):
        value_c.result()  # pyright: ignore

    # Try single loading results again to make sure the cancelled
    # futures are not being reused
    value_a = await loader.load(1)
    value_b = await loader.load(2)
    value_c = await loader.load(3)
    value_d = await loader.load(4)

    assert value_a == 1
    assert value_b == 2
    assert value_c == 3
    assert value_d == 4


@pytest.mark.asyncio
async def test_handles_cancelling_pending_dispatch_tasks():
    session_closed = False
    session_used_after_close = False

    async def db_load(ids: List[int]) -> Optional[List[int]]:
        nonlocal session_used_after_close
        await asyncio.sleep(0.1)
        if session_closed:
            session_used_after_close = True
            raise RuntimeError("session used after close")
        return ids

    user_loader = DataLoader(load_fn=db_load)

    async def load_user_or_raise(idx) -> int:
        return await user_loader.load(idx)

    async def load_post_or_raise(ids: List[int]) -> Optional[List[int]]:
        await asyncio.sleep(0.01)
        raise ValueError("post not found")

    task_1 = asyncio.create_task(load_user_or_raise(1))
    task_2 = asyncio.create_task(load_post_or_raise(2))

    # gather will raise error from task_2
    with pytest.raises(ValueError):
        await asyncio.gather(task_1, task_2)

    # task_2 is done because it raised an error
    assert task_2.done()

    # task_1 is not done: gather doesn't cancel tasks in case of an exception
    assert not task_1.done()
    # cancel task_1, like it would have been when used with TaskGroup
    task_1.cancel()

    user_loader.cancel_pending_tasks()

    # It should be fine to close the session now, no more tasks should be running
    session_closed = True

    assert task_1.cancelled
    assert not session_used_after_close

    # Wait for a bit longer than the load_fn to make sure the session is not used
    await asyncio.sleep(0.2)
    assert not session_used_after_close
    assert len(user_loader._dispatch_tasks_and_batches) == 0
    assert len(user_loader._scheduled_batches) == 0


@pytest.mark.asyncio
async def test_cancelling_dataloader_before_dispatch_task():
    has_loader_run = False

    async def db_load(ids: List[int]) -> Optional[List[int]]:
        nonlocal has_loader_run
        has_loader_run = True
        await asyncio.sleep(0.1)
        return ids

    dataloader = DataLoader(load_fn=db_load)
    future_1 = dataloader.load(1)
    future_2 = dataloader.load(2)

    # Cancel dataloader tasks before there was a chance to dispatch the first batch
    dataloader.cancel_pending_tasks()

    assert future_1.cancelled()
    assert future_2.cancelled()

    # Wait for a bit longer than the load_fn to make sure it's not running
    await asyncio.sleep(0.11)
    assert not has_loader_run

    # Check that cleanup was done correctly
    assert len(dataloader._scheduled_batches) == 0
    assert len(dataloader._dispatch_tasks_and_batches) == 0


@pytest.mark.asyncio
async def test_cancelling_dataloader_after_dispatch_task_but_before_dispatch():
    has_loader_run = False

    async def db_load(ids: List[int]) -> Optional[List[int]]:
        nonlocal has_loader_run
        has_loader_run = True
        await asyncio.sleep(0.1)
        return ids

    dataloader = DataLoader(load_fn=db_load)
    future_1 = dataloader.load(1)
    future_2 = dataloader.load(2)

    # Wait for the first batch task to be created, but not yet dispatched
    await asyncio.sleep(0)
    assert len(dataloader._scheduled_batches) == 0
    assert len(dataloader._dispatch_tasks_and_batches) == 1
    assert not dataloader.batch.dispatched

    dataloader.cancel_pending_tasks()

    assert future_1.cancelled()
    assert future_2.cancelled()

    # Wait for a bit longer than the load_fn to make sure it's not running
    await asyncio.sleep(0.11)

    # Check that cleanup was done correctly
    assert not has_loader_run
    assert len(dataloader._scheduled_batches) == 0
    assert len(dataloader._dispatch_tasks_and_batches) == 0


@pytest.mark.asyncio
async def test_handling_canceled_load_fn():
    loop = asyncio.get_event_loop()
    load_future = loop.create_future()

    async def db_load(ids: List[int]) -> Optional[List[int]]:
        await load_future
        return ids

    dataloader = DataLoader(load_fn=db_load)
    future_1 = dataloader.load(1)
    future_2 = dataloader.load(2)

    load_future.cancel()

    await asyncio.sleep(0.01)

    assert future_1.cancelled()
    assert future_2.cancelled()

    # Check that cleanup was done correctly
    assert len(dataloader._scheduled_batches) == 0
    assert len(dataloader._dispatch_tasks_and_batches) == 0


@pytest.mark.asyncio
async def test_load_after_cancel():
    session_closed = False
    session_used_after_close = False

    async def db_load(ids: List[int]) -> Optional[List[int]]:
        nonlocal session_used_after_close
        await asyncio.sleep(0.01)
        if session_closed:
            session_used_after_close = True
            raise RuntimeError("session used after close")
        return ids

    dataloader = DataLoader(load_fn=db_load)

    async def complex_action():
        await asyncio.sleep(0.01)
        await dataloader.load(2)

    future_1 = dataloader.load(1)
    future_2 = asyncio.create_task(complex_action())

    dataloader.cancel_pending_tasks()
    session_closed = True

    assert future_1.cancelled()

    await asyncio.sleep(0.2)
    assert future_2.done()
    assert not session_used_after_close

    await asyncio.sleep(0.1)
    # Check that cleanup was done correctly
    assert len(dataloader._scheduled_batches) == 0
    assert len(dataloader._dispatch_tasks_and_batches) == 0


@pytest.mark.asyncio
async def test_cache_override():
    class TestCache(AbstractCache[int, int]):
        def __init__(self):
            self.cache: Dict[int, Future[int]] = {}

        def get(self, key: int) -> Optional["Future[int]"]:
            return self.cache.get(key)

        def set(self, key: int, value: "Future[int]") -> None:
            self.cache[key] = value

        def delete(self, key: int) -> None:
            del self.cache[key]

        def clear(self) -> None:
            self.cache.clear()

    custom_cache = TestCache()
    loader = DataLoader(load_fn=idx, cache_map=custom_cache)

    await loader.load(1)
    await loader.load(2)
    await loader.load(3)

    assert len(custom_cache.cache) == 3
    assert await custom_cache.cache[1] == 1
    assert await custom_cache.cache[2] == 2
    assert await custom_cache.cache[3] == 3

    loader.clear(1)
    assert len(custom_cache.cache) == 2
    assert sorted(list(custom_cache.cache.keys())) == [2, 3]

    loader.clear_all()
    assert len(custom_cache.cache) == 0
    assert not list(custom_cache.cache.keys())

    await loader.load(1)
    await loader.load(2)
    await loader.load(3)

    loader.clear_many([1, 2])
    assert len(custom_cache.cache) == 1
    assert list(custom_cache.cache.keys()) == [3]

    data = await loader.load(3)
    assert data == 3

    loader.prime(3, 4)
    assert await custom_cache.cache[3] == 3

    loader.prime(3, 4, True)
    assert await custom_cache.cache[3] == 4

    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        await loader.load([1, 2, 3])  # type: ignore

    data = await loader.load((1, 2, 3))  # type: ignore
    assert await custom_cache.get((1, 2, 3)) == data  # type: ignore


@pytest.mark.asyncio
async def test_custom_cache_key_fn():
    def custom_cache_key(key: List[int]) -> str:
        return ",".join(str(k) for k in key)

    loader = DataLoader(load_fn=idx, cache_key_fn=custom_cache_key)
    data = await loader.load([1, 2, "test"])
    assert [1, 2, "test"] == data


@pytest.mark.asyncio
async def test_user_class_custom_cache_key_fn():
    class CustomData:
        def __init__(self, custom_id: int, name: str):
            self.id: int = custom_id
            self.name: str = name

    def custom_cache_key(key: CustomData) -> int:
        return key.id

    loader = DataLoader(load_fn=idx, cache_key_fn=custom_cache_key)
    data1 = await loader.load(CustomData(1, "Nick"))
    data2 = await loader.load(CustomData(1, "Nick"))
    assert data1 == data2

    data2 = await loader.load(CustomData(2, "Jane"))
    assert data1 != data2


def test_works_when_created_in_a_different_loop(mocker: MockerFixture):
    mock_loader = mocker.Mock(side_effect=idx)
    loader = DataLoader(load_fn=cast(IDXType, mock_loader), cache=False)

    loop = asyncio.new_event_loop()

    async def run():
        return await loader.load(1)

    data = loop.run_until_complete(run())

    assert data == 1

    mock_loader.assert_called_once_with([1])
