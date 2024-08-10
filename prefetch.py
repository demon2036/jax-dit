import collections
import functools
import itertools
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np


def convert_to_global_array(x, x_sharding):
    b, *res = x.shape
    # x = np.array(x)

    per_replica_batches_x = jnp.split(x, jax.local_device_count())

    global_batch_shape_x = (b * jax.process_count(), *res)

    # s = sorted(x_sharding.addressable_devices, key=lambda x: (x.coords[1], x.coords[0]))
    s = sorted(x_sharding.addressable_devices, key=lambda x:  x.coords[0])

    global_batch_array_x = jax.make_array_from_single_device_arrays(
        global_batch_shape_x, sharding=x_sharding,
        arrays=[
            jax.device_put(batch, device)
            for batch, device in zip(per_replica_batches_x, s)
        ]
    )

    # s = x_sharding.addressable_devices
    #
    # for batch, device in zip(per_replica_batches_x, x_sharding.addressable_devices):
    #     if jax.process_index() == 0:
    #         print(device, device.coords, type(device.coords))
    #
    # print()
    #
    #
    #
    # for batch, device in zip(per_replica_batches_x, s):
    #     if jax.process_index() == 0:
    #         print(device, device.coords, type(device.coords))
    #
    # while True:
    #     pass
    return global_batch_array_x


def prefetch_to_device(
        iterator,
        size: int,
        data_sharding=None,
):
    """Iterates data and transfers it to devices creating jax.Arrays.

  This utility takes an iterator and returns a new iterator which fills a
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.
  This is similar to `flax.jax_utils.prefetch_to_device` but works with `pjit`.

  Args:
    iterator: An iterator that returns a PyTree of ndarrays.
    size: The size of the prefetch buffer.
    mesh: If given, shards the arrays using this mesh. If None, uses the active
      mesh.

  Yields:
    The original items from the iterator where each ndarray is now sharded as
    specified by `axis_resources`.
    :param size:
    :param iterator:
    :param data_sharding:
  """

    if size and size > 0:
        # We fill items to this queue, and pop from it when a new item is yielded.
        queue = collections.deque()

        def enqueue(n):
            for data in itertools.islice(iterator, n):
                data = jax.tree_util.tree_map(functools.partial(convert_to_global_array, x_sharding=data_sharding),
                                              data)
                queue.append(data)

        enqueue(size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    else:
        # If size is None, 0 or negative, simply create jax.Arrays without
        # prefetching.
        for data in iterator:
            yield jax.tree_util.tree_map(data_sharding, data)
