"""Split cuObject registrations on complete storage IO units."""

import math

# cuobjclient.h: CUOBJ_MAX_MEMORY_REG_SIZE (CUDA 13.3 SDK).
CUOBJECT_MAX_REGISTRATION_BYTES = 4 * 1024**3 - 64 * 1024
REGISTRATION_GRANULARITY_ATTR = "_sglang_host_registration_granularity_bytes"
CUDA_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"


def plan_registration_ranges(
    base: int,
    size: int,
    granularity: int | None,
    cuda_ranges: list[tuple[int, int]] | None = None,
    *,
    limit: int = CUOBJECT_MAX_REGISTRATION_BYTES,
    alignment: int = 4096,
) -> list[tuple[int, int]]:
    """Partition a physical buffer without splitting a storage IO unit.

    Existing CUDA registrations are hard boundaries. The final segment need
    not fill the limit, and the logical tensor does not change shape or size.
    """
    if base < 0 or size < 0 or limit <= 0 or alignment <= 0:
        raise ValueError("Invalid host registration range or limit")
    if size == 0:
        return []
    if granularity is not None and (granularity <= 0 or size % granularity):
        raise ValueError("Host buffer size must contain complete registration units")
    parents = sorted(cuda_ranges) if cuda_ranges is not None else [(base, size)]
    result = []
    cursor = base
    for start, length in parents:
        if start != cursor or length <= 0 or start + length > base + size:
            raise ValueError(
                "CUDA registrations must cover the buffer without gaps/overlap"
            )
        if granularity and ((start - base) % granularity or length % granularity):
            raise ValueError("A CUDA registration boundary splits a storage IO unit")
        if length > limit and granularity is None:
            raise ValueError(
                "Large accelerated host buffer has no registration granularity"
            )
        if granularity and granularity > limit:
            raise ValueError(
                "One storage IO unit exceeds the cuObject registration limit"
            )
        step = limit
        if length > limit:
            unit = math.lcm(granularity, alignment)
            step = limit // unit * unit
            if step == 0 or start % alignment:
                raise ValueError("Cannot align cuObject registration boundaries safely")
        offset = 0
        while offset < length:
            chunk = min(step, length - offset)
            result.append((start + offset, chunk))
            offset += chunk
        cursor += length
    if cursor != base + size:
        raise ValueError("CUDA registrations do not cover the host buffer")
    return result
