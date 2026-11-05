"""Registration boundaries must preserve complete storage IO units."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.storage.nixl.registration import (
    CUOBJECT_MAX_REGISTRATION_BYTES,
    plan_registration_ranges,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRegistrationPlan(unittest.TestCase):
    def test_every_kv_page_fits_one_segment(self):
        # Include shared K/V allocations whose midpoint need not be a segment
        # boundary, convolution states, and large temporal-state slots.
        for unit in (512, 4096, 737280, 31457280):
            base, pages = 0x1000000000, 514
            ranges = plan_registration_ranges(
                base, pages * unit, unit, limit=100 * unit
            )
            cursor = base
            for ptr, length in ranges:
                self.assertEqual(ptr, cursor)
                self.assertLessEqual(length, 100 * unit)
                cursor += length
            self.assertEqual(cursor, base + pages * unit)
            for page in range(pages):
                ptr = base + page * unit
                self.assertTrue(
                    any(p <= ptr and ptr + unit <= p + n for p, n in ranges)
                )

    def test_existing_cuda_boundaries_are_preserved(self):
        base, unit = 0x1000000000, 4096
        parents = [(base, 101 * unit), (base + 101 * unit, 203 * unit)]
        ranges = plan_registration_ranges(
            base, 304 * unit, unit, parents, limit=64 * unit
        )
        for ptr, size in ranges:
            self.assertTrue(any(p <= ptr and ptr + size <= p + n for p, n in parents))
        with self.assertRaisesRegex(ValueError, "splits a storage IO unit"):
            plan_registration_ranges(base, 304 * unit, 2 * unit, parents)

    def test_large_pool_and_short_tail(self):
        base, unit = 0x1000000000, 30 * 1024**2
        ranges = plan_registration_ranges(base, 257 * unit, unit)
        self.assertGreater(len(ranges), 1)
        self.assertEqual(sum(n for _, n in ranges), 257 * unit)
        self.assertTrue(all(n <= CUOBJECT_MAX_REGISTRATION_BYTES for _, n in ranges))
        self.assertTrue(all(n % unit == 0 for _, n in ranges))

    def test_incompatible_or_unknown_ranges_are_rejected(self):
        for parents in ([], [(4096, 4096)], [(0, 8192), (4096, 4096)]):
            with self.assertRaises(ValueError):
                plan_registration_ranges(0, 8192, 4096, parents)
        for unit in (None, 8192):
            with self.assertRaises(ValueError):
                plan_registration_ranges(0, 8192, unit, limit=4096)
        self.assertEqual(plan_registration_ranges(0, 0, None), [])

    def test_one_native_registration_list_owns_all_segments(self):
        try:
            from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
        except ImportError:
            self.skipTest("NIXL is not installed")
        storage = HiCacheNixl.__new__(HiCacheNixl)
        storage._segmented_registration = True
        storage._host_regs = []
        storage.agent = SimpleNamespace(
            get_reg_descs=mock.Mock(side_effect=lambda desc, kind: desc),
            register_memory=mock.Mock(side_effect=lambda desc: desc),
        )
        base, unit = 0x1000000000, 30 * 1024**2
        storage._pre_register_host(base, 257 * unit, "state", granularity=unit)
        registration = storage._host_regs[0]
        self.assertEqual(len(storage._host_regs), 1)
        self.assertEqual(len(registration), 2)
        self.assertEqual(sum(desc[1] for desc in registration), 257 * unit)
        storage.agent.register_memory.assert_called_once()
        storage._host_regs.clear()
        storage.agent.register_memory.side_effect = RuntimeError("native batch failed")
        with self.assertRaisesRegex(RuntimeError, "pre-register"):
            storage._pre_register_host(base, 257 * unit, "state", granularity=unit)
        self.assertEqual(storage._host_regs, [])


if __name__ == "__main__":
    unittest.main()
