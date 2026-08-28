"""Unit tests for concurrent NIXL storage registrations."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.mem_cache.storage.nixl.nixl_registry import NixlRegistry
from sglang.test.test_utils import CustomTestCase


class _Registration:
    def __init__(self, descriptors):
        self.descriptors = descriptors

    def trim(self):
        return self


class _ConcurrentFileAgent:
    """Fake the POSIX plugin's process-wide devId registration map."""

    def __init__(self):
        self._arrival = threading.Barrier(2)
        self._lock = threading.Lock()
        self.file_devid_sets = []

    def get_reg_descs(self, descriptors, mem_type):
        return descriptors

    def register_memory(self, descriptors):
        # NixlRegistry probes path-mode support with this sentinel path.
        if descriptors[0][3].endswith("/nonexistent-nixl-probe"):
            raise RuntimeError("path mode supported")

        dev_ids = {descriptor[2] for descriptor in descriptors}
        with self._lock:
            self.file_devid_sets.append(dev_ids)

        # Force both async workers to have live registration attempts before
        # checking the process-wide devId namespace. With the old `i + 1`
        # assignment, both sets are {1, 2} and registration fails.
        self._arrival.wait(timeout=5)
        with self._lock:
            peer_sets = list(self.file_devid_sets)
        if any(dev_ids & peer for peer in peer_sets if peer is not dev_ids):
            raise RuntimeError("duplicate live FILE devId")
        return _Registration(descriptors)

    def deregister_memory(self, registration):
        return None


class _FileManager:
    use_direct_io = False


class TestNixlRegistry(CustomTestCase):
    def test_concurrent_path_mode_registrations_use_disjoint_devids(self):
        agent = _ConcurrentFileAgent()
        registry = NixlRegistry(agent, "FILE", _FileManager())
        results = [False, False]

        def register(worker_id):
            buffers = [(0x1000 + worker_id * 0x100, 64), (0x2000, 128)]
            keys = [f"/tmp/worker-{worker_id}-a", f"/tmp/worker-{worker_id}-b"]
            with registry.storage(buffers, keys, "WRITE") as descriptors:
                results[worker_id] = descriptors is not None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(register, worker_id) for worker_id in range(2)]
            for future in futures:
                future.result(timeout=10)

        self.assertEqual(results, [True, True])
        self.assertEqual(len(agent.file_devid_sets), 2)
        self.assertTrue(agent.file_devid_sets[0].isdisjoint(agent.file_devid_sets[1]))


if __name__ == "__main__":
    unittest.main()
