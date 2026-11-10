"""Exercise actual candidate adapter methods without importing GPU packages.

Run this file directly, or discover it with unittest/pytest. By default it reads
this checkout's backend; --source can select a separately prepared source port.

Only framework annotations/decorators are removed from the selected methods.
The production method bodies run unchanged. CPU byte tensors and a synchronous
pointer client stand in for torch/MoRI. These tests do not claim real RDMA,
standalone-service registration, device placement or serving correctness.
"""

import argparse
import ast
import ctypes
import logging
import struct
import threading
import types
import unittest
from pathlib import Path

# The extracted-method protocol fixture also runs with only the standard
# library. In a SGLang CI environment, use the normal CI registration and base.
# Do not hide missing transitive dependencies in an installed SGLang package.
try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError as exc:
    if exc.name != "sglang":
        raise
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(**kwargs):
        """CI registration is a runtime no-op in standalone fixture runs."""


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/storage/umbp/umbp_store.py"
)


class ByteTensor:
    dtype = "uint8"

    def __init__(self, size, *, storage=None, offset=0):
        self.storage = bytearray(size) if storage is None else storage
        self.offset = offset
        self.size = size
        self.pinned = False

    @classmethod
    def from_bytes(cls, payload):
        tensor = cls(len(payload))
        tensor.storage[:] = payload
        return tensor

    def numel(self):
        return self.size

    def pin_memory(self):
        self.pinned = True
        return self

    def data_ptr(self):
        return ctypes.addressof(ctypes.c_ubyte.from_buffer(self.storage)) + self.offset

    def untyped_storage(self):
        return types.SimpleNamespace(
            data_ptr=lambda: self.data_ptr() - self.offset,
            nbytes=lambda: len(self.storage),
        )

    def __getitem__(self, index):
        start, stop, step = index.indices(self.size)
        assert step == 1
        return type(self)(
            stop - start, storage=self.storage, offset=self.offset + start
        )

    def copy_(self, other):
        assert self.size == other.size
        self.storage[self.offset : self.offset + self.size] = other.to_bytes()
        return self

    def fill_(self, value):
        self.storage[self.offset : self.offset + self.size] = bytes([value]) * self.size
        return self

    def tolist(self):
        return list(self.to_bytes())

    def to_bytes(self):
        return bytes(self.storage[self.offset : self.offset + self.size])


class HostIndices(list):
    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def tolist(self):
        return list(self)


class FakeSharedAllocator:
    _standalone_process = True

    def __init__(self):
        self.allocations = []

    def allocate(self, dims, dtype, device):
        assert dtype == "uint8" and device == "cpu"
        # Model the allocator's mapping being larger than the requested view.
        storage = bytearray(dims[0] + 64)
        tensor = ByteTensor(dims[0], storage=storage)
        self.allocations.append(tensor)
        return tensor

    def mapped_size_for(self, ptr):
        return next(len(t.storage) for t in self.allocations if t.data_ptr() == ptr)


class WholePagePool:
    layout = "page_first"
    page_size = 1
    size_per_token = 32
    slot_state_device_tensors = (object(),)
    conv_buffer = [object()]
    temporal_state_elem_size = 1

    def __init__(self, capacity=5, *, schema=None, layout=None):
        self.size = capacity
        self.schema = schema or (
            ("torch.float32", (1,)),
            ("torch.float32", (1,)),
            ("torch.bfloat16", (2,)),
            ("torch.int64", (2,)),
            ("torch.int32", (1,)),
        )
        if layout is not None:
            self.layout = layout
        # Includes fake FP32 temporal/conv, BF16 PLE history and INT64 token
        # history bytes. The adapter must preserve all bytes, not cast types.
        self.pages = {
            i: struct.pack("<ffHHqqI", i + 0.25, i + 0.5, 0x3F80, 0x4000, i, i + 1, 17)
            for i in range(capacity)
        }
        self.restored = []

    def _iter_page_tensors(self, index):
        for dtype, shape in self.schema:
            yield types.SimpleNamespace(dtype=dtype, shape=shape)

    def get_data_page(self, index, flat=True):
        return ByteTensor.from_bytes(self.pages[index])

    def set_from_flat_data_page(self, index, tensor):
        payload = tensor.to_bytes()
        if len(payload) != self.size_per_token:
            raise ValueError("invalid complete checkpoint size")
        self.pages[index] = payload
        self.restored.append(index)

    def get_page_buffer_meta(self, indices):
        raise NotImplementedError("Mamba zero-copy omits registered slot side states")


class ZeroCopyPool:
    page_size = 1
    components = (object(),)
    slot_state_device_tensors = ()

    def __init__(self):
        self.pages = [ByteTensor.from_bytes(bytes([i + 1]) * 8) for i in range(5)]
        self.meta_calls = 0

    def get_page_buffer_meta(self, indices):
        self.meta_calls += 1
        return [self.pages[i].data_ptr() for i in indices], [8] * len(indices)


class FakeClient:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.registrations = []
        self.register_ok = True
        self.malformed = None
        self.race_once = False
        self.get_mode = "strict"

    def is_distributed(self):
        return False

    def register_memory(self, ptr, size):
        self.registrations.append((ptr, size))
        return self.register_ok

    def batch_exists(self, keys):
        result = [key in self.objects for key in keys]
        return result[:-1] if self.malformed == "exists" else result

    def batch_put_from_ptr(self, keys, ptrs, sizes):
        self.calls.append(("put", list(keys), list(sizes)))
        for key, ptr, size in zip(keys, ptrs, sizes):
            self.objects[key] = ctypes.string_at(ptr, size)
        if self.race_once and any("_mamba_wholepage_v2_" in key for key in keys):
            self.race_once = False
            return [False] * len(keys)
        result = [True] * len(keys)
        return result[:-1] if self.malformed == "put" else result

    def batch_get_into_ptr(self, keys, ptrs, sizes):
        self.calls.append(("get", list(keys), list(sizes)))
        result = []
        for key, ptr, size in zip(keys, ptrs, sizes):
            payload = self.objects.get(key)
            ok = payload is not None and (
                len(payload) <= size
                if self.get_mode == "capacity"
                else len(payload) == size
            )
            if ok and self.get_mode != "async-success":
                ctypes.memmove(ptr, payload, len(payload))
            elif payload is not None and self.get_mode == "partial-failure":
                ctypes.memmove(ptr, payload, min(size, len(payload)))
            result.append(ok)
        return result[:-1] if self.malformed == "get" else result


def load_store(source_path):
    tree = ast.parse(source_path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UMBPStore"
    )
    names = {
        "_needs_wholepage_io",
        "_ensure_wholepage_staging",
        "_wholepage_io_v2",
        "_new_wholepage_shared_allocator",
        "_verify_wholepage_backend_contract",
        "_wholepage_schema_digest",
        "_get_hybrid_page_component_keys",
        "_batch_io_v2",
        "batch_exists_v2",
        "_register_host_buffer_for_zero_copy",
        "_pool_physical_buffers",
        "_buffer_extent",
    }
    scope = {
        "torch": types.SimpleNamespace(
            uint8="uint8",
            empty=lambda size, **kwargs: ByteTensor(size),
            tensor=lambda values, **kwargs: ByteTensor.from_bytes(bytes(values)),
        ),
        "SimpleNamespace": types.SimpleNamespace,
        "logger": logging.getLogger("test.umbp"),
        "PoolName": types.SimpleNamespace(KV="kv", MAMBA="mamba", DRAFT="draft"),
        "PoolHitPolicy": types.SimpleNamespace(
            ALL_PAGES="all", TRAILING_PAGES="trailing"
        ),
        "PoolTransferResult": lambda count, hit_count: types.SimpleNamespace(
            count=count, hit_count=hit_count
        ),
    }
    methods = {}
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in names:
            continue
        static = any(
            isinstance(d, ast.Name) and d.id == "staticmethod"
            for d in method.decorator_list
        )
        method.decorator_list = []
        method.returns = None
        for arg in (
            *method.args.posonlyargs,
            *method.args.args,
            *method.args.kwonlyargs,
        ):
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        exec(compile(module, str(source_path), "exec"), scope)
        methods[method.name] = (
            staticmethod(scope[method.name]) if static else scope[method.name]
        )
    missing = names - methods.keys()
    if missing:
        raise AssertionError(f"Candidate methods missing: {sorted(missing)}")
    return type("ExtractedCandidateUMBPStore", (), methods)


class AdapterTests(CustomTestCase):
    store_type = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.store_type is None:
            cls.store_type = load_store(DEFAULT_SOURCE)

    def setUp(self):
        self.store = self.store_type()
        self.store.client = FakeClient()
        self.pool = WholePagePool()
        self.zero_pool = ZeroCopyPool()
        self.store.registered_pools = {"mamba": self.pool, "indexer": self.zero_pool}
        self.store.mha_suffix = "tp0"
        self.store.mla_suffix = "tp0"
        self.store.kv_rank_suffix = "tp0"
        self.store.config_prefix = "model-tag"
        self.store.is_mla_backend = False
        self.store._wholepage_staging = None
        self.store._wholepage_staging_allocator = None
        self.store._wholepage_backend_contract_verified = False
        self.store._new_wholepage_shared_allocator = FakeSharedAllocator
        self.store._wholepage_staging_cap_bytes = 64
        self.store._wholepage_io_lock = threading.Lock()
        self.store._registered_regions = set()
        self.store._is_standalone_process = True
        self.store._disable_zero_copy_register = False
        self.store._kv_anchor_is_logical = True

    def transfer(self, name="mamba", count=3):
        return types.SimpleNamespace(
            name=name,
            keys=[f"prefix{i}" for i in range(count)],
            host_indices=HostIndices(range(count)),
            hit_policy="trailing",
        )

    def test_complete_checkpoint_roundtrip_and_registered_chunk_bound(self):
        transfer = self.transfer()
        original = dict(self.pool.pages)
        self.assertEqual(
            self.store._batch_io_v2([transfer], True), {"mamba": [True] * 3}
        )
        staging = self.store._wholepage_staging
        self.assertFalse(staging.pinned)
        self.assertIsInstance(
            self.store._wholepage_staging_allocator, FakeSharedAllocator
        )
        self.assertEqual(self.store.client.registrations, [(staging.data_ptr(), 128)])
        actual_puts = [
            c
            for c in self.store.client.calls
            if c[0] == "put" and "_mamba_wholepage_v2_" in c[1][0]
        ]
        self.assertEqual([len(c[1]) for c in actual_puts], [2, 1])
        self.assertTrue(self.store._wholepage_backend_contract_verified)
        self.pool.pages.update({i: b"x" * 32 for i in range(3)})
        self.assertEqual(
            self.store._batch_io_v2([transfer], False), {"mamba": [True] * 3}
        )
        for i in range(3):
            self.assertEqual(self.pool.pages[i], original[i])
        self.assertIs(self.store._wholepage_staging, staging)
        self.assertEqual(len(self.store.client.registrations), 1)

    def test_exists_uses_same_wholepage_key_and_legacy_keys_miss(self):
        transfer = self.transfer(count=2)
        self.store.client.objects["model-tag_prefix0_tp0_temporal"] = b"old"
        self.store.client.objects["model-tag_prefix0_tp0_mamba_wholepage_v1"] = (
            b"v1" * 16
        )
        self.assertEqual(self.store.batch_exists_v2(transfer.keys, [transfer]).count, 0)
        self.store._batch_io_v2([transfer], True)
        expected, multiplier = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.assertEqual(multiplier, 1)
        for key in expected:
            self.assertRegex(
                key, r"^model-tag_prefix[01]_tp0_mamba_wholepage_v2_[0-9a-f]{64}$"
            )
        self.assertTrue(all(key in self.store.client.objects for key in expected))
        self.assertEqual(self.store.batch_exists_v2(transfer.keys, [transfer]).count, 2)

    def test_mixed_pools_keep_zero_copy_path(self):
        transfers = [self.transfer(), self.transfer("indexer")]
        self.assertEqual(
            self.store._batch_io_v2(transfers, True),
            {"mamba": [True] * 3, "indexer": [True] * 3},
        )
        self.assertEqual(self.zero_pool.meta_calls, 1)
        self.assertIn("model-tag_prefix0_tp0_indexer", self.store.client.objects)

    def _assert_changed_schema_is_safe_miss(self, replacement):
        transfer = self.transfer(count=1)
        old_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.assertEqual(self.store._batch_io_v2([transfer], True), {"mamba": [True]})
        replacement.pages[0] = b"r" * 32
        self.store.registered_pools["mamba"] = replacement
        new_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.assertNotEqual(new_keys, old_keys)
        self.assertEqual(self.store.batch_exists_v2(transfer.keys, [transfer]).count, 0)
        self.assertEqual(self.store._batch_io_v2([transfer], False), {"mamba": [False]})
        self.assertEqual(replacement.pages[0], b"r" * 32)
        self.assertEqual(replacement.restored, [])

    def test_same_size_dtype_change_is_miss_without_destination_mutation(self):
        schema = list(self.pool.schema)
        schema[2] = ("torch.float16", schema[2][1])
        self._assert_changed_schema_is_safe_miss(WholePagePool(schema=schema))

    def test_layout_change_is_miss_without_destination_mutation(self):
        self._assert_changed_schema_is_safe_miss(WholePagePool(layout="layer_first"))

    def test_same_size_shape_change_is_miss_without_destination_mutation(self):
        schema = list(self.pool.schema)
        schema[2] = (schema[2][0], (1, 2))
        self._assert_changed_schema_is_safe_miss(WholePagePool(schema=schema))

    def test_component_order_change_is_miss_without_destination_mutation(self):
        schema = list(self.pool.schema)
        schema[2], schema[3] = schema[3], schema[2]
        self._assert_changed_schema_is_safe_miss(WholePagePool(schema=schema))

    def test_same_schema_different_capacity_reuses_key_and_payload(self):
        transfer = self.transfer(count=1)
        old_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        payload = self.pool.pages[0]
        self.assertEqual(self.store._batch_io_v2([transfer], True), {"mamba": [True]})
        replacement = WholePagePool(capacity=17)
        replacement.pages[0] = b"r" * 32
        self.store.registered_pools["mamba"] = replacement
        new_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.assertEqual(new_keys, old_keys)
        self.assertEqual(self.store.batch_exists_v2(transfer.keys, [transfer]).count, 1)
        self.assertEqual(self.store._batch_io_v2([transfer], False), {"mamba": [True]})
        self.assertEqual(replacement.pages[0], payload)

    def test_page_size_changes_schema_namespace(self):
        transfer = self.transfer(count=1)
        old_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.pool.page_size = 2
        new_keys, _ = self.store._get_hybrid_page_component_keys(
            transfer.keys, transfer
        )
        self.assertNotEqual(new_keys, old_keys)
        self.assertEqual(self.store.client.calls, [])

    def test_missing_serializer_metadata_fails_before_checkpoint_io(self):
        self.pool._iter_page_tensors = None
        with self.assertRaisesRegex(ValueError, "ordered serializer tensor metadata"):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertEqual(self.store.client.calls, [])
        self.assertIsNone(self.store._wholepage_staging)

    def test_missing_layout_fails_before_checkpoint_io(self):
        self.pool.layout = None
        with self.assertRaisesRegex(ValueError, "host serializer layout"):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertEqual(self.store.client.calls, [])

    def test_missing_or_wrong_size_page_never_updates_destination(self):
        transfer = self.transfer(count=2)
        self.store._batch_io_v2([transfer], True)
        keys, _ = self.store._get_hybrid_page_component_keys(transfer.keys, transfer)
        del self.store.client.objects[keys[0]]
        self.store.client.objects[keys[1]] = b"short"
        self.pool.restored.clear()
        self.assertEqual(
            self.store._batch_io_v2([transfer], False), {"mamba": [False, False]}
        )
        self.assertEqual(self.pool.restored, [])

    def test_existing_set_is_idempotent_and_duplicate_race_succeeds(self):
        transfer = self.transfer(count=2)
        self.store.client.race_once = True
        self.assertEqual(
            self.store._batch_io_v2([transfer], True), {"mamba": [True, True]}
        )
        calls = len(self.store.client.calls)
        self.assertEqual(
            self.store._batch_io_v2([transfer], True), {"mamba": [True, True]}
        )
        self.assertEqual(len(self.store.client.calls), calls)

    def test_checkpoint_larger_than_cap_is_rejected_before_io(self):
        self.store._wholepage_staging_cap_bytes = 31
        with self.assertRaisesRegex(ValueError, "exceeding the staging cap"):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertEqual(self.store.client.calls, [])

    def test_registration_failure_is_fatal_in_standalone(self):
        self.store.client.register_ok = False
        with self.assertRaisesRegex(RuntimeError, "register_memory returned false"):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertEqual(self.store.client.calls, [])
        self.assertIsNone(self.store._wholepage_staging)

    def test_malformed_get_response_does_not_apply_partial_chunk(self):
        transfer = self.transfer(count=2)
        self.store._batch_io_v2([transfer], True)
        self.pool.restored.clear()
        self.store.client.malformed = "get"
        self.assertEqual(
            self.store._batch_io_v2([transfer], False), {"mamba": [False, False]}
        )
        self.assertEqual(self.pool.restored, [])

    def test_malformed_exists_response_does_not_start_put(self):
        self.store.client.malformed = "exists"
        self.assertEqual(
            self.store._batch_io_v2([self.transfer(count=2)], True),
            {"mamba": [False, False]},
        )
        checkpoint_puts = [
            c
            for c in self.store.client.calls
            if c[0] == "put" and "_mamba_wholepage_v2_" in c[1][0]
        ]
        self.assertEqual(checkpoint_puts, [])

    def test_capacity_style_get_fails_closed_before_checkpoint_io(self):
        self.store.client.get_mode = "capacity"
        with self.assertRaisesRegex(
            RuntimeError, "does not demonstrate synchronous exact-size"
        ):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertFalse(self.store._wholepage_backend_contract_verified)
        self.assertFalse(
            any("_mamba_wholepage_v2_" in key for key in self.store.client.objects)
        )

    def test_partial_failure_get_fails_closed_before_checkpoint_io(self):
        self.store.client.get_mode = "partial-failure"
        with self.assertRaisesRegex(
            RuntimeError, "does not demonstrate synchronous exact-size"
        ):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertFalse(self.store._wholepage_backend_contract_verified)

    def test_async_success_get_fails_closed(self):
        self.store.client.get_mode = "async-success"
        with self.assertRaisesRegex(
            RuntimeError, "does not demonstrate synchronous exact-size"
        ):
            self.store._batch_io_v2([self.transfer(count=1)], True)

    def test_nonexportable_allocator_is_rejected_before_registration(self):
        class WrongAllocator(FakeSharedAllocator):
            _standalone_process = False

        self.store._new_wholepage_shared_allocator = WrongAllocator
        with self.assertRaisesRegex(RuntimeError, "exportable shared-memory allocator"):
            self.store._batch_io_v2([self.transfer(count=1)], True)
        self.assertEqual(self.store.client.registrations, [])
        self.assertEqual(self.store.client.calls, [])

    def test_plain_mamba_keeps_component_keys_and_zero_copy_io(self):
        plain = self.zero_pool

        def unexpected_schema(_):
            raise AssertionError("Plain Mamba must not request a whole-page schema")

        plain._iter_page_tensors = unexpected_schema
        plain.conv_buffer = []
        plain.temporal_state_elem_size = 1
        self.store.registered_pools["mamba"] = plain
        self.assertEqual(
            self.store._batch_io_v2([self.transfer(count=2)], True),
            {"mamba": [True, True]},
        )
        self.assertEqual(plain.meta_calls, 1)
        self.assertIn("model-tag_prefix0_tp0_temporal", self.store.client.objects)
        self.assertIsNone(self.store._wholepage_staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args, unittest_args = parser.parse_known_args()
    AdapterTests.store_type = load_store(args.source)
    unittest.main(argv=[__file__, *unittest_args])
