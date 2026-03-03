import abc
import logging
import threading
from enum import IntEnum
from functools import wraps
from typing import Optional

import psutil
import torch

from typing import Dict, List, Tuple

from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.utils import is_npu, is_xpu

_is_npu = is_npu()
_is_xpu = is_xpu()
if not (_is_npu or _is_xpu):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
    )

logger = logging.getLogger(__name__)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the host memory pool size to the page size
        self.size = self.size - (self.size % self.page_size)
        self.page_num = self.size // self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
        )
        self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        return torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer(
                    src_k=self.k_buffer[layer_id],
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer[layer_id],
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    item_size=self.token_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_pf_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k_layers=self.k_data_ptrs,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v_layers=self.v_data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_lf_pf(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
        )
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = (
            self.kv_lora_rank + self.qk_rope_head_dim
        ) * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        return torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.kv_buffer[layer_id],
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    item_size=self.token_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.kv_buffer,
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=device_pool.data_ptrs,
                    dst_layers=self.data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=device_pool.data_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * (self.kv_lora_rank + self.qk_rope_head_dim)
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * (self.kv_lora_rank + self.qk_rope_head_dim)
                        * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = (
                self.dtype.itemsize
                * self.page_size
                * (self.kv_lora_rank + self.qk_rope_head_dim)
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * (self.kv_lora_rank + self.qk_rope_head_dim)
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * (self.kv_lora_rank + self.qk_rope_head_dim)
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MambaPoolHost:
    """
    Host memory buffer for Mamba (GDN) states.

    Unlike KV cache which is per-token, Mamba states are per-request
    and represent the cumulative state at a given prefix position.

    For hybrid attention models (e.g., Qwen3.5), each radix tree node
    stores the Mamba state at the end of that prefix. When a request
    matches a prefix, the Mamba state is restored to continue from
    where the prefix left off.
    """

    def __init__(
        self,
        num_mamba_layers: int,
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        max_nodes: int = 10000,
        device: str = "cpu",
        pin_memory: bool = True,
    ):
        """
        Initialize Mamba state host buffer.

        Args:
            num_mamba_layers: Number of linear attention (GDN/Mamba) layers
            conv_state_shape: Shape of conv state per layer (conv_dim, kernel_size-1)
            temporal_state_shape: Shape of temporal state per layer (heads, k_dim, v_dim)
            conv_dtype: Data type for conv state
            ssm_dtype: Data type for temporal (SSM) state
            max_nodes: Maximum number of cached nodes in host memory
            device: Device for host buffer (typically "cpu")
            pin_memory: Whether to use pinned memory for faster GPU transfers
        """
        self.num_mamba_layers = num_mamba_layers
        self.conv_state_shape = conv_state_shape
        self.temporal_state_shape = temporal_state_shape
        self.conv_dtype = conv_dtype
        self.ssm_dtype = ssm_dtype
        self.device = device
        self.pin_memory = pin_memory

        # Calculate sizes
        self.conv_state_size = num_mamba_layers * conv_state_shape[0] * conv_state_shape[1]
        self.temporal_state_size = (
            num_mamba_layers
            * temporal_state_shape[0]
            * temporal_state_shape[1]
            * temporal_state_shape[2]
        )

        # Pre-allocate buffers for all nodes
        # Shape: [max_nodes, num_layers] + state_shape
        conv_buffer_shape = (max_nodes, num_mamba_layers) + conv_state_shape
        temporal_buffer_shape = (max_nodes, num_mamba_layers) + temporal_state_shape

        self.conv_buffer = torch.zeros(
            conv_buffer_shape, dtype=conv_dtype, device=device, pin_memory=pin_memory
        )
        self.temporal_buffer = torch.zeros(
            temporal_buffer_shape, dtype=ssm_dtype, device=device, pin_memory=pin_memory
        )

        # Slot management
        self._free_slots: List[int] = list(range(max_nodes))
        self._node_to_slot: Dict[int, int] = {}
        self.lock = threading.RLock()

        logger.info(
            f"MambaPoolHost initialized with {max_nodes} nodes. "
            f"Conv state: {self.conv_state_size} elements, "
            f"Temporal state: {self.temporal_state_size} elements"
        )

    def _get_slot(self, node_id: int) -> Optional[int]:
        """Get the slot for a node, allocating if necessary."""
        with self.lock:
            if node_id in self._node_to_slot:
                return self._node_to_slot[node_id]

            if not self._free_slots:
                logger.warning("MambaPoolHost out of slots, cannot allocate for node")
                return None

            slot = self._free_slots.pop(0)
            self._node_to_slot[node_id] = slot
            return slot

    def _release_slot(self, node_id: int):
        """Release the slot for a node."""
        with self.lock:
            if node_id in self._node_to_slot:
                slot = self._node_to_slot.pop(node_id)
                self._free_slots.append(slot)
                # Zero out the buffers for security/cleanliness
                self.conv_buffer[slot] = 0
                self.temporal_buffer[slot] = 0

    def get_slot(self, node_id: int) -> Optional[int]:
        """Get the slot for a node (or None if not allocated)."""
        return self._node_to_slot.get(node_id)

    def backup_from_device(
        self,
        mamba_pool: MambaPool,
        node_id: int,
        req_mamba_index: int,
        mamba_layer_ids: List[int],
    ) -> bool:
        """
        Copy Mamba states from device (GPU) to host (CPU).

        Args:
            mamba_pool: The device-side MambaPool
            node_id: The radix tree node ID
            req_mamba_index: The request's index in the MambaPool
            mamba_layer_ids: List of global layer IDs that use Mamba (sorted)

        Returns:
            True if backup succeeded, False otherwise
        """
        slot = self._get_slot(node_id)
        if slot is None:
            return False

        # mamba_pool.mamba_cache[0] = conv_state
        # mamba_pool.mamba_cache[1] = temporal_state
        # Shape: [num_layers, req_index + 1, ...]
        mamba_cache = mamba_pool.mamba_cache

        # Copy states for each Mamba layer
        for local_idx, global_layer_id in enumerate(mamba_layer_ids):
            self.conv_buffer[slot, local_idx] = mamba_cache[0][
                global_layer_id, req_mamba_index
            ].clone()
            self.temporal_buffer[slot, local_idx] = mamba_cache[1][
                global_layer_id, req_mamba_index
            ].clone()

        return True

    def load_to_device(
        self,
        mamba_pool: MambaPool,
        node_id: int,
        req_mamba_index: int,
        mamba_layer_ids: List[int],
    ) -> bool:
        """
        Copy Mamba states from host (CPU) to device (GPU).

        Args:
            mamba_pool: The device-side MambaPool
            node_id: The radix tree node ID
            req_mamba_index: The request's index in the MambaPool
            mamba_layer_ids: List of global layer IDs that use Mamba (sorted)

        Returns:
            True if load succeeded, False otherwise
        """
        slot = self._node_to_slot.get(node_id)
        if slot is None:
            return False

        mamba_cache = mamba_pool.mamba_cache

        # Copy states for each Mamba layer
        for local_idx, global_layer_id in enumerate(mamba_layer_ids):
            mamba_cache[0][global_layer_id, req_mamba_index] = self.conv_buffer[
                slot, local_idx
            ]
            mamba_cache[1][global_layer_id, req_mamba_index] = self.temporal_buffer[
                slot, local_idx
            ]

        return True

    def get_state_size_bytes(self) -> int:
        """Get the total size of Mamba state buffers in bytes."""
        return (
            self.conv_buffer.element_size() * self.conv_buffer.numel()
            + self.temporal_buffer.element_size() * self.temporal_buffer.numel()
        )

    def clear(self):
        """Clear all cached states."""
        with self.lock:
            self._node_to_slot.clear()
            self._free_slots = list(range(len(self._free_slots)))
            self.conv_buffer.zero_()
            self.temporal_buffer.zero_()
