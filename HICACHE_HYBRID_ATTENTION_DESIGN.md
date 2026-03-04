# HiCache Support for Hybrid Attention Models (GDN/Mamba)

## Overview

This document describes the implementation of HiCache support for hybrid attention models like Qwen3.5-0.8B, which use both full attention and linear attention (GDN/Mamba) layers.

## Problem Statement

### The Issue

When HiCache loads a prefix match for hybrid attention models:
- **Full attention KV cache** is loaded from storage ✅
- **Linear attention (GDN/Mamba) states** were NOT loaded ❌

This caused **incorrect inference** because linear attention layers computed with zero-initialized states instead of the correct accumulated states from the prefix.

### Why This Matters

Hybrid attention models like Qwen3.5 use:
1. **Full attention layers** - Traditional KV cache, grows with sequence length
2. **Linear attention (GDN/Mamba) layers** - Fixed-size cumulative states per request

Both types need to be properly cached for correct prefix caching behavior.

## Understanding GDN/Mamba States

### State Structure

```python
# From MambaPool in memory_pool.py

# conv_state: Sliding window of recent inputs (convolution buffer)
# Shape: [num_mamba_layers, req_slot_id, conv_dim, kernel_size-1]
# Example: [24, 1025, 1536, 3] for Qwen3.5-0.8B
conv_state = torch.zeros((num_mamba_layers, size + 1) + conv_state_shape)

# temporal_state: Compressed SSM state (all past information)
# Shape: [num_mamba_layers, req_slot_id, num_heads, head_k_dim, head_v_dim]
# Example: [24, 1025, 8, 64, 64] for Qwen3.5-0.8B
temporal_state = torch.zeros((num_mamba_layers, size + 1) + temporal_state_shape)
```

### Key Characteristics

| Property | KV Cache (Full Attention) | Mamba State (Linear Attention) |
|----------|---------------------------|-------------------------------|
| **Indexing** | Per-token | Per-request |
| **Dependency** | Token-local | Cumulative (all previous tokens) |
| **Size** | Grows with seq_len | Fixed per request |
| **Sharing** | Shareable across requests with same prefix | Shareable across requests with same prefix |
| **HiCache Storage** | Per-page in host memory | Per-node in host memory |

### Memory Footprint

For Qwen3.5-0.8B with typical GDN configuration:
- `num_mamba_layers`: ~24 (layers that use linear attention)
- `conv_state_shape`: (1536, 3)
- `temporal_state_shape`: (8, 64, 64)

Per request Mamba state size (FP16):
```
conv_state: 24 * 1536 * 3 * 2 bytes ≈ 221 KB
temporal_state: 24 * 8 * 64 * 64 * 2 bytes ≈ 1.5 MB
Total per request: ≈ 1.7 MB
```

For 10,000 cached nodes: ~17 GB host memory for Mamba states.

## Solution Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     HiCache for Hybrid Models                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Radix Tree Node (represents a prefix):                        │
│  ├── KV Cache (full attention layers)                          │
│  │   ├── Stored per-token in MHATokenToKVPoolHost              │
│  │   ├── Backed up to host memory via write_backup()           │
│  │   └── Loaded from host via load_back()                      │
│  │                                                              │
│  └── Mamba States (linear attention layers)                    │
│      ├── Stored per-node (cumulative state at prefix end)      │
│      ├── Backed up via MambaPoolHost.backup_from_device()      │
│      └── Loaded via MambaPoolHost.load_to_device()             │
│                                                                 │
│  Host Memory Layout:                                           │
│  ├── MHATokenToKVPoolHost: [page, layer, head, dim]            │
│  └── MambaPoolHost: [node, layer, state_type, ...]             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

#### 1. MambaPoolHost (New)

**File**: `python/sglang/srt/mem_cache/memory_pool_host.py`

```python
class MambaPoolHost:
    """Host memory buffer for Mamba (GDN) states."""

    def __init__(
        self,
        num_mamba_layers: int,
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        max_nodes: int = 10000,
    ):
        # Pre-allocate buffers: [max_nodes, num_layers] + state_shape
        self.conv_buffer = torch.zeros(...)
        self.temporal_buffer = torch.zeros(...)

    def backup_from_device(self, mamba_pool, node_id, req_mamba_index, mamba_layer_ids):
        """Copy Mamba states from GPU to CPU for a radix tree node."""

    def load_to_device(self, mamba_pool, node_id, req_mamba_index, mamba_layer_ids):
        """Copy Mamba states from CPU to GPU for a radix tree node."""
```

#### 2. HiRadixCache Updates

**File**: `python/sglang/srt/mem_cache/hiradix_cache.py`

```python
class HiRadixCache(RadixCache):
    def __init__(self, ...):
        # ... existing KV cache initialization ...

        # For hybrid models, also create MambaPoolHost
        if isinstance(self.kv_cache, HybridLinearKVPool):
            self.mamba_pool_host = MambaPoolHost(...)
            self.mamba_layer_ids = sorted(mamba_map.keys())

    def write_backup(self, node, req_mamba_index=None):
        """Backup KV cache AND Mamba states to host."""
        # ... KV cache backup ...
        if self.cache_controller.is_hybrid_gdn and req_mamba_index is not None:
            self.cache_controller.backup_mamba_states(...)

    def write_backup_storage(self, node):
        """Backup KV cache AND Mamba states to storage."""
        # ... KV cache storage ...
        if self.cache_controller.is_hybrid_gdn:
            self.cache_controller.store_mamba_state_to_backend(...)

    def load_back(self, node, req_mamba_index=None):
        """Load KV cache AND Mamba states from host."""
        # ... KV cache loading ...
        if self.cache_controller.is_hybrid_gdn and req_mamba_index is not None:
            self.cache_controller.load_mamba_states(...)

    def load_mamba_states_for_node(self, node, req_mamba_index):
        """Load Mamba states after request is scheduled."""
```

#### 3. HiCacheController Updates

**File**: `python/sglang/srt/managers/cache_controller.py`

```python
class HiCacheController:
    def __init__(self, ..., mamba_pool_host=None, mamba_layer_ids=None):
        self.mamba_pool_host = mamba_pool_host
        self.mamba_layer_ids = mamba_layer_ids
        self.is_hybrid_gdn = isinstance(self.mem_pool_device, HybridLinearKVPool)

    def backup_mamba_states(self, node_id, req_mamba_index, mamba_pool):
        """Backup Mamba states to host memory."""

    def load_mamba_states(self, node_id, req_mamba_index, mamba_pool):
        """Load Mamba states from host memory."""
```

#### 4. Scheduler Integration

**File**: `python/sglang/srt/managers/schedule_batch.py`

```python
class Req:
    def init_next_round_input(self, tree_cache, req_to_token_pool):
        # ... prefix matching ...

        # Load Mamba states if prefix match and Mamba index allocated
        if self.host_hit_length > 0 and req_to_token_pool is not None:
            req_mamba_index = req_to_token_pool.rid_to_mamba_index_mapping.get(self.rid)
            if req_mamba_index is not None:
                tree_cache.load_mamba_states_for_node(self.last_host_node, req_mamba_index)
```

#### 5. Storage Backend Support

**Files**: `python/sglang/srt/mem_cache/hicache_storage.py`, `python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py`

```python
# Base storage interface (hicache_storage.py)
class HiCacheStorage(ABC):
    def store_mamba_state(self, node_hash, conv_state, temporal_state) -> bool:
        """Store Mamba states for a radix tree node."""
        raise NotImplementedError

    def load_mamba_state(self, node_hash, conv_state, temporal_state) -> bool:
        """Load Mamba states for a radix tree node."""
        raise NotImplementedError

    def mamba_state_exists(self, node_hash) -> bool:
        """Check if Mamba state exists for a node."""
        raise NotImplementedError

# File backend implementation
class HiCacheFile(HiCacheStorage):
    def store_mamba_state(self, node_hash, conv_state, temporal_state):
        # Store as separate files: {hash}_conv.bin, {hash}_temporal.bin
        ...

    def load_mamba_state(self, node_hash, conv_state, temporal_state):
        # Load from separate files
        ...

# NIXL backend implementation
class HiCacheNixl(HiCacheStorage):
    def store_mamba_state(self, node_hash, conv_state, temporal_state):
        # Combine and store as single object: {hash}_mamba
        ...

    def load_mamba_state(self, node_hash, conv_state, temporal_state):
        # Load and split combined tensor
        ...
```

## Implementation Details

### Flow Diagram

```
Request Arrival
      │
      ▼
┌─────────────────┐
│ Prefix Match    │─── No match ──► New computation
└────────┬────────┘
         │ Match found
         ▼
┌─────────────────┐
│ Load KV Cache   │ (from host memory)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Allocate Mamba  │ (get req_mamba_index)
│ Index           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Mamba      │ (from host memory)
│ States          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Continue        │ (with restored states)
│ Inference       │
└─────────────────┘
```

### Timing of Operations

| Operation | When | Notes |
|-----------|------|-------|
| KV cache backup | When node is written to host | `write_backup()` |
| Mamba state backup | When node is written to host | Same call, requires `req_mamba_index` |
| KV cache load | When loading evicted node | `load_back()` |
| Mamba state load | After Mamba index allocated | `load_mamba_states_for_node()` |

### Why Separate Mamba Loading?

The Mamba index is allocated **after** prefix matching in the scheduling flow:

1. `init_load_back()` - Load KV cache (Mamba index not yet allocated)
2. Request scheduled, Mamba index allocated
3. `init_next_round_input()` - Load Mamba states (now we have the index)

This is why we have `load_mamba_states_for_node()` as a separate method.

## Files Changed

| File | Changes |
|------|---------|
| `memory_pool_host.py` | +205 lines: New `MambaPoolHost` class |
| `hiradix_cache.py` | +162 lines: Mamba state integration |
| `cache_controller.py` | +77 lines: Mamba state methods |
| `schedule_batch.py` | +18 lines: Load Mamba states after scheduling |
| `schedule_policy.py` | +13 lines: Pass Mamba index to load |
| `scheduler.py` | +4 lines: Pass req_to_token_pool |
| `model_runner.py` | +6 lines: Remove forced radix cache disabling |

## Testing

### Unit Tests

```python
def test_mamba_pool_host():
    host = MambaPoolHost(num_mamba_layers=24, ...)
    # Test alloc, backup, load, clear
```

### Integration Tests

```bash
# Run Qwen3.5-0.8B with HiCache
python -m sglang.launch_server \
    --model Qwen/Qwen3.5-0.8B \
    --enable-hierarchical-cache \
    --hicache-storage-backend nixl \
    --hicache-storage-path /tmp/hicache

# Verify prefix caching works correctly
python -m sglang.launch_eval \
    --backend http \
    --dataset-name boolq \
    --request-rate 2
```

### Correctness Verification

Compare outputs with and without HiCache:
```python
# Without HiCache
output1 = generate(model, prompt, disable_hicache=True)

# With HiCache (after warming up cache)
output2 = generate(model, prompt, disable_hicache=False)

assert torch.allclose(output1, output2, atol=1e-5)
```

## Performance Considerations

### Memory Usage

```
Total HiCache memory = KV cache host buffer + Mamba state host buffer

For Qwen3.5-0.8B with 10,000 nodes:
- KV cache: ~X GB (depends on page_size, num_layers)
- Mamba states: ~17 GB (1.7 MB * 10,000)
```

### CPU-GPU Transfer Overhead

Mamba state transfer per node:
```
Transfer size: ~1.7 MB (FP16)
Transfer time: ~1-2 ms (PCIe Gen4)
Impact: Negligible compared to KV cache transfer
```

### Scalability

- `max_nodes` parameter controls MambaPoolHost size
- Default: 10,000 nodes (~17 GB for Qwen3.5-0.8B)
- Can be tuned based on available host memory

## Future Enhancements

1. **Storage Backend Integration**: Persist Mamba states to disk/object storage
2. **Compression**: Compress Mamba states to reduce memory footprint
3. **Selective Caching**: Only cache Mamba states for frequently-hit prefixes
4. **Recompute Fallback**: Recompute Mamba states from tokens if memory exhausted

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Qwen3.5 Documentation](https://qwenlm.github.io/)
- [SGLang HiCache Design](https://github.com/sgl-project/sglang)
