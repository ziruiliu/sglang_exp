# HiCache Support for Hybrid Attention Models (GDN/Mamba)

## Problem Statement

When HiCache loads a prefix match for hybrid attention models like Qwen3.5:
- **Full attention KV cache** is loaded from storage ✅
- **Linear attention (GDN/Mamba) states** are NOT loaded ❌

This causes **incorrect inference** because linear attention layers compute with zero-initialized states instead of the correct accumulated states from the prefix.

## Understanding GDN/Mamba States

### State Structure (from `MambaPool`)

```python
# conv_state: Sliding window of recent inputs
# Shape: [num_mamba_layers, req_slot_id, conv_dim, kernel_size-1]
conv_state = torch.zeros((num_mamba_layers, size + 1) + conv_state_shape)

# temporal_state: Compressed representation of all past tokens (SSM state)
# Shape: [num_mamba_layers, req_slot_id, num_heads, head_k_dim, head_v_dim]
temporal_state = torch.zeros((num_mamba_layers, size + 1) + temporal_state_shape)
```

### Key Characteristics

| Property | KV Cache (Full Attention) | Mamba State (Linear Attention) |
|----------|---------------------------|-------------------------------|
| **Indexing** | Per-token | Per-request |
| **Dependency** | Token-local | Cumulative (all previous tokens) |
| **Size** | Grows with seq_len | Fixed per request |
| **Sharing** | Shareable across requests with same prefix | Shareable across requests with same prefix |

## Solution Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     HiCache for Hybrid Models                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Radix Tree Node (represents a prefix):                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TreeNode                                                  │  │
│  │ ├── key: RadixKey (token_ids for this prefix)            │  │
│  │ ├── value: device_indices (KV cache on GPU)              │  │
│  │ ├── host_value: host_indices (KV cache on CPU)           │  │
│  │ ├── hash_value: List[str] (storage keys)                 │  │
│  │ └── mamba_state: Dict[layer_id -> (conv, temporal)]      │  │  NEW!
│  │       └── Stores Mamba states at the END of this prefix  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Storage Backend:                                               │
│  ├── KV cache pages (existing, per-page)                       │
│  └── Mamba states (NEW, per-node)                              │
│      └── Key: node hash, Value: (conv_state, temporal_state)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Components

#### 1. Extend `TreeNode` to Store Mamba States

**File: `python/sglang/srt/mem_cache/radix_cache.py`**

```python
class TreeNode:
    def __init__(self, id: Optional[int] = None):
        # ... existing fields ...

        # NEW: Store Mamba states for hybrid attention models
        # Dict: layer_id (local index) -> (conv_state, temporal_state)
        # conv_state: [conv_dim, kernel_size-1]
        # temporal_state: [num_heads, head_k_dim, head_v_dim]
        self.mamba_state: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None

        # Host-side Mamba state backup
        self.mamba_state_host: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
```

#### 2. Create `MambaPoolHost` for Host Memory Backup

**File: `python/sglang/srt/mem_cache/memory_pool_host.py`**

```python
class MambaPoolHost:
    """
    Host memory buffer for Mamba (GDN) states.

    Unlike KV cache which is per-token, Mamba states are per-request
    and represent the cumulative state at a given prefix position.
    """

    def __init__(
        self,
        mamba_pool: MambaPool,
        num_mamba_layers: int,
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        max_nodes: int = 10000,  # Max cached nodes
        device: str = "cpu",
    ):
        self.num_mamba_layers = num_mamba_layers
        self.conv_state_shape = conv_state_shape
        self.temporal_state_shape = temporal_state_shape
        self.conv_dtype = conv_dtype
        self.ssm_dtype = ssm_dtype
        self.device = device

        # Pool of pre-allocated state buffers (for efficiency)
        # Each entry: (conv_state, temporal_state) for all layers
        self._conv_pool = torch.zeros(
            (max_nodes, num_mamba_layers) + conv_state_shape,
            dtype=conv_dtype, device=device
        )
        self._temporal_pool = torch.zeros(
            (max_nodes, num_mamba_layers) + temporal_state_shape,
            dtype=ssm_dtype, device=device
        )
        self._free_slots = list(range(max_nodes))

        # Mapping: node_id -> pool_slot
        self._node_to_slot: Dict[int, int] = {}

    def alloc(self, node_id: int) -> int:
        """Allocate a slot for storing Mamba states."""
        if not self._free_slots:
            return -1
        slot = self._free_slots.pop(0)
        self._node_to_slot[node_id] = slot
        return slot

    def free(self, node_id: int):
        """Free the slot for a node."""
        if node_id in self._node_to_slot:
            slot = self._node_to_slot.pop(node_id)
            self._free_slots.append(slot)
            # Zero out the buffers
            self._conv_pool[slot] = 0
            self._temporal_pool[slot] = 0

    def get_buffers(self, node_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the conv and temporal state buffers for a node."""
        slot = self._node_to_slot[node_id]
        return (
            self._conv_pool[slot],
            self._temporal_pool[slot]
        )

    def backup_from_device(
        self,
        mamba_pool: MambaPool,
        node_id: int,
        req_mamba_index: int,
        mamba_layer_ids: List[int]
    ):
        """Copy Mamba states from device (GPU) to host (CPU)."""
        slot = self._node_to_slot.get(node_id)
        if slot is None:
            slot = self.alloc(node_id)
            if slot < 0:
                return False

        # Get device Mamba cache: [num_cache_tensors, num_layers, ...]
        mamba_cache = mamba_pool.mamba_cache
        # mamba_cache[0] = conv_state, mamba_cache[1] = temporal_state

        for local_idx, global_layer_id in enumerate(mamba_layer_ids):
            # Copy from device to host
            self._conv_pool[slot, local_idx] = mamba_cache[0][global_layer_id, req_mamba_index]
            self._temporal_pool[slot, local_idx] = mamba_cache[1][global_layer_id, req_mamba_index]

        return True

    def load_to_device(
        self,
        mamba_pool: MambaPool,
        node_id: int,
        req_mamba_index: int,
        mamba_layer_ids: List[int]
    ):
        """Copy Mamba states from host (CPU) to device (GPU)."""
        slot = self._node_to_slot.get(node_id)
        if slot is None:
            return False

        mamba_cache = mamba_pool.mamba_cache

        for local_idx, global_layer_id in enumerate(mamba_layer_ids):
            # Copy from host to device
            mamba_cache[0][global_layer_id, req_mamba_index] = self._conv_pool[slot, local_idx]
            mamba_cache[1][global_layer_id, req_mamba_index] = self._temporal_pool[slot, local_idx]

        return True
```

#### 3. Update `HiRadixCache` to Handle Mamba States

**File: `python/sglang/srt/mem_cache/hiradix_cache.py`**

```python
class HiRadixCache(RadixCache):
    def __init__(self, ...):
        # ... existing initialization ...

        # For hybrid attention models, also create Mamba state host buffer
        self.is_hybrid_gdn = isinstance(self.kv_cache, HybridLinearKVPool)
        self.mamba_pool_host = None

        if self.is_hybrid_gdn:
            from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

            if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
                mamba_pool = self.req_to_token_pool.mamba_pool
                config = self.req_to_token_pool.mamba_map  # layer_id -> local_idx

                self.mamba_pool_host = MambaPoolHost(
                    mamba_pool=mamba_pool,
                    num_mamba_layers=len(config),
                    conv_state_shape=mamba_pool.mamba_cache[0].shape[2:],
                    temporal_state_shape=mamba_pool.mamba_cache[1].shape[2:],
                    conv_dtype=mamba_pool.mamba_cache[0].dtype,
                    ssm_dtype=mamba_pool.mamba_cache[1].dtype,
                )
                self.mamba_layer_ids = sorted(config.keys())

    def write_backup(self, node: TreeNode, write_back=False):
        """Backup KV cache AND Mamba states to host memory."""
        # Existing KV cache backup
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        # ... rest of existing code ...

        # NEW: Backup Mamba states for hybrid models
        if self.is_hybrid_gdn and node.value is not None:
            # Get the request's Mamba index from the last token
            # This requires tracking which request owns this node
            # For now, we'll backup when the node is being written
            pass  # Implemented in cache_controller

    def load_back(self, node: TreeNode, ...) -> Optional[torch.Tensor]:
        """Load KV cache AND Mamba states from host to device."""
        # Existing KV cache loading
        device_indices = self.cache_controller.load(...)

        # NEW: After loading KV cache, also restore Mamba states
        if self.is_hybrid_gdn and self.mamba_pool_host is not None:
            # Mamba states will be restored when the request is scheduled
            # See scheduler integration below
            pass

        return device_indices
```

#### 4. Update `HiCacheController` for Mamba State Operations

**File: `python/sglang/srt/managers/cache_controller.py`**

```python
class HiCacheController:
    def __init__(self, ...):
        # ... existing initialization ...

        self.is_hybrid_gdn = isinstance(self.mem_pool_device, HybridLinearKVPool)
        self.mamba_pool_host = None
        self.mamba_layer_ids = []

        if self.is_hybrid_gdn:
            # Initialize MambaPoolHost
            pass

    def backup_mamba_states(
        self,
        node_id: int,
        req_mamba_index: int,
        mamba_pool: MambaPool
    ):
        """Backup Mamba states to host memory."""
        if self.mamba_pool_host is None:
            return False

        return self.mamba_pool_host.backup_from_device(
            mamba_pool, node_id, req_mamba_index, self.mamba_layer_ids
        )

    def load_mamba_states(
        self,
        node_id: int,
        req_mamba_index: int,
        mamba_pool: MambaPool
    ):
        """Load Mamba states from host memory."""
        if self.mamba_pool_host is None:
            return False

        return self.mamba_pool_host.load_to_device(
            mamba_pool, node_id, req_mamba_index, self.mamba_layer_ids
        )
```

#### 5. Storage Backend Integration

**File: `python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py`**

The storage backend needs to handle Mamba states alongside KV cache:

```python
class HiCacheNixl(HiCacheStorage):
    def store_node(self, node_id: int, hash_value: str, node_data: Dict):
        """Store a complete node including KV cache and Mamba states."""
        # KV cache pages (existing)
        # ...

        # Mamba states (NEW)
        if 'mamba_state' in node_data:
            # Flatten Mamba states for storage
            mamba_tensors = []
            for layer_id, (conv, temporal) in node_data['mamba_state'].items():
                mamba_tensors.append(conv.flatten())
                mamba_tensors.append(temporal.flatten())

            # Store with a derived key
            mamba_key = f"{hash_value}_mamba"
            self.batch_set([mamba_key], mamba_tensors)

    def load_node(self, hash_value: str) -> Dict:
        """Load a complete node including KV cache and Mamba states."""
        result = {}

        # KV cache pages (existing)
        # ...

        # Mamba states (NEW)
        mamba_key = f"{hash_value}_mamba"
        if self.exists(mamba_key):
            # Load and reconstruct Mamba states
            mamba_data = self.get(mamba_key)
            result['mamba_state'] = self._reconstruct_mamba_states(mamba_data)

        return result
```

#### 6. Scheduler Integration

**File: `python/sglang/srt/managers/scheduler.py`**

When a prefix match occurs, restore Mamba states to the new request:

```python
class Scheduler:
    def _handle_prefix_match(self, req: Req, match_result: MatchResult):
        """Handle a prefix cache hit."""
        # Existing KV cache handling
        # ...

        # NEW: Restore Mamba states for hybrid models
        if self.enable_hierarchical_cache and self.model_runner.is_hybrid_gdn:
            last_node = match_result.last_device_node

            # Get the request's Mamba index
            mamba_index = self.req_to_token_pool.get_mamba_index(req.rid)

            # Load Mamba states from the matched node
            if hasattr(last_node, 'mamba_state_host') and last_node.mamba_state_host is not None:
                self.tree_cache.cache_controller.load_mamba_states(
                    node_id=last_node.id,
                    req_mamba_index=mamba_index,
                    mamba_pool=self.req_to_token_pool.mamba_pool
                )
```

## Memory Considerations

### Mamba State Size per Request

For Qwen3.5-0.8B with typical GDN configuration:
- `num_mamba_layers`: ~24 (layers that use linear attention)
- `conv_state_shape`: (conv_dim, kernel_size-1) ≈ (1536, 3)
- `temporal_state_shape`: (num_heads, head_k_dim, head_v_dim) ≈ (8, 64, 64)

Per request Mamba state size:
```
conv_state: 24 * 1536 * 4 * 2 bytes (FP16) ≈ 580 KB
temporal_state: 24 * 8 * 64 * 64 * 2 bytes (FP16) ≈ 1.5 MB
Total per request: ≈ 2 MB
```

For 1000 cached nodes: ~2 GB host memory for Mamba states.

## Testing Checklist

1. **Unit tests**: MambaPoolHost alloc/free/backup/load
2. **Integration tests**: HiCache with hybrid model prefix matching
3. **Correctness tests**: Verify outputs match without caching
4. **Performance tests**: Measure overhead of Mamba state caching

## Alternative: Recompute Mamba States

If caching Mamba states is too memory-intensive, an alternative is to **recompute** them from the token sequence:

```python
def restore_mamba_by_recompute(
    model: nn.Module,
    req: Req,
    prefix_tokens: List[int],
    mamba_layers: List[int]
):
    """
    Recompute Mamba states by processing prefix tokens through linear layers.

    This is slower than caching but uses no extra memory.
    """
    # Run prefix tokens through GDN layers to rebuild states
    # This requires modifying the forward pass to support "state-only" mode
    pass
```

This approach trades compute for memory and may be preferable for very long prefixes or memory-constrained environments.
