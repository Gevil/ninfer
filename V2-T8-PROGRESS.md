# V2-T8 state-image re-target — progress (2026-09-09)

## Concrete increments (done)
1. **Repack spike — PROVEN (compiles, exit 0).** `/tmp/v2-t8-spike-repack.cpp`
   (container copy `/tmp/spike3.cpp`) compiles against the real `state_image.h` API.
   Pattern: `host.allocate()` -> `pool.copy_to_host(0, host.writable_view(*handle), stream)`
   -> read sub-regions via `StateImageHostLayout` (`linear_conv`, `linear_recurrent`,
   `continuation_hidden`, `dflash_local_k/v`, public `.offset`/`.bytes` fields) ->
   `memcpy` into the substrate's flat wire buffer (`raw + header.offset[N]`).
2. **`kv_ram_cache.h` — IN WORKTREE (git-verified).** `RamCaptureSource` gains
   `const StateImageDevicePool* state_image` + `HostStatePool* state_host` (null = KV-only
   capture) and `#include <ninfer/targets/qwen3_6/state_image.h>`.

## `.cpp` capture path — NOT yet wired (design finding, timeboxed)
The substrate's state-half pack is at `KVRamCache::capture` lines 918-952
(gdn->pack_slot_to_host, tail_hidden/rewrite_checkpoint_hidden cudaMemcpyAsync,
dflash_local/checkpoint copy_lane_to_host), all async-on-`source.stream`, tracked by
`record_copies` + the `copies_start` event.

**Key design constraint (why the .cpp edit is not a mechanical swap):**
`StateImageDevicePool::copy_to_host(source, HostStateImageView, stream)` is itself an
ASYNC device->pinned-host copy launched on the stream (returns immediately; the pinned
`HostStatePool` buffer is not readable until the copy completes on the stream). The
repack's CPU-side sub-region memcpys (HostStatePool buffer -> substrate flat
`raw + offset[N]`) therefore CANNOT follow inline in the capture path the way the spike
shows them. Two resolutions to choose (needs a decision, not a blind edit):
  (a) sync the copy before the memcpys (cudaStreamSynchronize / event-sync) — serializes
      the capture, changes the async-copy contract of KVRamCache::capture; or
  (b) capture the state image into the substrate flat buffer DIRECTLY (no intermediate
      HostStatePool round-trip) — requires the copy_to_host-equivalent to target an
      arbitrary host span, which the pinned-pool API does not expose.
Neither is a drop-in; both need design sign-off before touching the .cpp.

## Verification status
- Spike compile: PASS (exit 0, container epic_neumann, g++ -std=c++20, CUDA).
- Full per-TU compile of kv_ram_cache.cpp: BLOCKED — container epic_neumann baseline
  snapshot is stale across /src/include + /src/src (missing KvCacheStorage::Nvfp4Group16,
  ToolCallParseDiagnostics, PagedKVAllocation, PrefixReusePath::FullReset). Version-skew
  too deep for path-chasing; container predates the substrate target commit. Fallback:
  spike compile + manual diff review.

## Key API facts (verified from state_image.h)
- `LayoutRegion` = public fields `offset`/`bytes`/`alignment` (NOT methods).
- `HostStateImageConstView` = `const std::byte* data` + `const StateImageHostLayout* layout`.
- `HostStateImageView` = `std::byte* data` + layout (writable destination for copy_to_host).
- `HostStatePool` = fixed-capacity PINNED host storage (PinnedHostBuffer backing);
  `allocate()` -> HostStateSlotHandle; `writable_view(handle)` -> HostStateImageView.
- `StateImageDevicePool::copy_to_host(source, dest, stream=nullptr)` = async-on-stream
  device->pinned-host copy (the re-target's key async constraint).
- `StateImageHostLayout` regions: `linear_conv`, `linear_recurrent`, `continuation_hidden`,
  `dflash_local_k`/`dflash_local_v` (optional), `image_bytes`.

## CORRECTED design (2026-09-09, after reading HostStateImageView in full)
`HostStateImageView` is just `{ std::byte* data; const StateImageHostLayout* layout; }` —
constructible from RAW POINTERS. So the CPU-repack + pinned-pool round-trip is UNNECESSARY:
build a `HostStateImageView` whose `data` points straight at the state half inside the
already-allocated capture block (`raw + header.offset[<state-base>]`) and whose `layout`
is the source's `state_image->host_layout()`, then call
`source.state_image->copy_to_host(slot, view, source.stream)` ONCE. The async device->host
copy lands directly in the substrate flat buffer on `source.stream` — identical stream-async
pattern to the existing pack_slot_to_host/cudaMemcpyAsync/copy_lane_to_host calls. No
HostStatePool buffer, no CPU memcpys, no cudaStreamSynchronize. (This voids resolutions
(a)/(b) above.)

Remaining real work (the actual re-architecture, scope to plan before editing):
- The state half inside the capture flat buffer must be laid out per `StateImageHostLayout`
  (linear_conv @ layout.linear_conv.offset, linear_recurrent @ .linear_recurrent.offset,
  continuation_hidden @ .continuation_hidden.offset, dflash_local_k/v @ .dflash_local_*),
  NOT the current gdn@4/6 + hidden@8/9 + dflash@10/11 offsets. So `make_capture_header` +
  `finalize_capture_layout` (which emit `header.offset[N]`/`header.length[N]`/entry_bytes)
  must source the state-half geometry from `state_image->host_layout()`, and the RESTORE
  path (unpack_device ~line 1016 + the "missing GDN/DFlash/hidden" geometry checks
  ~1035-1054) must read back via the same `StateImageHostLayout` regions (or
  copy_from_host with a matching const view).
- `RamCaptureSource.state_image` slot selection: which absolute slot to capture
  (current vs checkpoint vs both) must map onto the single `copy_to_host(source_slot)`
  call; the rewrite/checkpoint second copy needs its own slot + view.

## Increment: state-image bridge APPLIED + verified (2026-09-09, commit 67c0818d)

### What landed
- `RamCaptureSource`/`RamRestoreTarget` gained `state_image` (StateImageDevicePool) +
  `state_slot`/`state_checkpoint_slot` (absolute slots); `.h` verified host-side.
- Capture/restore state half re-targeted: `has_state_image` record flag (kRamVersion 2->3),
  sections 4/5 = complete StateImage payloads laid out per `StateImageHostLayout`,
  captured/restored via `StateImageDevicePool::copy_to_host`/`copy_from_host` landing
  DIRECTLY in the capture block (`HostStateImageView{raw + offset[4], &pool.host_layout()}`
  — `HostStateImageView` is just {data, layout}, validated against the layout, NOT against
  HostStatePool ownership — so the spike's 2-hop HostStatePool repack is unnecessary).
- Legacy per-type path retained verbatim (dormant when `state_image` is set).

### Verified compile state (buildstage container t33bg2-buildstage, g++ -std=c++20 -fsyntax-only,
include roots: src/ include/ src/targets/qwen3_6/export/ + /usr/local/cuda/include,
mount with :z for SELinux, --user for rootless uid mapping)
- HEAD (pre-increment): 96 errors. Post-increment: 96 errors. **State-image path: 0 errors.**
- CORRECTION: the [X] "Per-TU g++ -fsyntax-only" todo item was wrong — the WIP file never
  compiled clean; 96 pre-existing substrate-API errors exist at HEAD (categorized below).
- All 96 are pre-existing legacy-path references to substrate-only APIs absent from the
  baseline:
  1. WIP field-name drift (WIP .h vs .cpp): `source.text/text_pool/backend/backend_pool`
     and `target.text/...` vs the .h's `text_cache/text_cache_pool/backend_cache/
     backend_cache_pool` (+ `backend_pages`).
  2. PagedKVCache (qwen3_6): no `pack_residual_slot_to_host`/`pack_paged_kv_allocation_to_host`/
     `pack_paged_kv_ring_to_host`/`unpack_residual_slot_from_host`/`residual_enabled`/
     `residual_slot_host_bytes`/`ring_valid_slot_host_bytes` (baseline equivalent =
     `DeviceKVPagePool::copy_to_host/copy_from_host` + HostKVAllocation views, or the
     HostKVExtentStore demotion path per the hostkv spike).
  3. `LinearAttentionStatePool`: no host-image API at all in the baseline (conv_host_image_
     bytes/pack_slot_to_host/unpack_slot_from_host) — the state-image bridge is the replacement.
  4. `CyclicKVCache`: no `lane_host_bytes`/`copy_lane_to_host`/`copy_lane_from_host` in the
     baseline (slot_view + plane memcpys, or the state-image bridge).
  5. `PagedKVPool` type + `plane_count/plane/plane_order` (substrate name; baseline =
     DeviceKVPagePool, which DOES have plane_count/plane).
  6. `unsigned char*` -> `std::byte*` conversions in the paged-KV host calls.
- Baseline API facts pinned this session: `StateImageSpec.dflash_local` is
  `std::optional<DFlashLocalStateSpec>`; `HostStatePool` owns storage only (allocate/release/
  writable_view/view — NO copy methods; device copy goes through StateImageDevicePool);
  `copy_to_host(slot, HostStateImageView, stream) const` / `copy_from_host(view, slot, stream)`.

### Next work unit (the remaining paged-KV half) — AWAITING GO-AHEAD
Bridge the substrate's paged-KV host-image half onto the baseline's HostKVExtentStore /
DeviceKVPagePool host-copy APIs (the "paged-KV replica path" of the in-progress bridge todo;
mapping in ADOPTION-V2.md section 6.6, branch v2/adoption; proven by /tmp/v2-t8-spike-hostkv.cpp).
This is the bulk of the 96 errors and is a large rewrite (capture/restore/layout/header +
field renames + byte casts) — the scope the project previously dropped as weeks-scale and
re-scoped as the policy bridge. Not started this session.
