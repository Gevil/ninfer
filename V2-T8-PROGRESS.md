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

## Increment: paged-KV bridge APPLIED + verified clean (2026-09-09, this commit)

### Design (settled this session)
- The record's paged-KV image lives in the lane's **shared `HostKVArena`** (the same pool
  the extent store demotes into), NOT in the flat capture block: `KVRamCache` takes
  `HostKVArena&` in its constructor (non-owning; the arena outlives the cache) and each
  `Record` holds one move-only `HostKVAllocation` per non-empty page span
  (`text_host_kv`/`backend_host_kv`). `RetiredCopy` takes those over in `consume()` so the
  image survives until the in-flight copy's `copies_done` event is reaped (the flat block
  had the same lifetime rule; the arena allocation now has it too).
- Capture: `arena_->layout_for(text_cache->page_pool().geometry())` ->
  `allocate(*layout, pages)` -> `writable_view(alloc)` ->
  `text_cache->page_pool().copy_to_host(text_pages, view, stream)` (stream-async, same
  demotion-copy path the extent store uses). Restore is the mirror:
  `copy_from_host(arena_->view(record.text_host_kv), target.text_pages, stream)`.
- `RamCaptureSource.text_pages/backend_pages` and `RamRestoreTarget.text_pages/backend_pages`
  are now `std::span<const DeviceKVPageHandle>` (exact physical extents); the caches
  (`qwen3_6::PagedKVCache`) are the geometry + copy source/destination.
- `verify_pool` -> `DeviceKVPagePool` with `geometry().device_plane_order`; per-plane
  fingerprints unchanged (`Tensor` geometry words).
- The substrate **residual side-store half is gone**: no residual fields in source/target,
  no residual sections 12..17 (always 0), no `residual_*` header scalars (kRamVersion 3->4,
  kFixedHeader 372+... -> `348 + 6*16`, i.e. -24 B). Sections 2/3 stay in the format but are
  always length 0 (image bytes live in the arena, not the block).
- The **per-type GDN (`LinearAttentionStatePool`) and DFlash (`CyclicKVCache`) legacy state
  path is removed**: the baseline has no per-type host-image API for either; their state
  exists only in the unified `StateImage` container, so a record without a state image
  carries no linear-attention/cyclic state. The standalone hidden tensors
  (`tail_hidden`/`rewrite_checkpoint_hidden`, plain `cudaMemcpyAsync`) are retained as the
  only non-state-image state half. `verify_cyclic` deleted with them.

### Verified compile state
- Per-TU `g++ -std=c++20 -fsyntax-only` (container t33bg2-buildstage, image id 123801dd7584;
  include roots src/ include/ src/targets/qwen3_6/export/ + /usr/local/cuda/include):
  **EXIT 0** on `kv_ram_cache.cpp` (1173 lines) — the last substrate references are gone.
  Remaining: one pre-existing nodiscard warning (`r.u8()` skip in `read_header`, present at
  WIP HEAD, untouched).
- No other TU includes `kv_ram_cache.h` (grep-verified); CMake registration unchanged
  (already added in 8321291c).

## FINAL: full port complete, full build GREEN (commit a5ba1ee5)

The WIP half of the feature (never committed to any branch) is now ported onto
pure upstream master:
- `kv_ram_cache.{h,cpp}` re-targeted onto the baseline API: `create_active` /
  `mapped_pages` (count) / `physical_page(handle, idx)` / indexed page
  enumeration; capture/restore via `LogicalKVPageStore` + `DeviceKVPagePool`
  host-copy; state half via the state-image bridge (kRamVersion 4).
  `KvRamSnapshot` gained the two copy-seconds fields the stats plumbing reads
  (`save_seconds`, `load_seconds`, harvested via the WIP cpp).
- `request_plan_impl.h`: `plan_ram_reuse()` — terminal admission-time prefix
  match against the RAM tier (`reuse_source=HostRam`, `ram_entry_id`), MTP-aware.
- `program_impl.h`: `capture_retained_lane()` + `restore_ram_entry()` wired into
  the lane lifecycle; `kv_ram_capacity_bytes` member + constructor init.
- API/serve plumbing: `kv_ram_snapshot()` + `--kv-ram-mib` (0 = disabled) through
  `serve_options` -> `SequencePlanningInputs` (layouts.h) -> `SequencePlanImpl`
  (layouts_impl.h) -> `ProgramImplCore`; stats published via `engine_core.h`
  (`RuntimeStats` kv_ram_captures/restores/evictions/drops).

Verification (this session, buildstage-merge container, CUDA 13.1, arch 120a):
- Per-TU g++ -fsyntax-only: kv_ram_cache.cpp, 27b variant.cpp (pulls
  program_impl.h + engine), serve_options.cpp, generation_service.cpp — all RC=0.
- Full `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON` +
  `cmake --build build --target ninfer ninfer-serve` — **RC=0, green**
  (build log /tmp/v2-t8-wt/build.log).

Tier status: V2-T8 = ADOPTED (2 substrate picks + WIP port), code complete and
build-verified on quasar-master; runtime A/B (host-RAM hit path) still pending
a lane window per the doc's tier table.

---

## Test suite port (2026-09-09, commit 857f8ca7)

`tests/targets/qwen3_6/test_kv_ram_cache.cpp` (738 lines) registered as
`ninfer_qwen3_6_kv_ram_cache_test` (LIBRARIES ninfer_engine ninfer_core,
SKIP_RETURN_CODE 77). Re-targeted unit suite for the ported substrate:

- KV image round trip through HostKVArena (fill pages -> capture -> restore to
  fresh pool -> byte-verify), incl. non-contiguous physical page runs
- complete StateImage round trip (linear conv/recurrent, continuation hidden,
  DFlash local cyclic K/V)
- honest prefix_hash_chain index: longest match, frontier beats checkpoint,
  checkpoint fallback for short prompts, exclusive-claim hiding,
  consume-erase, multi-claim stays matchable + survives consume
- tiered FIFO eviction: demonstrated lineage -> protected record outlives cold
  records under capacity pressure
- dtor safety with in-flight copies + host-arena reuse afterwards

Verified: cmake reconfigure with BUILD_TESTING=ON (needs python3, apt-installed
in the buildstage image), ninja build of the test target RC=0 (compile + link),
binary runs the SKIP path (rc 77) without a GPU. GPU execution (ctest) is part
of the supervised runtime gate together with the host-RAM A/B + battery.
