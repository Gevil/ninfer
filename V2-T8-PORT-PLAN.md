# V2-T8 hand-port plan — gpillon RAM-KV agentic cluster

**Status: WIP — step 1 (substrate in, not yet re-targeted to the baseline; not building).**
Branch `v2/t8-agentic` off `v2/t5-decode` @ `324a8de3` (the current V2-T5 lane state, so the T8
picks stack on the shipped T1/T2/T3/T5). Lane untouched: this is code-work on a branch — no build,
no restart, no quadlet change until the gated ship step.

## Step 1 — the substrate (in)
`src/targets/qwen3_6/impl/runtime/kv_ram_cache.{h,cpp}` (1463 lines) + `kv_ram_snapshot.h`, brought in
verbatim from `gpillon/gpillon/coding`. This is the **temporal** RAM-KV role: caches finished/agentic
conversations by content hash (distinct from the baseline's `--host-kv-mib` device→host demotion mirror,
which is the *spatial* role, `HostKVArena`/`HostKVExtentStore`).

## Dependency surface (measured vs `v2/t5-decode`)
| Dependency | In baseline | Gap |
|---|---|---|
| `src/core/arena.h` | yes | `HostPinnedArena` is **gpillon-specific** — baseline `arena.h` has only `DeviceArena`/`DeviceBuffer`/`PinnedHostBuffer`/`DeviceSpan`. The substrate's `arena_` needs: `try_alloc(bytes,align)`, `can_alloc(bytes,align)`, `free(block)`, `capacity()`. |
| `src/core/{cyclic_kv_cache,linear_attention_state,paged_kv_cache,device}.h` | yes | verify API match during re-target |
| `include/ninfer/types.h`, `.../decoder_state.h` | yes | — |
| `prefix_identity.h` | yes, but **DIFFERS** | the substrate's `prefix_identity` usage re-targets to the baseline's version |
| `kv_ram_snapshot.h` | (substrate) | brought in |

The substrate does **not** reference the old `concurrent_executor.h` / `Executor` / `request_memory`
(the "admission" hits in `kv_ram_cache.h` are comments only). So the substrate compiles independently
of the executor re-target — that re-target belongs to the cluster picks (step 2).

## Re-target steps (in order, step 1b)
1. **Host-RAM pool (the design decision).** The substrate's `HostPinnedArena arena_` is a first-fit
   pinned-host block allocator. Baseline's `HostKVArena` is a handle/extent pool (different API).
   - *Interim to get it compiling:* add `HostPinnedArena` to `src/core/arena.h` (gpillon's ~40-line
     class, verbatim) **with a hard cap** (capacity bounded by a fraction of `--host-kv-mib`) so it is
     not the unbounded fork the scoping warns against.
   - *The intended design (decision 2, "shared, not forked"):* re-target `arena_` to draw on
     `HostKVArena`. Settled by the host-RAM-accounting A/B (shared vs capped-forked) on a multi-lane
     agentic load — that A/B is a later step, not a blocker for the substrate compiling.
2. **`prefix_identity.h` re-target:** align the substrate's `prefix_identity` usage to the baseline's
   version (the two differ).
3. **CMake:** add `kv_ram_cache.cpp` to `src/targets/qwen3_6/CMakeLists.txt`.
4. **Build:** per-TU `g++ -fsyntax-only` (buildstage container, lane live) → full build in a quiet
   lane window (a full build while the lane is live would OOM).
5. **Tests:** bring in `tests/test_kv_ram_cache{,_large,_opt,_perf}.cpp` + re-target (they construct
   `KVRamCache` + the surrounding runtime); run them in the quiet window.

## Step 2 (the multi-week bulk) — cluster picks + executor integration
The 4 cluster picks wire `kv_ram_cache` into the executor + serve:
`de386ad6` (+7483/−227, ~45 files, the base), `f144f052` (+284/−59), `27665883` (+68), `7bdee888` (+158).
Their `concurrent_executor.h` hunks re-target to the split executor: scheduling → `Scheduler`/
`EngineCore`, cache-policy → `ResourceManager`, RAM-snapshot/stats → `KVRamCache` + `EngineCore`.
The `HostKVArena`/`HostKVExtentStore` (the existing `--host-kv-mib` pool) is the integration point the
scoping decision 2 points to.

## Gate
Build + `kv_ram_cache` ctest + battery + a cache-hit/TTFT/decode A/B + greedy parity (V2-T8 is
decode-affecting via the cache-hit path). The shared-vs-forked pool is settled by a host-RAM-accounting
A/B. Ship only through the supervised pipeline (shipwatch supervisor on a non-lane model) — no
lane-touching job without it.
