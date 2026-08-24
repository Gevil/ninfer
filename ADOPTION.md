# Adoption record — personal fork of Neroued/ninfer

This is a **personal fork** of [Neroued/ninfer](https://github.com/Neroued/ninfer)
(linked fork: [Gevil/ninfer](https://github.com/Gevil/ninfer)), maintained for one
purpose: a fast single-GPU inference lane running
**Qwen3.8-27B NVFP4full on an RTX 5090** (cometkim's `nvfp4full` weights profile,
~225K-token INT8 KV, MTP3 spec decoding, vision).

Rather than wait on upstream, this fork **merges multiple open PRs from different
community forks and cherry-picks improvements and features** that are not yet in
upstream `master`, so the lane can use them. This file is the running record of
what is in, where it came from, and how it is verified. See the note at the top of
[README.md](README.md).

## Pipeline (how PRs are adopted)

1. **Base stays clean**: upstream `master` is fully contained in the lane branch
   (re-verified with `git rev-list --count HEAD..upstream/master` before every batch).
2. **One PR = one tagged merge commit**: `git merge --no-commit --no-ff <pr>` followed
   by a commit whose message names the PR
   (`merge upstream-PR#NN (local): <what + why>`). Keeping each PR in its own commit
   means a future rebase onto upstream master is a clean drop: once a PR merges
   upstream, its local commit disappears from the rebase automatically.
3. **Cherry-picks from forks** (when a fix only exists in a contributor's branch) land
   as regular commits that name the source fork/commit in the message.
4. **Conflict policy**: conflicts are pre-checked with `git merge-tree --write-tree`
   (the authoritative signal is a `CONFLICT (content)` line) and resolved **one PR at
   time — never stacked** (an unresolved merge poisons subsequent merges).
5. **Verification per batch**: rebuild the lane image from the branch, restart the
   lane, and check the boot ledger (`KV capacity explicit …`), the `/v1/models`
   contract, and a thinking-mode smoke test. Content gates compare trees against the
   pre-batch tip so metadata-only rewrites are provably behavior-neutral.

## What is in the lane (baseline, pre-adoption)

| Item | What it is |
|---|---|
| cometkim `feat/qwen3.8-nvfp4full` merge | full-backbone NVFP4 weights profile (`qwen3_8_27b_nvfp4full.ninfer`), natively registered by the engine |
| upstream PR #43 (Jinja chat templates) | vendored `third_party/minja` + `--chat-template-file` override; the lane swaps templates by file, no rebuild |
| `/v1/models` max_model_len contract | reports the configured `--max-context` ceiling as `max_model_len` (plus the webui `meta.n_ctx` dialect) so clients can size requests against the per-sequence cap |
| P1 `starts_in_reasoning` fix | derived from the rendered prompt instead of a brittle string probe |
| docker.io base-image pin | podman registry resolution for the build |

## Tiers

### Tier 1 — decode & serve quality (DONE 2026-08-23, `tier1` branch @ ac76ad21)

Open upstream PRs adopted in one batch; each lands as its own tagged merge commit:

| PR | What it brings |
|---|---|
| #55 | `usage.prompt_tokens_details.cached_tokens` on chat completions — prefix-cache reuse becomes visible to OpenAI-protocol clients |
| #67 | removes 3 redundant decode graph nodes — ~2% decode speedup on dense 27B (measured on sm_120a/CUDA 13.1.2) |
| #69 | L2-warms the next MoE consumer from the MoE tail — inert on our dense model, included for completeness with #67 |
| #65 | preserves string-typed tool params (a string `taskId` stays a string) — kills schema-reject/self-heal retries in agent loops |
| #57 | accepts content-part arrays in tool messages — Qwen-Code-style tool results stop 400ing |
| #61 | `--image-token-budget N` (draft) — caps vision tokens per image (default 0 = unchanged) |

**Verification (2026-08-23):** full 279-target production build clean on CUDA 13.1.2;
serve-layer suites 5/5 (`openai_schema` — incl. the new `cached_tokens` contract test and
three pre-existing stale-test fixes — `serve_options`, `tool_call_parser`,
`responses_schema`, `anthropic_schema`). Lane image `ninfer-nvfp4:latest` @ a73e184c8074
built from this tip; restart + runtime smokes (thinking-mode, `cached_tokens` presence,
decode-rate probe) recorded in `~/.local/share/ninfer/logs/tier1-restart-verify-2026-08-23.log`.
Public review: [Gevil/ninfer PR #1](https://github.com/Gevil/ninfer/pull/1)
(`tier1` → `qwen3.8-nvfp4full`, open).

### Tier 2 — the xhigh track (EXECUTED 2026-08-23, `tier2` branch @ 9a9d0964)

Built on the tier1 tip (`4eae6d15`); each item lands as its own commit (merge
commits for the PRs, so a future rebase drops them cleanly):

| Item | What it brings | Commit |
|---|---|---|
| **NEW-A** | `chat_template_kwargs.reasoning_effort` forwarded into `GenerationRequest` — the Sharp template's per-request effort steering (same string validation as the top-level field; top-level wins on agreement, explicit disagreement = 400 `conflicting_template_option`; enable_thinking conflict rule applies to the kwargs channel) | `0e498b5b` + unit tests |
| **NEW-B** | chat template swapped to Sharp **v22.3.1** (peculiar-ragdoll `Qwen-Sharp-Chat-Templates` @ `98a76e8`, 28174 B, sha256 `8753210e…`) — tool-error escalation tiers, no false retry loops, system-message merge, `tojson` tool-arg serialization, prefix-cache-stable empty-think, fast-mode fixes, effort aliases (high/xhigh/max). File swap on the bind-mounted path — no rebuild | template swap, `.bak` kept |
| **#79** | `fix(serve): decouple warmup from client-facing request deadline` (local adaptation: tier1's webui-dir block re-inserted at the top of the new runtime try-block) | `f91b7c85` |
| **#54** | `fix(serve): log the exception that terminates the process` (`std::set_terminate` handler; local adaptation: install kept alongside the options declaration) | `9a9d0964` |

**#50 evaluated → skipped (skip-list entry):** the chat-completions kwargs
`enable_thinking` is already functionally present in this tree (top-level-wins
semantics, established by the jinja work and live-tested); pr-50's only delta is a
4-line Responses-API hunk that carries no value for this lane's clients (OMP
openai dialect, webui — both steer via chat-completions kwargs). Re-audit only if
the Responses endpoint gets a template-kwargs client.

**Verification (2026-08-23):** full 279-target production build clean (CUDA 13.1.2,
buildstage `ceb8be98981b`); ctest 2/2 (`ninfer_openai_schema_test` incl. the new kwargs
contract tests, `ninfer_serve_options_test`). Lane image `ninfer-nvfp4:latest` @
b4791be57a90 built from this tip; restart battery all-PASS: `/v1/models` 225280
contract, thinking smoke, **xhigh steering smoke** (kwargs `reasoning_effort=xhigh` →
729-char trace; 40-token prompt delta vs the no-effort run = the v22.3.1 steering line),
NEW-A 400 contract ×3 (unknown value / conflicting spellings / enable_thinking=false,
correct codes + channel attribution), decode 136.9 tok/s wall + `cached_tokens`
regression. Log: `~/.local/share/ninfer/logs/tier2-restart-verify-2026-08-23.log`.
Public review: [Gevil/ninfer PR #2](https://github.com/Gevil/ninfer/pull/2)
(`tier2` → `qwen3.8-nvfp4full`, open).

### Tier 3 — cherry-picks from community forks (WAVE A EXECUTED 2026-08-24, `tier3` branch @ 22c46df7)

Built on the pre-fix tier2 line; each pick lands as its own commit naming the
source fork/commit. The stray-vision-pad fix (the 400-storm fix) is backported
from tier2 (`caec0dee`, patch-identical to tier1 `7e845d45`) because the branch
was cut before it landed.

| Pick | Source | What it brings | Commit |
|---|---|---|---|
| CUDA-graph allowance | dylanbrodiefafard/experimental `d85278b1` | stop over-reserving CUDA graphs at long context | `d48c9e62` |
| INT8 KV fill | dylanbrodiefafard/experimental `7374b27a` | page-tiled INT8 KV fill for 27B GQA | `b82f1ed3` |
| decode-side MTP | dylanbrodiefafard/experimental `0787d7cf` | cut AR hidden D2D copies + last-chunk host sync | `03514cfa` |
| MTP/draft fuse | dylanbrodiefafard/experimental `dd1bd651` | residual_add + rmsnorm fused into one op on the MTP/draft path | `ebe76d22` |
| nvfp4 short-T decode | dylanbrodiefafard/experimental `bc341b75` | streamed decode weights + unrolled short-T MLP-down | `98bbc47d` |
| GDN prefill conv | MichaelDementii/chunked-prefill `286627dd` | GDN prefill convolution written straight into q/k/v | `ceddaf28` |
| decode-suite (2) | MichaelDementii/decode-suite `10fc5bbd` + `f8ad432a` | shared-expert down-weight prefetch behind the moe router; gated post-mixer norm folded into the GDN mixer tail | `40022442` `32c0ded2` |
| tool-call prefix reuse | dylanbrodiefafard/fix-tool-calls-checkpointing `673d695c` | closed tool turn (empty think block) still hits response prefix reuse — checkpoint snapshotted after the open think block; 3-hunk test conflict resolved keeping both sides' tests | `281cf9ec` |
| stray vision-pad fix | local tier2 `caec0dee` (≡ tier1 `7e845d45`) | `expand_placeholders` strips stray image/video pad markers (e.g. re-injected via `--preserve-thinking`) instead of 400ing `invalid_media` | `22c46df7` |

**Verification (2026-08-24):** buildstage production build clean from this tip
(CUDA 13.1.2); ctest 6/6 (`openai_schema`, `serve_options`, `tool_call_parser`,
`responses_schema`, `anthropic_schema`, `qwen3_6_frontend` — the frontend suite
needs the official Qwen3.6-27B tokenizer resources bind-mounted at the upstream
path; environment note, no test changes). Lane image `ninfer-nvfp4:tier3-<sha>` +
`:latest` built from the public clone; restart battery all-PASS (boot ledger,
`/v1/models` 225280 contract, thinking + xhigh smokes, kwargs 400 contracts ×3,
decode probe, `cached_tokens` regression). Log:
`~/.local/share/ninfer/logs/tier3-restart-verify-2026-08-24.log`.
Public review: [Gevil/ninfer PR #3](https://github.com/Gevil/ninfer/pull/3)
(`tier3` → `qwen3.8-nvfp4full`, open).

**Wave B (EXECUTED 2026-08-24):** `dylan/experimental` decode-path perf adopted
(Wave B1, below); `eason/develop` SKIPPED (inert on the 1-GPU lane); the
NVFP4-KV `--sage` family REJECTED, vLLM work out of scope — decision
2026-08-24: **the KV-cache precision floor is INT8/FP8: no lower-precision KV
(NVFP4/sage/hyperquant/compressed-KV) is adopted, and vLLM is completely
ignored.**

**Tier 4 (PROPOSED 2026-08-24 — "upstream convergence + agent-workload" tier:
new `tier4` branch stacked on `tier3-waveb`, per user's "open a new tier"
direction).** Upstream `Neroued/ninfer` audit (2026-08-24: 26 open issues,
32 open PRs, Discussions disabled) compared against the shipped tree (Wave A
+ B1 + 22c46df7 vision fix; lane runs code @ 89354ca7):

- **Already covered in-tree (no action):** #24 (context in `/v1/models` —
  battery contract), #43 (Jinja `--chat-template-file` — Sharp template path),
  #54 (exception naming — B1 `55036778`), #55 (`cached_tokens` — tier2), #57
  (tool message content parts — tier1 `fdcf6839`), #61 (image-token budget —
  tier1 `325986ea`), #65 base (tool-param typing — tier1), #79 base (warmup
  decoupling — tier2 `f91b7c85`), top-level `reasoning_effort` (#60: parsed in
  the chat path, `openai_schema.cpp:517/567` + tier2 kwargs path — OMP's
  `xhigh` is handled; a behavioral confirmation belongs in the battery).
- **REJECTED per the precision-floor decision:** #35 compressed-KV E8 lattice
  (pr-35 family) and `cometkim/feat/hyperquant` (54 commits).
- **WAVE C1 — serving correctness for agent sessions (small, high value):**
  - #86 (mr-september, 2026-08-24): drop stray think-close markers leaking
    into tool-capable output (reproduced live on Qwen3.8-27B NVFP4 — the
    exact "model error" class our OMP sessions hit);
  - `99d39090` (pr-65 follow-up): preserve string-typed tool params instead
    of eager-deserializing (fixes issue #66 — string params containing
    JSON-shaped text ship coerced types);
  - #88 (geoffwatts, 2026-08-24): U+2060 word-joiner neutralizes literal
    Qwen vision-token spellings inside text — complements our 22c46df7 strip
    fix (covers the "chat media order does not match rendered placeholders"
    class; NOT in tree);
  - #79 follow-up: fail-fast warmup exit — verify the tier2 adoption already
    covers it, pick the diff if not;
  - `9ed62e70` (dylan/chat-template): skip empty think wrappers on past
    assistant turns (qwen3.8 template fix).
- **WAVE C2 — decode perf (measured; bit-exact where applicable):**
  - #67 (MichaelDementii): remove three redundant decode graph nodes,
    +2.3% decode (nsys: `sigmoid_mul` 10/step, q/k `rmsnorm_warp` 20,
    `sparse_moe_d2` routing 40 — measured on 35B; model-fit check on 27B);
  - #85 (devan-carlin): adaptive CUDA-graph allowance (linear interpolation
    vs our step function `d85278b1`) — the graph table jumps 2 MiB@128k →
    223 MiB@256k, so at our 225280 context the step function undershoots;
  - `cfb96526` GDN 27B gating-proj cooperative fuse + `604bdc5f`
    thinking-preserving prefix reuse (dylan/perf/rtx5090-qwen38) —
    grid-safety check per issue #39 (GB203 laptop overflow; our GB202
    desktop has ~2× the SMs, should fit — verify the cooperative grid
    sizing);
  - `md` micros `90d4c423`/`8330672c`, `baseopt` m13/m14, `adaptive-gamma`
    v2.1 (+1.9% mean, opt-in) — model-fit check (35B origin);
  - #69 (MichaelDementii): warm the next consumer's L2 from the MoE tail,
    +3.7% decode — needs a cross-layer contract; evaluate last.
- **WAVE C3 — agent-workload KV lifetime (precision-neutral, opt-in):**
  - #64 (gzenz; = dylan/ram-cache + origin/pr-64): host-backed KV cache —
    park evicted lanes to pinned RAM;
  - #73 (iamwavecut): content-addressed host KV (`--kv-host-cache-mib N`,
    N=0 inert) — 118k prefix restore 422 ms vs 53.4 s cold (126×), branch
    switch over a shared prefix ~100 ms (subagent workloads); the #75 design
    doc converges #64 as fast path + #73 as shared page substrate.
- **Battery additions (per wave):** 5090 land-gate T=4 decode harness
  (`332b7a97`/`eeb6c9b3`/`596940d5`/`0f05e643`), think-close marker
  regression (C1), tool-params schema check (C1), host-KV restore probe
  (C3).
- **Flagged non-code decision (user, from #38):** the official Qwen3.8-27B
  NVFP4 artifact is ~20 GB vs the community artifact's ~14.3 GB → ~20%
  decode-speed ceiling (weights read per step; maintainer-confirmed in
  issue #38). The lane runs the official artifact (GPQA 89.39% gate).
  Switching to the community artifact = ~20% faster decode, quality
  unverified (maintainer has asked for a reproducible comparison; none
  exists yet).

### Tier 3 Wave B1 — dylan decode-path perf (EXECUTED 2026-08-24, `tier3-waveb` branch)

Three always-on decode picks from `dylanbrodiefafard/experimental` (no feature flag;
the current int8-KV lane takes them as-is). Each lands as its own commit naming the
source fork/commit.

| Pick | Source | What it brings | Commit |
|---|---|---|---|
| SwiGLU W4A4 evict-first prereq | dylan/experimental `46d2f59e` | T=4 SwiGLU runs W4A4 with L2 evict-first weights; adds `Cache::EvictFirst` to `ops/common/memory.cuh` + `memory_evict.cu` noinline helper (inlining createpolicy+cache_hint into large MMA is illegal on sm_120/ptxas 13.1), kernel `Cache` template param | `55036778` |
| staged split-KV reduce | dylan/experimental `83b16e71` | the small-T attention reduce bulk-stages its partial column (acc slice + m/l stats) into dynamic smem via cp.async and runs the m/l/acc tree over smem — 3 serial global-load chains became one parallel bulk copy; bit-identical numerics (4-way quarter-sum order unchanged); reduce 19.2→7.2us (T=1), 22.2→13.8us (T=6); `NINFER_SMALL_T_REDUCE_DCHUNK` knob (default 64). Hand-merged with the tier2 `ed505ebc` fused-sigmoid-gate epilogue: both stores stay gate-aware with the exact BF16-replicated arithmetic | `0d5efa28` |
| merge fix | local | the 83b16e71 hand-merge dropped one closing brace (the `if (q == 0)` closer in the reduce epilogue), leaving the kernel function scope open and leaking the namespace into the launcher TU's later includes (`ninfer::ops::std` errors, unbound `cp_async`). Closed the scope; production build + ctest re-verified | `89354ca7` |
| GEMV cp.async evict_first | dylan/experimental `2660be68` | in the T<=4 one-shot GEMV (MTP verify/decode) every weight byte is streamed once, so the weight cp.async loads are marked `Cache::EvictFirst` (L2 evict-first) at all W4A4 sites (gdn_input_proj / attn_input_proj / linear / linear_add) — the 10s-100s MB stream no longer displaces the downstream consumer's L2 set; quality-neutral (identical bytes, eviction priority only) | `e481ccd8` |

**Wave B evaluation record (2026-08-24, per ADOPTION plan "evaluated per commit"):**
- dylan/experimental series (98972696..2660be68, 21 commits): 2 PICKs inside
  the window (+ `46d2f59e` just below it, 3 total); the NVFP4-KV
  `--sage`/s3/sparge family (49790f72, 7dbbbcd0, 24c23686, 26276e0b,
  ea0cd367, 060dba3b, 7ea7efc3, a37580f2, 0f2bd013, d98fb9a4) REJECTED
  2026-08-24 — KV-cache precision floor is INT8/FP8, no lower-precision KV,
  vLLM ignored; 9 research/tooling/bench commits (kdev control plane, TMA A/B
  harness, NIAH fixtures, op-dumps, PPL campaign `98972696`, bench sweeps,
  merge 9808de6b = pure history-join) SKIPPED.
- eason/develop series (117d83e0..9168239b, 5 commits): SKIPPED — the
  cross-GPU VRAM cold-tier chain is inert on the 1-GPU lane (auto-disable, zero
  VRAM/behavior change) and its exact-`cached_tokens` half duplicates the in-tree
  pr-55 reporting (`usage.prompt_tokens_details.cached_tokens` from
  `prefix_cache_hit_tokens`).

**Verification (2026-08-24):** buildstage production build clean from this tip
(CUDA 13.1.2); ctest 6/6. Lane image `ninfer-nvfp4:tier3-waveb-<sha>` + `:latest`
built from the public clone; restart battery all-PASS (boot ledger, `/v1/models`
225280 contract, thinking + xhigh smokes, kwargs 400 contracts, decode probe
vs 154.5 tok/s Wave-A baseline, `cached_tokens` regression). Log:
`~/.local/share/ninfer/logs/tier3-waveb-restart-verify-2026-08-24.log`.
Public review: [Gevil/ninfer PR #4](https://github.com/Gevil/ninfer/pull/4)
(`tier3-waveb` → `qwen3.8-nvfp4full`, open, stacked on PR #3).
### Tier 4 — upstream convergence + agent-workload (EXECUTED 2026-08-24, `tier4` branch, stacked on `tier3-waveb`)

Three waves per the Tier 4 plan (`7bafd8f3`), picked per commit with the source
fork/commit named in each commit message. C1 = correctness for agent sessions,
C2 = measured decode perf, C3 = agent-workload KV lifetime (all opt-in,
inert unless flagged).

**WAVE C1 — correctness fixes (agent sessions):**

| Pick | Source | What it brings | Commit |
|---|---|---|---|
| #86 stray think-close markers | mr-september `fix/stray-think-close` (5662e1a7+51d51723+2d00ed0f) | `</think>` markers leaking into tool-capable content are dropped in the rendered chat | `e4beff22` |
| #65 string tool params | mr-september `99d39090` (port of upstream PR #65; supersedes local `a8cdc1a6` + issue #66) | string-typed tool params preserved instead of eager-deserializing (agents passing structured-but-string params stopped 400-ing) | `102ab113` |
| empty think wrappers | dylan/chat-template `9ed62e70` | an empty think block on a history turn is Qwen's no-thinking cue and poisons later reasoning; history assistant turns without reasoning content now omit the wrapper. **Test port adjustment** (`1ffbc388`): the pick's two preserve-on replay-checkpoint assertions encode the *upstream* prologue byte-layout; our tree keeps the verified BPE-stable prologue from `281cf9ec` (checkpoint at the `<think>` boundary, one byte before the trailing newline), so the two assertions were aligned to our layout — no engine-code change | `16f405d4` + `1ffbc388` |
| pr-88 literal vision tokens | upstream PR #88 tip `1c4bfa55` (cherry-picked as `6da2efef`) | neutralizes the exact Qwen Vision control-token spellings in textual chat inputs so only structured image/video parts create media placeholders — literal tokens that leak into re-injected thinking text can no longer produce unbound placeholders (the 400 `invalid_media` class); also preserves Anthropic `tool_result` and nested-content order. Pairs with the in-tree processor strip (tier3 backport of the tier1/tier2 `7e845d45`/`caec0dee` vision-pad fix) to fully close the 400 class | `6da2efef` |
| #79 fail-fast follow-up | SKIPPED | no new follow-up commits since the last audit pass; the fail-fast behaviour is already in-tree | — |

**WAVE C2 — decode perf (measured; bit-exact where applicable):**

| Pick | Source | What it brings | Commit |
|---|---|---|---|
| #85 adaptive graph allowance | devan-carlin `21bdfae2` (port of upstream PR #85; supersedes Wave A step-function `d85278b1` at the call sites) | CUDA-graph table allowance scales linearly with context window instead of our step function (the step table jumps 2 MiB@128k → 223 MiB@256k, undershooting at our 225280 context); conflict resolved PR-side at both call sites | `988d72e8` |
| #67 graph-node removal | MichaelDementii `a2de4e32` | already in-tree before Wave C started (origin/pr-67 head is an ancestor) — nothing to do | (in-tree) |
| #69 MoE-tail L2 warm | MichaelDementii `5750f15c` | already in-tree via Wave A `10fc5bbd` (its content is the MoE-tail prefetch) — nothing to do | (in-tree) |
| dylan 5090 GDN gating fuse | dylan `cfb96526` (perf/rtx5090-qwen38) | 27B small-T GDN gating proj fused into one cooperative kernel; grid-safety check per issue #39 (GB203 laptop overflow — our GB202 32 GB is unaffected) | `04090445` |
| 604bdc5f / 064bdc5f (dylan 5090, thinking-preserving prefix reuse) | SKIPPED (deferral) | perf optimization written before our C1 correctness chain; deep structural collision in the frontend reasoning area (removes/restructures `prompt_ends_in_open_reasoning` / `opens_reasoning` that C1 + media-cache work rewrote). Per the "evaluated per commit" policy: record as deferral, re-evaluate after C1 settles (can apply cleanly on top of a later state) | — |
| md micros `90d4c423`/`8330672c`/`f79acd8` | SKIPPED (covered in-tree) | `90d4c423` ≡ #67 (`a2de4e32`, in-tree); the q/k-rmsnorm fusion content is already in-tree via the q/k-norm work + the baseopt m13 port below | — |
| baseopt m13/m14 | md/research `f0f61cc1`/`680ec4ae` | MTP draft head routed through the fused q/k rmsnorm op (code already present via q/k-norm work; comment added) + sigmoid gate handed to the attention epilogue (the fused-gate param was already in our `gqa_attention` signature); two trivial conflicts resolved | `6634ea70`, `b3814f6c` |
| adaptive-gamma v1→v2.1 | md/research `62f46358`/`39768733`/`28ee8a3c` | opt-in MTP gamma policy: unbiased censored-geometric alpha (decayed-counter ratio) + 4-round hold; A/B vs fixed mtp4: prose +11%, json +16%, code +13% (first rep), corpus verdict +1.9% mean (the `5c82bed1` verdict commit is an empty marker — content landed in v1–v2.1). OFF by default: zero behaviour change unless enabled | `ef4e45c7`, `6b39770d`, `12633ff1` |

**WAVE C3 — agent-workload KV lifetime (precision-neutral, opt-in):**

| Pick | Source | What it brings | Commit |
|---|---|---|---|
| #64 host-KV lane parking | gzenz `07bf8b4b`→`d6f3fcc4` (4 commits, = dylan/ram-cache) | park evicted lanes to pinned RAM instead of discarding; `--host-kv-cache-mib N` (flag renamed in `43cd3418`), variable-size pinned entries, evicting restore, A/B bench. Inert when unset. Conflicts resolved: usage-string/help union with our webui line, test-CMake union with jinja + host-kv targets | `0bb10f9a`, `967c2ba5`, `8a14316c`, `4a5e5646` (+ `8dde1013` marker fix) |
| #73/#74 content-addressed host KV | dylan `d0b3db41`→`2bd350cb` (9 commits) | `--kv-host-cache-mib N` (N=0 = off): pinned-host content cache — previously computed prefixes restore through PCIe instead of re-prefilling, branches sharing a prefix dedup against the same pages; in-flight identical-prompt coalescing (#74); flag-conflict validation, stale-plan revalidation, store invariants + t-invariance tests. **Merged alongside #64: the two opt-in host-KV subsystems coexist (fields/flags/CMake unioned), both inert by default** — see open question below | `7fa78bec`, `0d45cfce`, `b5ecbd1b`, `a5202b29`, `258bd89b`, `57d52d5e`, `e30d0843`, `ad89f3d3`, `beb2c7ba` |
| VisionResidency types backport | minimal from `919124dd` (skipped overlay series) | the `d0b3db41` serve/generation wiring needs the `VisionResidency` enum + two `EngineOptions` fields; backported minimally. The overlay *engine* implementation is intentionally NOT ported (upstream WIP "compile pending", 35B-targeted): the `--vision-residency overlay` flag is accepted/validated but the engine behaves as `resident` until the overlay series is ported. Dangling `vmm_graph_remap` + `evictable_weight_pool` test targets dropped (their sources live in the unported overlay commits) | `839e3547` (+ `cd8b4be4` duplicate-field build fix) |

**Open questions (for the user):**
- Two parallel opt-in host-KV subsystems now coexist (#64 lane-parking `--host-kv-cache-mib` +
  #73 content-addressed `--kv-host-cache-mib`). Upstream #75 proposes converging them.
  The lane deploys with **both flags off** (zero behaviour change); pick one at runtime once
  measured. The content-addressed one is the superset (prefix dedup + trajectory restore);
  the parking one is the simpler budget-bounded variant.
- adaptive-gamma (v2.1) is OFF by default; enable per-model when the +1.9% mean is wanted.

**400 `invalid_media` class (this session’s starting error):** fully covered — the processor strips stray pad tokens (tier3 backport of the tier1/tier2 `7e845d45`/`caec0dee` vision-pad fix, in-tree since the tier3 Wave A backport) and `6da2efef` neutralizes literal vision tokens at the template layer.

**Verification (2026-08-24):** buildstage production build from this tip (CUDA 13.1.2);
ctest standard 6 suites + host-KV suites (host-KV GPU suites verified on a free GPU in the
ship phase — the live lane holds 31.6/32.6 GB VRAM, so pinned/GPU allocations OOM while it
runs: environmental, not code). Lane image `ninfer-nvfp4:tier4-<sha>` + `:latest` built from
the public clone; restart battery all-PASS + C1/C3 probes (stray-think-close, string tool
params, host-KV flags inert). Logs: `~/.local/share/ninfer/logs/tier4-restart-verify-2026-08-24.log` (battery) + `~/.local/share/ninfer/logs/tier4-hostkv-ctest-2026-08-24.log` (free-GPU ctest).

**Live battery verdicts (lane, 2026-08-24, log ~/.local/share/ninfer/logs/tier4-restart-verify-2026-08-24.log):**
- /v1/models: http=200 after ~10s
- VERDICT MODELS: PASS (expect qwen3.8-27b-nvfp4full)
- VERDICT IMAGE: PASS (lane runs latest tag)
- VERDICT think-smoke: PASS
- VERDICT THINK-SMOKE: PASS
- VERDICT xhigh: PASS
- VERDICT XHIGH: PASS (trace 729 chars)
- VERDICT decode: PASS
- decode: 334 tokens in 3.0s = 111.2 tok/s (baseline 136.9; decode-path changes should hold or improve) cached_tokens field present = 0
- VERDICT DECODE: PASS

**Host-KV ctest (free GPU, buildstage-tier4, log ~/.local/share/ninfer/logs/tier4-hostkv-ctest-2026-08-24.log):** 100% tests passed, 0 tests failed out of 5; Total Test time (real) =   2.95 sec

Public review: [Gevil/ninfer PR #5](https://github.com/Gevil/ninfer/pull/5)
(`tier4` → `qwen3.8-nvfp4full`, open, stacked on PR #4).
### Wave C — post-Tier-4 candidate list (planned 2026-08-24, `tier5` branch → PR #6, stacked on PR #5)

Audited 2026-08-24: all remotes fetched; classification by content (`git cherry`/patch-id), not SHA counts —
cherry-pick artifacts inflate raw `rev-list` counts. Upstream is fully in tier4 (`tier4..upstream/master` = 0).

| # | Candidate | Source | Status (2026-08-24 audit) |
|---|-----------|--------|----------------------------|
| C1 | thinking-preserving prefix reuse: pair is `6065a6b3` (fix(serving): preserve ordered instruction turns) → `604bdc5f` (perf(runtime): optimize thinking-preserving prefix reuse) — the B2 list's "064bdc5f" was a typo for `6065a6b3` | dylan/perf/rtx5090-qwen38 | **CONFLICT TRIAGE (2026-08-24):** the only C2 item deferred on "re-evaluate after C1 settles"; tier4 shipped, re-picked on `tier5`: `6065a6b3` conflicts in 4 files (serving docs, `chat_template.cpp`, `processor.cpp`, `test_frontend.cpp`), `604bdc5f` in 10 files (frontend + runtime core). The 5090 branch (2026-08-16) predates the tier3/4 frontend work (vision-pad fix, think-close series, pr-88 vision-token escaping) → keep-both resolution. Resolution pass is the wave's first build/ctest gate  **FINAL (2026-08-24): ABORTED — superseded.** Re-evaluated after tier4 shipped: the fix half (`6065a6b3`) was already incorporated upstream (net diff = 4 documentation lines); the perf half (`604bdc5f`) carries **divergent checkpoint semantics** from tier4 host-KV/ContentRestore (their checkpoint rolls to `**`; ours keeps the last assistant opener on a newly closed turn) — merging the two parallel rewrites risks subtle KV-state bugs. A dedicated porting pass (not a cherry-pick) is required; recorded as a deferred Tier 5 follow-up, not in this ship. |
| C2 | MoE-prefill widening: stage MoE activations from x (`9c92220b`), W8 dequant in row-split MMA (`958c594d`), Q6 down-weight decode (`27825d7e`), Q5/Q4 down-weight decode (`f082ac35`) | md/perf/chunked-prefill | **PICKED (2026-08-24)** — all 4 cherry-picked cleanly onto `tier5` (no conflicts): `f082ac35`→`a290f5a1`, `27825d7e`→`76ed67c7`, `958c594d`→`3ea9a778`, `9c92220b`→`0a52d8be`. Independent ops (MoE prefill decode width, activation staging); gated by the wave build + ctest + battery decode probe. |
| C3 | pr-86 base `cdc3bb8e` (drop stray think-close markers from content output; ~22 lines frontend.cpp + tests) — `git cherry` = absent: the `e4beff22` squash only brought in the 3 commits built on top of it | origin/pr-86 | **FINAL (2026-08-24): ABORTED — superseded.** The stray-think-close fix exists in tier4 in its refined form (the 3 commits squashed into `e4beff22`: hold split markers, keep leading-strip, skip raw sessions — the improved `if (state.in_reasoning)` block with `think_marker_pending` is already in-tree); every hunk of `cdc3bb8e` is keep-ours (net-zero pick, abort recorded). |
| C3b | MTP D2D-copy cut `c50fcf58` + short-T weight streaming `ad930145` | dylan 5090 | **CLOSED** — patch-equivalent in tier4 (`git cherry` = `-`), covered via tier3 picks |
| C4 | DFlash-2 (27B) spec-decode line: W8 QKV route for the 3-output attn_input_proj, SIMT fp32 small-T QKV, SWA window fixes, `nvfp4-dflash2` weights identity (+66-object `dflash/` section), staged acceptance — 70 commits (2026-08-23/24), community work, not upstreamed | taylor-shift/ninferno `dflash2-27b` (fetched as `audit/ninferno-dflash2-27b`) | **ANALYSIS (2026-08-24, user: "analyze for single 5090 + nvfp4"): no useful standalone commits.** ~35 deploy/fleet commits (RunPod/Vast/LiteLLM multi-GPU) N/A to a single-GPU quadlet lane. Engine picks checked one by one: `2565bb3f` (SM-count launch geometry) = verified no-op on our 170-SM 5090 (commit proves all derived values reproduce the old constants); `84663ebb` (concurrency 16) = opt-in, no benefit at 32 GB single-user; `666771df`/`6949d45c` (device binding) = multi-GPU fix, no-op single GPU; `0ce264ee` (GDN conv) = 27B target has no GDN layers. The entire DFlash-2 block (`afd1bcd4` drafter + ~25 commits) is **gated on the author's dflash2 drafter checkpoint** (`--dflash-model` dir with an exact BF16 tensor set, verified by `dflash2.py` preflight; only present in the author's WSL, not published). → **MONITOR**: revisit as its own wave if the DFlash-2 line upstreams to Neroued/ninfer or the checkpoint is published |
| C5 | Sharp chat template: live file = **v22.1** (23026 B, md5 `e21426b7`) vs the NEW-B record (v22.3.1) — mtimes show a 2026-08-23 15:12 rewrite + both `.bak` snapshots, so either the swap silently wrote the old file or it was rolled back; upstream since then: v22.1.1 (fast-mode think Fix 3), v22.3.1, **v22.3.2** (per-request `terse` kwarg toggle, default on; Jinja-guarded → upgrade needs no engine patch; using `terse:false` needs a NEW-A-style whitelist line in `openai_schema.cpp`) | peculiar-ragdoll/Qwen-Sharp-Chat-Templates @ `9d98369` (cloned to `/tmp/sharp-templates-audit`) | **DECIDED (2026-08-24): upgrade to v22.3.2, battery-gated** — **terse wiring committed `d19efff1` (2026-08-24)**: the `terse` kwarg is now wired through the OpenAI kwargs channel (whitelist + parse → `GenerationRequest.terse` → `PromptOptions.terse` → `ChatRenderOptions.terse` → minja context, mirroring the NEW-A `reasoning_effort` forwarding; content-addressed KV cache stays safe since rendered bytes differ). Template swap pending with the lane ship: bind-mounted file v22.1 (23026 B, `d1f22a89`) → v22.3.2 (`/tmp/sharp-templates-audit/chat_template.jinja` @ `9d98369`, 28577 B, sha `3071f3ea`), current file backed up first. Swap + restart + battery gate. |
| C6 | open PR #87: `feat(serve): accept json_object/json_schema response_format with prompt injection` | Neroued/ninfer (open, post-audit) | **PICKED (2026-08-24)** — `fac7f932` cherry-picked as `a568115b` (1 comment-only conflict in `openai_schema.h`, resolved by merging both comment sides; json_object/json_schema response_format now accepted with prompt injection, matching the llama.cpp soft-JSON approach). |

Also: "upstream #75" (host-KV convergence, open question above) is a **public 404** on GitHub (open PRs run to #88) — internal reference, no local mirror; stays an open question until mirrored.

**Execution order:** C1 + C2 + C3 + C6 → cherry-picks onto `tier5` (per PR: `merge-tree` triage then a real `git merge --no-commit --no-ff`, never continue a batch after an unresolved conflict) → build + ctest in the CUDA container (host-KV suites on a free GPU) → lane ship (restart battery + probes) → ADOPTION Tier 5 record + push `tier5` + PR #6 (stacked on #5). C4: audit in parallel; if "adopt", it becomes its own wave (not folded into C1–C6). C5: after the user decision, independent restart + battery.

## Lane build

The lane image is built **from this branch** (public repo clone → `podman build`),
then the lane is restarted and verified; build/verification logs live in the host's
`~/.local/share/ninfer/logs/`. The lane is the verifier of its own history: the model
serving developer sessions runs on it.

*Record maintained 2026-08-24 — Tier 1 merged + verified (PR #1 open); Tier 2 executed + verified (PR #2 open); Tier 3 Wave A executed + verified (PR #3 open). Update this file at each tier boundary.*