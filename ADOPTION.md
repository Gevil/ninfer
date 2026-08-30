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

**Wave C execution record (2026-08-24, tier5 lane ship):**
- C2 (4 MoE-decode picks `a290f5a1`/`76ed67c7`/`3ea9a778`/`0a52d8be`), C6 (`a568115b` response_format json), and the C5 `terse` kwarg wiring (`d19efff1`) all picked onto `tier5`; C1/C3 recorded as superseded (see table above)
- C5: bind-mounted Sharp template v22.1 (`d1f22a89`) -> **v22.3.2** (`3071f3ea`, 28577 B) with the per-request `terse` kwarg (default on; `terse:false` now usable via `chat_template_kwargs`, engine-side in `d19efff1`)

**Live battery verdicts (lane, 2026-08-24, tier5 image + v22.3.2 template):**
- /v1/models: http=200 after ~10s
- VERDICT MODELS: PASS (expect qwen3.8-27b-nvfp4full)
- VERDICT IMAGE: PASS (lane runs latest tag)
- VERDICT think-smoke: PASS
- VERDICT THINK-SMOKE: PASS
- VERDICT xhigh: PASS
- VERDICT XHIGH: PASS (trace 733 chars)
- VERDICT decode: PASS
- decode: 300 tokens in 2.6s = 114.0 tok/s (baseline 136.9; decode-path changes should hold or improve) cached_tokens field present = 0
- VERDICT DECODE: PASS

**Host-KV ctest (free GPU, buildstage-tier5):** 100% tests passed, 0 tests failed out of 5

Public review: [Gevil/ninfer PR #6](https://github.com/Gevil/ninfer/pull/6) (`tier5` -> `qwen3.8-nvfp4full`, open, stacked on PR #5).

## Lane build

The lane image is built **from this branch** (public repo clone → `podman build`),
then the lane is restarted and verified; build/verification logs live in the host's
`~/.local/share/ninfer/logs/`. The lane is the verifier of its own history: the model
serving developer sessions runs on it.

*Record maintained 2026-08-24 — Tier 1 merged + verified (PR #1 open); Tier 2 executed + verified (PR #2 open); Tier 3 Wave A executed + verified (PR #3 open). Update this file at each tier boundary.*
**2026-08-26 (master):** the Tier 7 merge (20798a3b) was REVERTED on request - the ~50% decode
regression (142 -> ~55-71 tok/s) was not accepted. Lane rolled back to the Tier 6 image
(aaff16e226eb, `tier6-156424cd`): probe 146.1 tok/s fresh-context, 175.9 tok/s at 98k context
with MTP acceptance ~71%. Model mainline `qwen3.8-nvfp4full` restored to the Tier 6 tree
(1b9aef3b). Tier 7 remains available on the `tier7` branch (MTP width-invariant parity work
still stands - it just costs the decode fast paths); resuming it requires the fast-path
recovery work recorded in `~/.local/share/ninfer/tier8-experiment-record.md` (pass 2 design).

## Audit 2026-08-25 — new upstream PRs + forks (not yet merged)

Upstream master unchanged (`feaf4dd0`). Upstream PR activity 08-22 → 08-25:
- **#89** (igorls, OPEN, `f52e2099`): size persistent grids from the active device's SM count — 1 commit / 5 files; `merge-tree` vs `tier5` conflicts in `sparse_moe_prefill_kernels.cu` (our MoE edits). Candidate, small.
- **#90** (igorls, OPEN **DRAFT**, 31 commits, tip `db7a1220`): cross-request prefix seeding via a content-addressed `PrefixSeedStore` (device arena behind `--prefix-cache-mib`, copy-only restore, `reuse=seed_prefix`). Sibling/alternative to shipped #73 (host content cache). Upstream architecture discussion open — **watch, do not merge** until settled. Standalone sub-fixes: SM-count portability (= #89), tool-message media parts (in-tree via #57/#65); also rides #10 tolerant tool-call parser + boot watchdog + #84 Windows.
- **#91** (x-n2o) home-agnostic model paths — CLOSED 2026-08-25 without merge. Skip.
- **#84** (devan-carlin) native Windows MSVC+CUDA — not lane-relevant. Skip.
- #83/#82 superseded by #85/#84 (#85 ported as `988d72e8`).

Forks pushed 08-23 → 08-25 (vs `gevil/master` `4302983a`):
- **knoopx** (58 ahead of upstream master, active line): 08-25 `b9a27541` program as resource authority + `4eef14a7` ownership transfer after alias eviction; 08-24 `e0829866` restore anonymous prefix reuse + `fc5c4834` vision workspace reclaim + `c80b2bf2` optional failed timing observation; 08-22 `5fb9a138` nix flake (CUDA 13); 08-18 `665e35fb` MTP adaptive draft depth + `99fbebec` W8G32/FP8 dual-artifact binder. Deep divergence (conflicts vs `tier5`) — **watch**, cherry-pick individually if valued.
- **MirkoCovizzi** (4 ahead): `7d566547` fix(mtp): greedy verification width-invariant (08-23; ~25-file ops diff, moderate conflicts with our GQA/MoE files) + `3eb6193b` .gitignore. MTP fix = next-wave candidate.
- **dylan/experimental**: `d8cb420f` dflash2 nvfp4 (bf16 codebooks, tree verify, C=3 identity; 08-25, not in the dflash2 audit branch) + 08-23 `a37580f2` TMA S3 prefill, `2660be68` GEMV L2 evict_first. Next perf-wave candidates.
- **natpate** (diverged): `b686696e` array content on OpenAI tool messages (08-24) — redundant with in-tree #57/#65. Skip.
- **cometkim**: 08-25 18:41 push = `cometkim/dev` "roadmap updates" (docs); `feat/qwen3.8-nvfp4full` (`1455676b`) unchanged → no lane/weights impact. New 08-23 branches: `feat/1m-context`, `feat/build-speed`, `feat/hyperquant` (refuted).
- taylor-shift / Aoyagi-29: nothing in the window. JCraigWasTaken: fork inaccessible (private/renamed).

Re-verified in-tree this window (all on `tier4`/`tier5` + `gevil/master`): #85 `988d72e8`, #86 `e4beff22`, #87→C6 `a568115b`, #88 `6da2efef`, #65 follow-up `102ab113`, empty-think `16f405d4`+`1ffbc388`. #73 zero-drift vs housekeeping head `0ede955f`.

## Tier 6 (2026-08-25) — portability ops fix

**Wave D — portability (no-op on the 170-SM lane; device-portable going forward):**

| Item | Source | What | Local |
|---|---|---|---|
| #89 SM-count persistent grids | igorls upstream PR #89 `f52e2099` (cherry-picked as `808bd1d1`) | persistent MoE/GDN prefill grids sized from the active device's SM count (new `ops/common/device_info`, `cudaDevAttrMultiProcessorCount`, fallback 170) instead of the hardcoded `kRtx5090SmCount = 170`; on the 5090 lane `device_sm_count()` = 170, so behavior is unchanged here — the value is portability to other GPU SM counts. Conflict resolved at the Q4 gate_up launch site (kept our route-job kernel args, adopted the runtime `prefill_persistent_blocks`) | `808bd1d1` |

**Deferred from this wave (recorded 2026-08-25):** the MirkoCovizzi MTP width-invariant fix (`7d566547`) — deep merge conflict with our perf work (staged-smem GQA reduce, cooperative GDN, T=4 swiglu, fused gate, and a bf16-vs-FP32 Int8-partial divergence). Plan: dedicated wave gated on the MTP greedy-parity test (first confirm our tree has the width-variance bug, then merge).

Public review: [Gevil/ninfer PR #7](https://github.com/Gevil/ninfer/pull/7) (`tier6` -> `qwen3.8-nvfp4full`, stacked on PR #6).
## Quasar adoption (2026-08-26)
- Lane live on `quasar-303dbcaa` (image ninfer-nvfp4:latest, QUASAR artifact
  /home/gevil/.local/share/ninfer/models/qwen3.8-27b-quasar/qwen3_8_27b_quasar.ninfer, quadlet swapped, model-id qwen3.8-27b-quasar).
- Battery: 9 PASS / 0 FAIL; decode probes within -5% of the pre-ship
  baseline (/home/gevil/.local/share/ninfer/logs/quasar-baseline-2026-08-26.json).
- Rollback: restore /home/gevil/.config/containers/systemd/ninfer-nvfp4.container.nvfp4full.bak + retag ninfer-nvfp4:latest from
  ninfer-nvfp4:tier6-pre-quasar + systemctl --user restart ninfer-nvfp4.service.
## Audit 2026-08-26 - 24h fork re-audit + Tier 8–10 proposal (MERGES PARKED)

**Decision (2026-08-26, user):** park all further merges until the Tier 8 probe runs. No new tier branches; this section is the standing proposal.

**Fork state (vs `upstream/master` `feaf4dd0`, all remotes fetched 2026-08-26):**

| Fork | ahead/behind | Last-24h activity |
|---|---|---|
| eason/master | 0 / 0 | none — **fully upstreamed, dead source** (work already in our tree via master) |
| md/master | 0 / 0 | none — fully upstreamed, dead source |
| cometkim/cometkim/dev | 108 / 6 | DFlash2 engine integration (end-to-end, 4 gather bugs fixed), draft ops, eval benches, 768k/786k presets |
| dylan/experimental | 59 / 34 | 4 XAttention prefill commits (on top of `d8cb420f`) |
| dylan/perf/rtx5090-qwen38 | 13 / 35 | none in window |
| upstream/dev (Neroued) | 58 ahead of master | runtime-ownership refactor + cache/serve fixes — **in-flight dev branch, not the default branch** |

**Merged status (verified in-tree, 2026-08-26):** our PRs #1–#8 all MERGED; **#8 (tier7) merged then REVERTED** (50% decode regression at 98k — not accepted; `tier7` branch survives, resume needs the decode fast-path fix). `master` = tier6 + quasar (`37bc977f`); mainline `qwen3.8-nvfp4full` = tier6 (`1b9aef3b`); lane live on `quasar-303dbcaa`. Upstream PRs cherry-picked and in-tree: #54 #55 #57 #61 #65 #67 #69 #79 #85 #86 #87 #88 #89 (`808bd1d1`). Upstream **master fully contained (0 behind)**; **`upstream/dev` NOT contained (58 behind)** — that is the "dev convergence" item below; we miss nothing stable.

### Tier 8 — cometkim DFlash2 — PROBE-GATED (no merge)

DFlash2 = separate draft model (DFlash2DraftModel: 5 layers, hidden 5120, ~1.6B params, ~1 GB+ in NVFP4, from the incoai BF16 checkpoint) that proposes 8-token blocks the target verifies. **nvfp4full v2 = our tier1–6 nvfp4full backbone byte-identical (same inputs, same encode path) + the drafter module embedded as a REQUIRED member** (`DFLASH2_REQUIRED`, config pinned: block_size 8, conv 2/16, selector rank 256/top-16, capture from target layers 5/19/33/47/61). Re-conversion cost ≈ encoding the small module only.

| Pick (oldest→newest, vs `1b9aef3b`) | What | Merge-tree conflict |
|---|---|---|
| `71c934e0` | width-8 hq block verify + MTP K≤7 plumbing | 8 hunks — gqa_attention_decode.cu/.h, gqa_attention.cpp, gqa_attention_decode_bf16.cuh |
| `8ac76cbf` | nvfp4full v2 — DFlash2 module quantized to NVFP4 | 10 — bindings.cpp, convert/inventory/verify nvfp4full scripts, artifact doc |
| `1375ae93` | draft primitives — two-tap dyn conv + selector scores | clean |
| `b7bae1bf` | top-k selection + selector path walk (device) | 1 — tests/CMakeLists.txt |
| `7900c2e1` | engine integration — executes end-to-end | 8 — bindings.h/.cpp, variant.h, config.h, layouts_impl.h, serve_options.cpp, cli/options.cpp, tests/CMakeLists |

Skip `0948f662`/`db3797d8`/`973c5b3c` (add/revert/restore artifact-ride; net-zero — tree diff empty). Series total: 90 files, +4055/−593. Ships 4 new tests (`test_dflash2_{dynamic_conv,selector_predecessors,selector_scores,topk_walk}`) + edits to test_load_plan/test_nvfp4_a16/test_cast.

**Why not merge yet:** tier7 precedent — MTP is the *cheaper* draft (native head) and at 71% acceptance still regressed decode 142→55–71 tok/s at 98k (reverted by user); the root cause (decode fast-path cost) is still open; DFlash2 drafts via a full second-model forward pass with the same context-scaled verify cost — a strictly harder case. Cometkim's "acceptance survives long context" gate was measured on llama.cpp, not this kernel path. Also implies a quasar → nvfp4full-v2 profile switch on the live lane.
**Why not shelve:** spec decode is the decode lever if net-positive; picks are clean (29 localized hunks); artifact cost low; a decisive A/B is hours, not a project.
**Gate to merge:** free-GPU buildstage — build stage with the 5 picks + v2 artifact, decode battery at 98k/225k vs tier6 baseline (141.7 tok/s). Merge only if net decode beats baseline. In-house alternative on the same bottleneck: resume tier7 with the fast-path fix (known scope, no fork dep, no artifact change).

### Tier 9 — dylan XAttention prefill — DEFERRED (DFlash2 decision first)

| Pick (vs `1b9aef3b`) | What | Merge-tree conflict |
|---|---|---|
| `d8cb420f` (prereq carrier) | dflash2 nvfp4 codebook + sage/sparge + NVFP4 attn kernels (gqa_attention_prefill_nvfp4.cuh, gqa_kv_compact.cuh) + KV-compact path — 108 files, +6410/−736 | 81 — convert_nvfp4.py×9, layouts_impl×6, dflash_impl×6, gqa×8, GDN×6… |
| `3b730ee8` | exact-NVFP4 XAttention prefill (paper inverse-stride, packed-K) | 82 — test_gqa×20, decoder_state×10, gqa_prefill×7… |
| `55dde21c` | local-tile keep, min_len dense skip, workspace scratch | 18 |
| `a56c3a77` | stop per-page K-centering on sage NVFP4 fill | 3 |
| `1e9c5dda` | paper B=128 keep-set + tensor-core score GEMM | 2 |

Total ~168 files, ~186 conflict hunks. **Not separable:** the xattn kernels + sage KV-compact path exist only in `d8cb420f` (absent from our tree). The 28 "dflash" refs in the xattn diff are the existing upstream `--spec mtp|dflash` plumbing (fine). `d8cb420f` is dylan's **competing DFlash2 variant** — merging it adopts a DFlash2 implementation before the Tier 8 probe decides. Conflict density (82 hunks in one commit) sits in core attention/KV/GDN — exactly the files our tiers edited.
**Value:** prefill/TTFT — a real orthogonal lever for long-context agent workloads; tensor-core score GEMM fits sm_120a.
**Revisit when:** DFlash2 implementation decided (if dylan's line wins, xattn rides on it; if cometkim's or neither, re-baseline or drop) + prefill/TTFT measured as an actual bottleneck. Gate if pursued: their +1257-line GQA suite + PPL parity + prefill/TTFT battery + free-GPU ctest.

### Tier 10 — upstream/dev sync — WATCH (converge when dev lands on master)

`upstream/dev` is a **separate non-default development branch** (58 commits ahead of master, 0 behind): runtime-ownership refactor ("program = resource authority", ownership transfer after alias eviction), context-cache scheduling + correctness (retain prefixes across rewritten suffixes, keep warmup out of context cache), serve/vision perf, fp8/int8 KV, paged-KV physical containers, TTFT campaign. We are **0 behind `upstream/master`** — nothing stable is missing; dev is in-flight work not yet on master.
Conflict surface if merged today (merge-base = `feaf4dd0`): **100 hunks / ~30 files** — program_impl.h(13), text_context_impl.h(7), frontend.cpp(7), program.h/chat_template.cpp/types.h/test_cli_options/test_frontend(4 each), api_impl.h/serve_options.cpp/request_log.cpp/generation_service.h(3 each), ~20 files at 1–2.
**Why watch, not merge now:** in-flight moving target (5 commits in 48h); the cache-correctness fixes sit on the refactor and cannot be cherry-picked alone; with T8 probe-gated and T9 deferred, the "shrink their conflict surface" argument is secondary; 0 behind master means nothing is lost by waiting.
**Trigger to execute:** dev merges to upstream master → full `git merge --no-commit --no-ff` (no cherry-pick batch), per-file resolution, build + free-GPU host-KV ctest + full battery re-baseline, `tier10` branch + stacked PR + record here.

### Hold / do-not-merge (2026-08-26)

| Item | What | Why not |
|---|---|---|
| upstream #90 (igorls, DRAFT, upd 08-26) | cross-request prefix seeding, content-addressed seed store (31 commits) | competing design vs shipped #73 host content cache; upstream discussion unsettled |
| cometkim 768k presets + `b18aeab4` (TMA stream) + `b7bf6e97` (gqa U8 workspace) | long-context presets; fixes only bite at 524k–786k | lane runs 225k on 32 GB; latent until a context-extension decision |
| cometkim eval harness (LBv2/GPQA/campaign8) | quality-validation tooling | not lane code — pull in to validate T8/T9 when they run |
| eason, md | 0/0 vs master | fully upstreamed — dead sources |
| cometkim windows-port / 1m-context / hyperquant | off-lane | Windows, 1M context, refuted |
| knoop fork | deeply divergent | its "program authority" line is upstreaming via dev — arrives with Tier 10 |
| matpape tool-message array content | OpenAI tool-message array content | redundant with in-tree #57/#65 |
| dev fp8/int8 KV, paged physical containers | new KV formats | not standalone-adoptable; arrive with the dev→master merge |

### Sequencing (once un-parked)

1. **Tier 8 probe** (free-GPU A/B vs 141.7 tok/s tier6 baseline) — the only decision the plan hinges on.
2. **tier7 fast-path fix** — in-house fallback on the same spec-decode bottleneck.
3. **Watch:** dev → master (Tier 10 trigger), #90 settlement, dylan/cometkim DFlash2 evolution.
4. **Tier 9** only after the DFlash2 decision + prefill-bottleneck evidence.

## T11 - quasar re-verification on the T10 upstream merge (2026-08-30)
- Branch `t11-quasar-merged` @ 02d7b041 = T10 merge (upstream ce09aee5 into
  qwen3.8-nvfp4full @ 1b9aef3b, 114 fix commits) + the 5 quasar commits
  (c6da81ff..303dbcaa) re-applied. Conflict in package.cpp resolve_weights
  restored the nvfp4full + quasar-nvfp4 profile cases dropped by T10.
- Free-GPU full ctest (buildstage-merge, GPU passthrough): rc=0.
- Battery (quasar-restart-verify): 9 PASS / 0 FAIL; decode
  probes within -5% of the 08-26 quasar baseline (/home/gevil/.local/share/ninfer/logs/quasar-baseline-2026-08-26.json).
- Lane live on image ninfer-nvfp4:t11-02d7b041 (tags :quasar, :latest).
- Rollback: retag the previous :quasar image (61c74d6b6e9d40b6c8f26f51c8f5ba9720dc9d72e7805838634ff15a3e54b9fd) to
  ninfer-nvfp4:quasar + :latest, systemctl --user restart ninfer-nvfp4.service.

## T12 - upstream convergence wave (2026-08-30)
- Branch `t12-upstream-merged` @ 4567363e = T11w line (185decf5) + merge of upstream
  master ce09aee5..d9dbe1ce (16 commits; upstream dev merged into master). New:
  value-aware shared prefix scheduling (099d9032 + 4ce08e34 bench + 6b94b8c5
  checkpoint-pressure fix), w8 rowsplit eight-code decode (93cdc264), fused TMA
  SwiGLU width registration (00369f63 + 5327d676 test), nvtx engine profiling
  (89e297e1), perplexity evaluator + corpus + logprob reduction (f6b3ba93/
  11e76d8d/0b5b7c9a), artifact-decoupling docs (953a7b2f), four removal chores
  (python reference stack af0bd2d0, v1 artifact migration 6143271f, obsolete
  benches 68324a97, peripheral tooling d9dbe1ce).
- t12->quasar merge: clean (9-file conflict resolution on the quasar branch:
  frontend.cpp/frontend.h upstream marker machinery + kept P1 reasoning
  derivation; package.cpp x2 upstream call form; w8_rowsplit upstream
  eight-code optimization; types.h additive).
- t12->master merge: 1 conflict (sparse_moe_prefill_kernels.cu — master kept
  the local q4_decode_eight_bf16 byte_perm decode + old q4 kernel signature;
  t12/upstream superseded it with the grouped_io redesign) — resolved to t12,
  file byte-identical to the verified t12 tree.
- Verified on the lane: consolidated battery 16/16 (UP, IMAGE, MODELS, LEDGER,
  WARMUP, VISION, VISION-HIST, VISION-POISONED, REPLAY x2, XHIGH, THINK-SMOKE,
  DECODE-FRESH, DECODE-8K, QUALITY, SOAK, 4XX-WATCH) on image d85a4163, plus
  free-GPU ctest recheck 100 tests rc=0 (skip set within baseline, incl. new
  baseline entry ninfer_qwen3_6_27b_score_real_test).
- Propagated to master via no-ff merge; PR #9 (t11-quasar-merged) closed as
  superseded — its content is fully contained in this merge.

## T11 rollback (2026-08-30) - the "PASS" record above was incomplete
The t11 image (`t11-02d7b041`, d6735a08) shipped green on the T11 gate
(9/9 text-only battery verdicts + free-GPU ctest rc=0) but is functionally
broken for vision requests: any chat completion carrying image parts
returns `400 chat media count does not match rendered placeholders`.
The OMP harness re-sends full conversation history including images, so
every session with image history 400'd on every turn (sessions became
permanently unusable until rollback).

Root cause (T10 upstream merge, new frontend contract):
`JinjaTemplate::render` (src/targets/qwen3_6/impl/frontend/chat_template.cpp)
returns a `RenderedChat` with only `.text` set - `media_placeholders` is
never recorded for jinja-template renders (the C++ clone path records
them via `RenderBuilder::append_media_placeholder`). The new
`expand_placeholders` (processor.cpp:547) enforces
`rendered.media_placeholders.size() == items.size()` -> 0 != N -> 400 for
every image/video request on the bind-mounted Sharp template. The marker
strings match (`<|⁠vision_start|><|⁠image_pad|><|⁠vision_end|>` in the
template and in `kImagePad`); the jinja path simply never records them.
The pre-merge tree has no such check (verified: error string absent at
0f61e1db).

Rollback (executed by the user, verified):
- Running image restored to `61c74d6b` (previous quasar, tag
  quasar-303dbcaa); `:quasar` + `:latest` retagged to it; lane active,
  model `qwen3.8-27b-quasar` loaded.
- Verified on the old image: vision probe 200 ("Red" for a red square);
  replay of the exact captured 313-message/5-image failing request:
  HTTP 200, 51.5s, full reasoning stream.

Fix path (next wave, on top of t11-quasar-merged): make the jinja render
path record media placeholders from the rendered text (scan
`<|⁠vision_start|><|⁠image_pad|>` / `<|⁠video_pad|>` runs), gated by a
vision battery probe. PR #9 (t11-quasar-merged -> quasar-nvfp4) must NOT
be merged until that fix lands.

Shipping-gate re-evaluation (post-mortem):
1. Battery was text-only for a `--vision` profile - the failing
   invariant only fires with `items.size() > 0`; text probes cannot see
   it.
2. No serve-level integration coverage of the chat endpoint through the
   jinja template + media path (ctest is engine-level).
3. Verdict logic equated "all probes green" with "lane healthy" - no
   4xx watch, no consumer-shape probes (long history + image in an old
   message, streaming, tools, chat_template_kwargs).
4. Session coupling: the ship stopped the lane serving the live OMP
   sessions and the FAIL path (retag + restart) only triggers when the
   battery fails - the battery passed, so no automatic recovery existed;
   rollback was only reachable from a new session.

## Audit 2026-08-30 - post-T12 re-evaluation (adoption queue)

**Base state (verified 2026-08-30):** master @ 1019ef7f = T12 tree, **0 behind upstream/master**
(tip `d9dbe1ce`; upstream merged its dev branch into master 08-29 — the dev line is gone,
Tier 10 watch item resolved by T12). Live lane: image t12-4567363e (d85a4163, tags
:quasar/:latest), battery 16/16 PASS (3 distinct vision probes + REPLAY 10/10 + SOAK 5/5
incl. vision x2 + 4XX-WATCH; decode 141.6/147.3 tok/s in gate), fresh live vision probe 200.
Ship gate: both T11-vision-400 script bugs fixed and codified (battery rule R1: Python
probes build payloads via json.dumps; ship rule G4: ctest builds targets, cd's to the build
dir, rc!=0 or TOTAL<=50 = FAIL).

**What changed since the 08-26 audit:**

- **Tier 10 (dev sync) — RESOLVED.** dev merged into upstream master 08-29; T12 consumed it.
  The knoop "program authority" line arrived in-tree with T12.
- **#90 (igorls prefix seeding) — CLOSED UNMERGED 08-27**, no maintainer feedback. The
  competing-design question is settled: in-tree #73 host content cache is the surviving
  design. #73 itself was closed unmerged upstream (merge state DIRTY) — stays fork-only.
- **#74/#75 are closed design-doc issues, not PRs.** #75 (host-KV convergence: #64 park
  fast-path over the #73 content store) is design-only — no unified PR exists; watch for
  one (it would change the host-KV lane-ops item below).
- **Upstream merged a large PR batch 08-27..08-29** (md perf #99/#106/#112/#113/#115 =
  gevil copies 92bb06eb/8d343527/00369f63/93cdc264/5327d676; protocol-adapter completion
  superseded #55/#108) — all in our tip, nothing to adopt.
- **cometkim DFlash2 line REWRITTEN 08-27**: the five 08-26 audit SHAs are dangling;
  re-pushed as 9f36497c/74dc35fc/e5c33792/9101d425/0a7bef84/2470a6d8, plus NEW 71677e35
  (context norm gains plain w — native acceptance reproduced at 3.4-3.7 tok/round),
  folded into dev via db9a9051 (08-28) with f92234ca (width-8 int8 verify tile — the
  single-pass DFlash2/MTP7 verify on the int8 lane, i.e. our lane's KV dtype) + f6bb4f10
  (serving presets). No PRs exist on cometkim/ninfer or upstream from cometkim.
- **cometkim feat/kernel-perf: 14 commits** (08-27, not in-tree): fp16-accumulate PV i8
  prefill, key-range splits (+14.2% pp65536 i8 / +6.0% hq), PDL decode chain, fused
  GDN decode route, per-request error boundary, 524k/786k prefill fixes. Folded into the
  same dev squash; dev is 111 commits behind upstream (has not absorbed d9dbe1ce).
- **dylan: the only fork with activity today (08-30 07:08).** experimental moved
  13a2e5d0 → 582431c0: KV-RAM/DFlash entitlement poison fixes (13a2e5d0), decode-band
  Linear pinning (2c295b6e), dflash2 paper-chain verify (eb5bedb5), HTTP timing parity
  (582431c0). The xattn/prefill pick list from the 08-26 audit needs re-derivation.
- **md / eason: dead sources.** md ships via upstream PRs (all 5 merged = in-tree);
  eason dead since 08-22.

**Adoption queue (ordered):**

1. **Tier 8 DFlash2 probe — the hinge, rebased pick set, quasar-v2 artifact (user decision
   2026-08-30: probe rides on QUASAR, not nvfp4full-v2 — no lane profile switch).** The
   08-26 pick list (71c934e0 etc.) is dangling; use the rewritten line: 9f36497c (width-8 hq
   verify + MTP K≤7) + 9101d425 (draft primitives) + 0a7bef84 (top-k walk) + 2470a6d8 (engine
   integration) + 71677e35 (norm-gain fix) + f92234ca (int8 verify tile — the single-pass
   DFlash2/MTP7 verify on the int8 lane = our exact KV dtype). 74dc35fc (module rides
   nvfp4full artifact) + e5c33792 (nvfp4full v2 quantization) are profile-specific: adapt for
   QUASAR instead — extend the in-tree quasar recipe (tools/convert/qwen3_8_27b/
   drafter checkpoint — **repo confirmed 2026-08-30: incoai/Qwen3.8-27B-DFlash2**
   (mirror: z-lab/Qwen3.8-27B-DFlash2; 3.58 GiB model.safetensors, 81 BF16 tensors,
   5 layers, selector codebooks [248320,256], fc [5120,25600] = 5 taps; downloaded to
   ~/.local/share/ninfer/models/qwen3.8-27b-dflash2/) per
   cometkim's recipe (docs/maintainer/qwen3.8-27b-artifact.md §15). **Module format
   (2026-08-30, user question):** port BOTH 74dc35fc (module rides the artifact) AND
   e5c33792 (nvfp4full-v2: the module's 34 weight matrices quantized with the fork's own
   NVFP4_MAXABS_DIVISOR_RNE_V1 encoder — weight-only, no calibration; norms + conv base
   kernels stay BF16): module 3.85 GiB BF16 -> ~1.3 GiB NVFP4, ~2.55 GiB freed per held
   envelope. The v1-style BF16 module (3.85 GiB resident) would NOT fit the quasar lane's
   ~2.36 GiB free-after-startup slack without shrinking the KV pool; the NVFP4 module is
   the viable path. Community NVFP4 drafts (maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal
   modelopt 1.55 GiB; YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4 compressed-tensors W4A16
   group-16 1.36 GiB) are parity references only — the engine loads the module from the
   .ninfer artifact in the fork's NVFP4 encoding, so our converter quantizes the BF16
   incoai repo either way. Gate: free-GPU buildstage + quasar-v2 artifact (NVFP4 module),
   boot ledger (module resident ~1.3 GiB NVFP4 vs ~2.36 GiB free-after-startup slack —
   fits without touching the KV pool), decode battery at 98k/225k gated vs the quasar
   baseline JSON (-5%), full battery incl. vision; merge only if net decode beats baseline.
   The probe was NEVER run (08-25 pass-1 69.2 tok/s was the pre-DFlash2 staged-smem
   attempt). If it passes the live lane ships quasar-v2 (same profile family). In-house
   fallback on failure: resume tier7 with the decode fast-path fix (known scope, no fork dep).
   **Port status (2026-08-30):** the mechanical cherry-pick is BLOCKED by the T10/T12
   restructure — the picks' gqa decode files (gqa_attention_decode_*.cuh/.cu,
   gqa_attention.h/.cpp) no longer exist in master; decode moved to
   src/ops/softmax_attention/dense/causal_cache/ (small_t*.cuh/cu, launch.h,
   causal_softmax_attention.cpp). T8 is therefore a PORT, not a pick. File map so far:
   pick 9f36497c (width-8 hq verify: TokenTile assert small_t_bf16.cuh:26 <=6 -> <=8,
   new gqa_small_t_chunk_tokens(DType) U8->8 else 6, uses_small_t/resolve_route gain a
   DType param) -> small_t_bf16.cuh + launch.h:30-32 + small_t.cu:228 +
   causal_softmax_attention.cpp; pick f92234ca (int8 tile) -> small_t_i8.cuh:88 + same
   routing files; picks 9101d425/0a7bef84 (dflash2 primitives/top-k) are ~all-new files
   (dflash2_draft.{cuh,cu,h}, dflash2_draft.cpp, dflash2_selector_*, tests) + CMakeLists
   (conflicts likely); pick 2470a6d8 (engine integration) touches cast.{h,cuh,cu,cpp},
   bidirectional_gqa_attention.cuh, swa.{h,cu}, nvfp4 config/dispatch/gemv/small_t —
   per-file mapping pending. Branch t8-dflash2 created at master.
   **Ginfer audit (08-30, /tmp/ginfer-audit clone):** gadflyii/ginfer is a
   heavily rewritten ninfer fork — artifact v3 `.ginfer` contract (producer-
   final version-3), runtime/artifact/KV rewritten; engine code NOT portable
   to our tree (different artifact contract + runtime); value = measurements
   + design, not picks. Same z-lab draft confirmed (81 tensors, 5 layers
   SWA-2048, taps [5,19,3,47,611], hidden 5120, rank 256, vocab 248320).
   Their production draft storage is **Q4G64 (int4 group-64), not NVFP4**:
   5090 measurement NVFP4 backbone + Q4 draft 154.4 tok/s > MTP3 143.8 > W8
   draft 142.0 (+7.4% over MTP3); Q4 companion 0.95 GiB (nvfp4-dflash2-q4
   artifact 22.5 GiB total). Our T12 tree already has Q4G64_F16S/W8G32_F16S
   numeric formats + the groupwise converter lineage, so a Q4G64 draft
   companion is feasible — add as a second probe variant after the NVFP4-
   module probe if it fits the same artifact window. Acceptance expectations
   (their engine, same draft on Qwen3.8): 2.4-2.73 tok/round at k=4 (36-43%);
   their MTP3 2.40 (46.52%) — our probe settles it on our engine. Port
   candidates beyond the draft: (i) adaptive context-copy verification —
   request-local 6-12-token suffix match reuses the earlier occurrence's
   continuation as point-mass proposals, expanded to 15 proposals after two
   copy rounds; lossless (target verifies exact tokens); engine logic, not
   kernel — strong fit for agentic/repetitive sessions; (ii) compact Q4
   kernel schedules for sm_120a (34-47% op speedup on exact Qwen DFlash2
   geometry) — only needed if the Q4 variant is chosen. "Generating our own
   draft" = training (the z-lab draft is learned; no training pipeline in the
   repo) — the practical path stays re-quantizing z-lab's draft via our
   converter. The doc's DFlash2 section independently corroborates the
   quantized-draft VRAM math (community: 3.53 -> 1.37 GiB draft = +44% KV
   context at equal acceptance) and the converter landscape (taylor-shift
   `--dflash-model` = proven route; agwosdz convert_dflash2.py; no unified
   converter anywhere — our in-tree recipe is the gap-filler).
   **taylor-shift/ninferno audit (08-30, user point, /tmp/taylorshift-audit,
   branch dflash2-27b):** upstream feaf4dd (T10 start) + 103 commits, 105
   behind upstream master; last commit 08-25 (dead since). Complete
   DFlash2-27B package: engine (afd1bcd, +4983: grouped_dynamic_causal_conv
   + dflash_selector family (full-vocab top-16, 256-rank projection, 16x16
   edge scores), 5-layer DFlashConfig, 66-object dflash/ load section, 4
   conv insertions), converter (tools/convert/qwen3_8_27b/dflash2.py: 81 HF
   tensors -> 66 objects, 2,226,792,960 B; convert_nvfp4.py --dflash-model),
   tests (op units, linear dispatch, 27B load-plan, engine real = golden +
   D4 negative + determinism + long-restore, python converter), env-gated
   diagnostic matrix (k=1..7 sweep + MTP probe + token streams), 5-stage
   acceptance script, PyTorch reference drafter (vllm PR 52816 port,
   stage-by-stage cross-check of engine [dflash.trace]), published
   nvfp4-dflash2 image + the phaseonx11 artifact. **Module format =
   W8G32_F16S (2.07 GiB payload, 21 draft GEMMs), NOT NVFP4.** **CRITICAL:
   unresolved acceptance bug at branch tip — engine accepts ~1.1
   drafts/step vs published ~4.8** while every op passes its unit tests;
   tip commits (08-25) are the debugging stage (reference drafter, round
   dumps, per-layer checksums, W8-dequant-compare: the W8 draft GEMMs vs
   the BF16 references is the one structural difference — a bad
   scale/stride/row mapping degrades every draft while leaving the target
   correct). Their published DFlash numbers are 35B-A3B only (upstream-
   origin: 4.0-5.6 tok/round; DFlash beats MTP3 on long-reasoning/
   structured but -14%..-43% on code/story) — NO 27B-8 DFlash2 numbers.
   **Implications for the T8 probe:** (1) draft-weight quantization
   fidelity is a LIVE acceptance-rate risk: the W8 hypothesis is exactly
   the class our NVFP4 module (4-bit, more aggressive than W8) faces —
   gate addition: module-fidelity check (port their dequant-compare:
   dequantize each module object, compare vs the raw checkpoint tensor,
   tolerance-gated) must PASS before the quasar-v2 artifact ships; (2)
   port the PyTorch reference drafter as probe oracle (engine trace vs
   reference, one propose block, stage-by-stage); (3) acceptance gate:
   probe must beat MTP3 (~2.40 tok/round) — an acceptance collapse
   surfaces as net-decode loss in the 98k/225k battery; (4) module
   fallback chain if NVFP4 fails on acceptance: W8G32 module (2.07 GiB —
   FITS the ~2.36 GiB slack, unlike the 3.85 GiB BF16) -> KV-pool shrink
   -> BF16. Cross-checks: their 34ba938 (kv_cache_append_prefix per-cache
   window) independently corroborates our pick-5 DU change; their a2ba70f
   (selector walk IMC: sampling config by value, not host pointer) is a
   bug class to audit in our ported walk; their 5713946/bf24792 (drafter
   shared-KV entitlement gating, hybrid vs pure-SWA) + 7f84d5a (masked
   verify blocks off the Prompt prefill path) + aeba589 (oversized
   prefill chunks) = equivalence checks during pick 5/6. **Decision: NOT
   the T8 engine source** (tip broken on acceptance, 105 behind master,
   W8 module, 60+ fleet-deploy commits of noise) — the dylan line stays
   (active 08-30, norm-gain fix, int8 verify tile); taylor-shift = the
   verification asset + the W8 fallback-format source. **Port state
   (08-30):** 4/8 picks committed + gate-verified (8b113373, aa6acabf,
   f7f3acbe, 2cc5292c); pick 5 (2470a6d8, 57 files / 2125 ins, 21
   conflicts) was in progress (7 easy resolutions + 5 new files staged,
   8 hard qwen3_6-runtime conflicts) — an accidental `git checkout
   t8-dflash2 -- .` wiped the staged resolutions to existing files (the 5
   new files survived); recovery = cherry-pick --abort + fresh re-pick,
   then re-resolve.
2. **T13 upstream wave — after the T8 decision** (#107 touches the same qwen3_6_27b
   package.cpp the DFlash2 v2 artifact edits; do not interleave):
   - #107 (nvfp4 wire-format profile detection) — our quasar artifact IS the W8+NVFP4
     community layout; auto-detection unblocks the official-vs-community artifact call.
   - #97 (container build ccache) — build-only; verify against our customized Dockerfile.
   - #72 (on-demand vision residency, DRAFT but complete, measured on RTX 5090) — the
     vision-lane VRAM/context lever; sequence last in the wave, full battery incl.
     vision probes.
3. **Lane ops: host-KV + template (COMPLETE 2026-08-30 16:45).** Finding:
   `--kv-host-cache-mib` no longer exists in the engine — the #73 content-addressed
   host-KV subsystem was dropped in the T10/T12 upstream merge (fix(merge) 944a9be:
   "caller was the dropped #73 host-KV path"; 43298ee3 dropped the stale real-test).
   The 2026-08-24 quadlet comment ("--kv-host-cache-mib 32768 added") was wrong: the
   flag was never in the Exec. The lane runs the built-in `--host-kv-mib` #64 parking
   arena, 8 GiB default (`kDefaultHostKvCapacityBytes`, include/ninfer/types.h;
   ledger: `host-state=8 host-kv=8.00 GiB`).
   Sequence: 08-26 enabled `--host-kv-mib 32768` + Sharp v22.4.0 → battery FAIL:
   DECODE 119.6/116.8 tok/s (~15% below gate) + 34 engine rejections (503) during
   soak → auto-rollback. 08-30 16:07 re-ran the battery on the SAME config (32 GiB
   pin + v22.3.2): DECODE 119.6/116.8 FAIL again + 34 rejections + SOAK 4xx FAIL →
   rollback. 16:34 re-shipped **default 8 GiB + v22.4.0** (the 16384 flag append
   silently failed — sed pattern missed the Exec line — and the default proved
   sufficient, so it stays off).
   Note: both of today's "A/B" runs actually ran the 32 GiB pin — the v2 job's
   rollback restored the 08-26 quadlet (`--host-kv-mib 32768`, the #64 parking
   arena flag; `--kv-host-cache-mib` is the dropped #73 name and matches nothing),
   and the 16384 sed targeted the wrong flag name. So the A/B is clean on the pin
   axis: pin-on (both battery runs, v22.4.0) -> decode FAIL; pin-off (cold,
   v22.4.0) -> decode PASS 160.9/138.7. The pin is the regression variable, not
   the template.
   **Cold verification (lane idle during the probes):** decode-fresh 160.9 tok/s
   (gate ≥132.4), decode-8k 138.7 tok/s (gate ≥130.5) — fastest measurements of the
   day; soak 5/5 200; 313-msg/13-media fixture replay http=200 wall=76.6s; live
   session decode 122–160 tok/s with `reuse=private_response_replay`.
   **Diagnosis (two confounds, both now documented):**
   (a) The 32 GiB pin correlated with the decode regression in both battery runs —
   pinning 24 GiB above the 8 GiB default at boot (MemFree 26→1.8 GiB); consistent
   with reclaim pressure on the pinned arena's extent copies.
   (b) Both DECODE FAIL windows overlapped the live OMP session generating against
   the same lane (178k-context xhigh stream, C=4 pool shared with the probe) — the
   probe stream was crushed (3.6 tok/s in the worst window). **Battery design rule
   (new, R2): DECODE/SOAK gates are only valid in a quiet window (no concurrent
   session generation against the lane-under-test); a concurrent session is the
   T11-postmortem session-coupling class, now hitting the battery. Cold probes in a
   tool-call window are the valid measurement.**
   **Final state (shipping):** image t12-4567363e + Sharp template v22.4.0 +
   default 8 GiB `--host-kv-mib` (no pin). The 32 GiB pin is NOT re-shipped: no
   measured benefit, and both pin runs regressed decode under the gate. Revisit only
   if session-parking demand appears (then pin at ≤16 GiB with a quiet-window A/B).
   v22.4.0 is SHIPPED (multi-part reasoning + default_reasoning_effort + whitespace
   fixes; vision markers byte-unchanged).
4. **Tier 14 host-KV content cache — re-port the #73 subsystem onto the T12 tree**
   (new tier, 2026-08-30). The built-in `--host-kv-mib` arena (page-granular
   HostKVArena + HostKVExtentStore, program_impl.h) pins host RAM for KV parking but
   is NOT content-addressed: no prefix dedup, no 0.2s branch restore, no #74
   identical-burst coalescing. The tier4-wave #73 port (9 commits: 7fa78bec core,
   0d45cfce flag conflicts, b5ecbd1b stale-plan revalidation, a5202b29 store
   invariants, 258bd89b admission replan, 57d52d5e anchor-bytes+epoch gate,
   e30d0843 coalescing, ad89f3d3+beb2c7ba tests) was dropped by the T10 merge —
   re-apply it on master. Conflicts expected in runtime/program (the T12
   runtime-ownership refactor restructured the pool APIs that 944a9be orphaned),
   serve_options (flag + ContextCacheOptions fields), and CMake. Upstream status:
   #73 closed unmerged (DIRTY), #90 closed unmerged, #75 design-only (no unifying
   PR) — this stays fork-only. Gate: free-GPU ctest + full battery + a new probe:
   two sessions sharing a ~76k-token prefix, switch latency target <2s vs the
   18-40s re-prefill at 8 GiB. Budget: start at 16384 MiB (two ~200k-token sessions);
   the 32768 pin measured MemFree 1.8 GiB and failed the decode gate — verify the
   pin size against decode tok/s before shipping larger.
5. **Tier 9 dylan xattn prefill — deferred, rebased.** Only after the T8 decision
   (if dylan's dflash2 line wins, xattn rides on it) + prefill/TTFT bottleneck evidence.
   Re-derive the pick list from experimental@582431c0 (the 08-26 list predates
   e3f40a24 + 4 more commits). Gate if pursued: their GQA suite + PPL parity +
   prefill/TTFT battery + free-GPU ctest.
6. **Sub-8-bit KV (E8 lattice / KVarN / KIVI) — HOLD, evidence updated
   (08-30, user doc review "is there a version of ninfer that supports kvarn
   kv context").** KVarN proper does NOT exist in any ninfer fork: KVarN
   (Hadamard rotation + dual-scaling variance normalization, calibration-free)
   is vLLM-native (huawei-csl fork, `kvarn_k4v2_g128` fixed preset) +
   beellama.cpp (kvarn2-8, independent K/V bit widths, F16 precision tail).
   KIVI (per-channel K / per-token V): vLLM plugin 4-bit only, no ninfer port.
   NVFP4 KV cache = vLLM SM120 flashinfer patch only (NVIDIA: <1%
   LiveCodeBench/MBPP loss; Terminal-Bench 38.2 vs FP8 36.0, not significant)
   — out of scope for ninfer. The ninfer-ecosystem 4-bit path is the E8
   Conway-Sloane lattice family (upstream #35 held; fork lineage
   UDPSendToFailed/ninfer-4090 -> pefman/sergiuszm/KobusG): `rk4v4-e8`
   (4-bit K + 4-bit V) / `rk2v4-e8` (2-bit K); 250-350k context reported on a
   4090; 4x KV compression vs BF16 (= 2x vs our INT8). Evidence: no E2E
   coding benchmark for E8 (98.7% = signal fidelity only); multi-source data:
   4-bit KV near-lossless for coding (naive INT4 ±0.5%, KIVI-4 -1..-2%),
   2-bit only viable via KVarN variance normalization (-0.4..-0.7%) while
   naive INT2 collapses to 0%. Our T12 tree confirmed: DType enum =
   BF16/I8/FP8_E4M3FN only — no sub-byte KV types (the "lattice" hit in
   dflash2_draft.cuh is a comment). Hold rationale unchanged: 40-50% decode
   slowdown (recorded on #35) + lane fits 225k INT8. Revisit ONLY as the lever
   for (a) context >225k, or (b) T8 draft-module headroom if the ~1.3 GiB
   NVFP4 module (or the 2.07 GiB W8G32 fallback) turns out too tight vs
   the ~2.36 GiB free-after-startup slack (4-bit KV halves the 10.85 GiB
   INT8 pool -> ~5.4 GiB freed, or ~450k tokens at pool size)
   Candidate = `rk4v4-e8`, never 2-bit. If pursued: port from
   KobusG/ninfer-engine (5090-class measurements) rather than pefman/
   sergiuszm; gate = decode tok/s regression ceiling + coding-battery parity
   + free-GPU ctest.

**Holds (revised 2026-08-30):**

| Item | What | Why not |
|---|---|---|
| upstream #35 (E8 compressed KV) | 262k context via KV lattice quant (`rk4v4-e8`/`rk2v4-e8`: Hadamard + Conway-Sloane; forks pefman/sergiuszm/KobusG, lineage UDPSendToFailed/ninfer-4090) | 40-50% decode slowdown (recorded on #35); lane fits 225k INT8. 08-30 doc review: no E2E coding benchmark for E8 (98.7% = signal fidelity only); 4-bit KV near-lossless for coding (naive INT4 ±0.5%, KIVI-4 -1..-2%), 2-bit only via KVarN variance norm; KVarN/KIVI have NO ninfer port (vLLM-native); NVFP4 KV = vLLM SM120 patch only. Revisit as the lever for context >225k or T8 draft-VRAM headroom — see queue item 6 |
| upstream #84/#59 (Windows), #37 (Ollama) | platform/protocol additions | not the 5090 OpenAI-protocol lane |
| cometkim dev (111 behind upstream) | whole dev line incl. kernel-perf 14 commits | moving target on a stale base; re-audit when it absorbs d9dbe1ce — then the prefill-perf subset may be adoptable separately from DFlash2 |
| cometkim 768k presets + eval harness | long-context presets; quality tooling | lane runs 225k; pull the eval harness in to validate T8/T9 when they run |
| dylan perf/rtx5090-qwen38 | width-invariant fix line (7d566547 et al.) | dead since 08-18; the live line is experimental |
| eason / md / knoop / matpape | dead or redundant | eason dead 08-22; md ships via upstream (in-tree); knoop line arrived via T12; matpape redundant with #57/#65 |

**Sequencing (un-parked 2026-08-30, updated after the host-KV A/B finding):** (1)
re-ship the winning lane config (template v22.4.0 + a host-KV pin size that passes the
decode gate — isolated by A/B) + full battery; (2) T8 DFlash2 probe on the rebased
PORT set (the 6 picks must be re-landed onto the T12 softmax_attention layout — see
port status under item 1) — the only decision the plan hinges on; (3) T13 upstream wave
(#107 + #97 + #72) after the decision; (4) T14 host-KV content-cache re-port whenever a
free-GPU window opens (independent of 1-3); (5) the dylan lane-perf wave (packed verify
`2c7785d9` + decode-band pins + blocking host sync + no-re-tokenize + cancel-retain —
see the dylan re-audit section) after the T8 verdict; (6) Tier 9 only after the
DFlash2 decision + TTFT evidence. Standing watch: cometkim rebase onto d9dbe1ce, dylan's daily
experimental commits, #75 follow-up PR.

## Audit 2026-08-30 — dylan/experimental deep re-audit (@ 582431c0)

Full clone `/tmp/dylan-audit`; range `d8cb420f..582431c0` = 23 commits (08-25..08-30)
since the 08-26 dylan audit tip. dylan ships daily on a 5090 (same silicon as ours).
Per-commit triage; conflict surface spot-checked against `~/containers/ninfer-github`.

### P0 — fixes the in-flight T8 port is missing (cherry-pick after pick 8, handed to the port agent)

| Commit | Fix | Why P0 |
|---|---|---|
| `13a2e5d0` stop KV-RAM and DFlash entitlement poisons | (a) cancelled suffix prefill must keep `ledger_frontier == execution_frontier + 1` (one trailing ledger slot) or the next RAM spill throws and marks the executor failed; (b) Main KV entitlement must cover `verify_width` in BOTH tree and chain verify — chain verify also materializes frontier+W, so short-`max_tokens` requests throw "materialize extent is outside entitlement" and poison the executor | the harness cancels streams; the DFlash path we port runs chain verify (k=4/W=5). program_impl.h (revert_cancelled_prefill_lane) + request_plan_impl.h (plan_request_base) + 3 regression test files (159 lines) |
| `eb5bedb5` verify the paper chain and keep C>1 on C=1 GDN math | Qwen3.8 DFlash2 = **chain-only** (k=4/W=5 on 5090 — packed-tree verify lost to W=k+1 on AIME/story; drop tree verify from the probe config). The T=W×B flattened NVFP4 GDN conv-record compose flipped greedy column 0 vs the C=1 fused SmallT+FP32; the fix = one fused launch with grid.x=B (batch-isolated) + S3 packed-append hybrid block max. `dflash2_path_select.cuh` now takes the shared `SamplingConfig` (ninfer/ops/sampling.h) instead of its own per-batch temp/seed struct — interface change if our pick 6 (0a7bef84) brought the old one | ~15 source files, mostly the GDN nvfp4 decode kernels the T8 module rides on |

### Lane-perf wave (post-T8 verdict, free-GPU window, battery gate incl. 98k/225k quiet-window R2)

| Rank | Commit | What | Payoff | Effort |
|---|---|---|---|---|
| 1 | `2c7785d9` packed speculative verify as one aggregate T=width×B pass | removes the per-request T=width re-launches (B weight re-reads + B graph nodes); C=1 route pinning (`packed_route_tokens`) keeps the W4A4/A16 cutover stable | their 5090 A/B: MTP3 C=2 178.4→275.9 tok/s (+54.7%), DFlash2 C=3 +55.9%, C=1 unchanged, no PPL drift — the biggest single lever for our C=4 MTP3 lane | medium (text_context_impl.h, variant.cpp 27b/35b) |
| 2 | `583d8e10` block host sync to fix 100% CPU during decode | cudaDeviceScheduleAuto resolves to spin on 5090-class boxes → per-round synchronize() busy-waits a core; switch to blocking sync (primary-ctx flag + blocking events) | live lane shows the class (ninfer-serve ~88% lifetime CPU, 0% when idle); 46 lines across device.cu/h + kv_ram_cache.cpp | small |
| 3 | `20f622f6` + `2c295b6e` decode-band linear routes (measured W4A4 schedules, pinned A16/W4A4 cutovers) | per-T schedule/crossover pins for the decode-band linears; GEMV/SmallT grow a batch/token grid axis so C>1 fused GDN shares one launch without flattening T to W×B | decode tok/s on our exact silicon; their docs/maintainer/nvfp4-decode-linear.md bound cards are the measurement authority | medium (nvfp4 linear kernels + variant.cpp — we edited variant for the quasar profile) |
| 4 | `48d18570` don't re-tokenize the whole conversation on every turn | engine remembers committed chat tokens (16 histories × 262144 tok, 48 MiB worst case), encodes only the new suffix; ~21 ms saved at 150k | the harness re-sends full history every turn — a per-request win | medium (new encoded_history_cache.cpp + frontend.cpp/tokenizer.cpp — C1 collision surface) |
| 5 | `3946c441` retain cancelled sequences for prefix reuse | decode-ready / speculative in-flight cancels retain like OutputLimit (previously the lane was wiped at every GPU boundary); incomplete suffix prefill reverts to the occupy-base rollback | the harness cancels a lot → better prefix-reuse hits on resumed sessions | medium (program_impl.h +115, +477 lines of tests) |

Secondary (same wave or later): `ac8fea98` MTP prefix-reuse checkpoint fallback (decide_resident_reuse extracted as the single source of truth; MTP append-not-ready → usable checkpoint instead of FullReset; 12-scenario test); `6951a47f` persistent event in order_copy_after_compute; `4f99f6da` isolate gdn snapshot leaf workspace (port of upstream f08597d's intent — our wrapper diverged, variant.cpp only); `9ec0d8c4` MTP layer weights BF16→NVFP4 (converter tool `convert_mtp_nvfp4`; MTP propose rides the NVFP4 kernels + fused input mix — VRAM saving for the quasar-v2 recipe); `582431c0` HTTP timings match the engine log (unified token bases, graph install excluded from decode.ms); `f708cb21` ninfer_bench `--spec dflash` surface — use it for the T8 probe battery.

### Deferred / N/A

| Commits | Class | Reason |
|---|---|---|
| `6fac74a7` host-RAM context-checkpoint restore on MTP+DFlash (ladder marks 24576..151552) | engine feature | competing/complementary subsystem to T14 (#73 re-port) — defer until the host-KV shape settles |
| `d98f3fdb` request-pinned checkpoint + freeze-ladder replacement | serve API | useful (the harness could pin checkpoints) but restructures the automatic ladder — adopt with the T13 wave |
| `3b730ee8` / `55dde21c` / `a56c3a77` / `1e9c5dda` xattn series | Tier 9 | confirmed T9 candidates; `a56c3a77` (per-page K-centering PPL blowup fix) is only relevant if sage/xattn gets adopted |
| `7dd5feeb` / `0edd6778` / `56b0196b` eval suites (qwen3.8 groupwise-int/nvfp4 reasoning; ifbench/erqa/real_world_qa) | eval | pull in for the T8/T9 quality gates, not lane code |
| `3ef301ee` / `88b79ea2` kdev kernel-dev tooling | tools | dylan's internal iteration loop; the decode-band bound cards in docs are the reference asset |

**Status (08-30):** `13a2e5d0` + `eb5bedb5` handed to the T8 port agent as picks 9/10. The lane-perf wave queues after the T8 verdict. None of this is upstreamed — fork-only dylan work.
