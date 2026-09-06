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
## T22 in progress (2026-09-04, post-audit)

**#170 fix verified:** the user's pointer was correct — `b8786751`
(fix(runtime): correct aliased state ownership, 356-line program_impl.h rewrite +
state_image_store/request_plan/resource_projection changes) is the fix, and it adds the
`shared-rewrite-materialization` regression scenario to `test_engine_prefix_real.cpp`
("shared/private rewrite alias did not materialize through its active Fork") — the exact
#170 halt class. It rides the T22 wave; the wave's ctest + REPLAY/XHIGH battery cover it.

**T25c needle gate (T17 validation, user-approved): PASS.** 64k-needle probe
(`probes/t25c-needle-64k.py`) on the live t15-yarn image (T17 pv-f16acc live): 66.4k-token
prompt, needle at ~50% — retrieved (`ZQX-7741`, 11.3s, http 200). The md-catalogue caveat
("pv-f16acc fails the 64k needle") was measured on their 35B-A3B target and did NOT
reproduce on our qwen3.8-27b lane. **T17 stays shipped.** (First run was a probe-side
artifact: max_tokens=60 was fully consumed by reasoning tokens — empty content, no
retrieval verdict; re-ran with max_tokens=1024.)

**T22 merge complete:** branch `t22-converge` @ `9b148057` = t15-yarn `0d49ac8f` +
upstream master `863aa8a5` (9 commits). 7 conflict files resolved:
- `apps/perplexity/main.cpp`: kept our T15 rope-scaling fields + upstream `--log-level`.
- `apps/serve/main.cpp`: kept `log_terminate` (set_terminate stays) + our webui
  `<filesystem>` includes; dropped our `log_engine_capacity`/`kv_capacity_mode_name`
  (dead — upstream's `OperationalLog::engine_capacity` supersedes; the LEDGER marker is
  preserved by the 6e2786c5 log restore).
- `src/runtime/engine/engine_core.h`: kept both includes (`<iostream>` ours,
  `<limits>` theirs).
- `src/targets/qwen3_6/impl/frontend/frontend.cpp`: kept our `state.raw_output`
  (upstream dropped raw_output entirely — our raw path still needs it) + upstream
  `prefix_execution.tracking` (719d56ef).
- `tests/CMakeLists.txt`: dropped the upstream `ninfer_cli_options_test` block —
  duplicate of our existing target (ours already compiles `apps/cli/options.cpp`).
- `tests/targets/qwen3_6/test_frontend.cpp`: kept both helpers (our `poison_resources`
  + upstream `fixture_tokenizer`).
- `tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp`: kept both blocks (our
  empty-reasoning checkpoint checks + upstream `shared-rewrite-materialization`
  degradation stats check).

**Ship scheduled:** `ninfer-ship.sh --branch t22-converge --tag t22converge-9b148057`,
delayed start (lane stop at G4 is by design; G7 auto-rollback retags :quasar back to
`28623fdc57dd` + restart + ImageID verify). Pre-ship: vllm inactive, quadlet pinned to
:quasar, clone clean at 9b148057, buildstage-merge present. ctest-baseline: watch for
new skips from the new upstream test scenarios — if G4 reports NEW_SKIPS, triage and
`--refresh-baseline` if legitimate.

**First ship attempt (17:15) FAILED at G4 — auto-rolled back, lane back on the old image.** The ctest gate caught two post-merge bugs:
1. `tests/CMakeLists.txt`: the merged `apps/cli/options.cpp` calls `product::parse_log_level`, but the merged `ninfer_cli_options_test` target lacked `ninfer_product_logging` in LIBRARIES → link error. Fix `c8ba7c2a`.
2. `tests/targets/qwen3_6/test_frontend.cpp`: upstream 5973313d ("test(frontend): make fixtures self-contained", #156) renamed `official_tokenizer()` to `fixture_tokenizer()` — the auto-merge kept our wave's callers (empty-reasoning checkpoint test) and dropped the definition → compile error at line 642. Fix `1eff9672` restores `official_tokenizer()` (our tests need the real 27B tokenizer; the ctest container mounts it at the /home/neroued path).

**Pre-check (2026-09-04, post-fix): GREEN.** Free-GPU build + full ctest on t22-converge @ `1eff9672` (lane stopped like G4): BUILD_EXIT=0, CTEST_EXIT=0, 100% of 106 tests (real-model tests Skipped per baseline). Lane restarted.

**Reship scheduled:** `ninfer-ship.sh --branch t22-converge --tag t22converge-1eff9672` at 19:45 CEST (systemd timer `ninfer-ship-t22v2.timer`; same gates, G7 auto-rollback verified).

## T23 — upstream TMA wave (queued, `t23-tma-pair` branch)
Stacked on the fixed t22-converge. Cherry-picks from open upstream PRs (MichaelDementii/ninfer):
- `7eb52dba` — PR #167 (`3d6f7f2e`): stage the fp8 A8 GEMM operands through TMA.
- `c4cbcbbc` — PR #160 (`545f64b0`): make the TMA route read NVFP4 activation scales tile-contiguous (1-line include conflict in `nvfp4_linear_swiglu_w4a4_tma.cu`: kept our `#include "op_tester.h"` + their TMA include).
Gates before shipping: post-merge compile + full ctest + battery (after T22 lands).

## T18 — dylan wave 2 (queued, `t18-dylan-wave2` branch)
Stacked on t23. Cherry-pick of dylan's P0 GDN chunked-prefill precision fix (`f25f5463`, "fix(gdn): improve chunked prefill numerical accuracy" — FP16 private normalized Q/K + chunk workspaces, state rel-err −79.2%, output rel-err −52%):
- `plans/qwen3.8-27b-performance.md` (DU conflict): kept our deletion (fork hygiene — no dylan plan files in our tree).
- `tests/ops/test_gated_delta_net.cpp` (UU): dylan's commit replaced `batch_update_case` with a new `partition_case` test + a refactored `batched_snapshot_case` whose BODY differs (they are NOT one shared-body function — a plain 3-way merge left a dangling opening that would not compile). Resolved as the union: their complete `partition_case` + their complete `batched_snapshot_case` + our complete `batch_update_case`, with all three call-site sets in main() (2 partition + 3 batched_snapshot + 4 batch_update).
- Kernel files (launch.cu/h, output.cu/cuh, prepare_wy_wu.cu/cuh, state_passing.cu/cuh, gated_delta_net.cpp, launch.h) auto-merged clean.
Gates before promotion: GPU ctest (`ninfer_gated_delta_net_test`) + MTP/dflash accuracy battery on the lane.

**T22 first-ship failure (17:15) + post-merge fixes:** the 17:15 ship failed at G4
(ctest, 12 failures, all inside the NEW upstream scenarios from the wave, not
regressions in existing coverage) and auto-rolled back (lane back on
`28623fdc57dd`). Two genuine post-merge bugs found and fixed on the branch:
- `c8ba7c2a`: upstream's fixture rework added `--product-logging` handling to
  `apps/cli/options.cpp`; our `ninfer_cli_options_test` needed the
  `ninfer_product_logging` target linked (CMakeLists).
- `1eff9672`: the upstream fixture rework dropped our `official_tokenizer()`
  helper that the wave's qwen3.6/27b tests call; restored verbatim
  (tests/targets/qwen3_6/test_frontend.cpp, test_engine_prefix_real.cpp).
Pre-check re-run GREEN (build 0, ctest 106/106, real-model skips per baseline).
Reship scheduled 19:45 via `ninfer-ship-t22v2.timer` ->
`ninfer-ship.sh --branch t22-converge --tag t22converge-1eff9672` (same
auto-rollback gates).

**T23 (TMA wave) prepared:** cherry-picked upstream PR #167 (3d6f7f2e: TMA route
reads activation scales tile-contiguous) + PR #160 (545f64b0: TMA descriptor
cache) onto the FIXED t22 line -> `t23-tma-pair @ a05618f7`, pushed. Gates before
shipping: post-merge ctest + REPLAY/XHIGH battery AFTER T22 lands (stacked).

**T18 (dylan wave 2) prepared:** cherry-picked dylan's P0 GDN chunked-prefill
precision fix (`f25f5463`, state rel-err -79.2%) onto the current line ->
`t18-dylan-wave2 @ 30a1f805`, pushed. Compile gate GREEN (build 0,
ninfer_gated_delta_net_test target). Conflict resolution: plan file (kept our
deletion) + test file (reconstructed merge). Scope note: dylan's commit is not
self-contained — its `snapshot_case`/`batched_snapshot_case` tests call
`gated_delta_net_snapshot`, an op we have NOT adopted (impl + declaration +
runtime integration landed earlier on dylan's line, separate feature). T18 is
scoped to the self-contained part: kernel fix + the `partition_case` FP64-oracle
regression (dylan's new self-contained test) + our existing tests, all 4
`batch_update_case` cases preserved. **Remaining gate at adoption: GPU ctest.**
The `gated_delta_net_snapshot` op (impl in gated_delta_net.cpp, declaration in
include/ninfer/ops/gated_delta_net.h, text_context_impl.h integration) + its
tests = separate adoption item (T19 candidate) if the MTP/draft-state path is
wanted.

## T-Q — QUASAR A/B campaign (2026-09-04, `t24-quasar-a` branch @ c178fae9)

**Context:** the user (Mirko) maintains `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` — a
QAT (quantization-aware-training) NVFP4 variant of the SAME Qwen3.8-27B model our
lane serves (same 27B backbone, same MTP draft module, different training +
quantization). Our lane artifact (`qwen3_8_27b_quasar.ninfer`, `quasar-nvfp4`,
full-NVFP4 backbone) vs his artifact (`QUASAR-QAT` repo, `nvfp4` weights ID,
400/496 source NVFP4 matrices preserved + 96 bf16 control matrices + QAT-tuned
MTP). Plan (user-approved): A/B on the same engine, keep the winner.

**T-Q1 — engine port (DONE 2026-09-04, `t24-quasar-a` @ c178fae9):** cherry-picked
his `2d11c992` (feat(nvfp4): support Qwen3.8 QUASAR artifacts,
[PR #19](https://github.com/MirkoCovizzi/ninfer-rtx5090-laptop/pull/19),
`MirkoCovizzi/ninfer-rtx5090-laptop`) onto `t22-converge` — the LIVE lane tree —
so the A/B runs on the current engine, not the stale t15yarn base. Conflict
resolutions (his commit predates the t22 rework):
- `27b package.h`: appended `Qwen38Nvfp4LegacyW8` + `Qwen38Nvfp4Quasar` after
  `Qwen38Quasar` (no enum renumber); `resolve_weights` signature
  `ArtifactIdentity` → `Reader` (content detection needs tensor access);
  `Reader` fwd decl.
- `27b package.cpp`: adopted his `tensor_matches`/`endpoint_matches` helpers +
  Reader signature. `"nvfp4"` weights id now routes by endpoint storage:
  W8 endpoints + NVFP4 layer-3 attention QKV → `Qwen38Nvfp4Quasar`;
  W8 endpoints otherwise → `Qwen38Nvfp4LegacyW8` (the ostfralla-era W8 artifact
  keeps working); FP8 endpoints → `Qwen38Nvfp4` (t22 behavior preserved).
  `"nvfp4full"` → `Qwen38Nvfp4Full`, `"quasar-nvfp4"` → `Qwen38Quasar` kept.
- `bindings.cpp`: `Qwen38Nvfp4Quasar` binds the NVFP4 text layout with
  `full_nvfp4=true` (all 256 attention/GDN/MLP parents NVFP4 + fused
  `gdn/a_b_projection [96,5120]` bf16 — his exact 32 attn + 96 GDN + 128 MLP
  layout); `Qwen38Nvfp4LegacyW8` binds `full_nvfp4=false` (legacy Qwen3.6-style
  exceptions). Both W8G32 endpoint format.
- `variant.cpp`: both profiles join the NVFP4 workspace-capacity case group (7 sites).
- `registry.cpp`: `Target::resolve_weights(reader)` (template call — all targets
  now take a Reader); `qwen3_6_35b_a3b` aligned (decl + def + `Reader` fwd decl).
- `test_load_plan.cpp`: adopted his final file (adds `verify_qwen38_quasar`:
  1268/1006/6 materialization, 256 NVFP4 parents, fused-GDN-control format
  check; `verify_qwen38_modern`; generic 3-arg `verify_nvfp4` for the legacy
  ostfralla W8 artifact); restored t22's `StartupObserver{}` arg in
  `verify_profile_mismatch_rejection`; dropped t22's `verify_rejection` (needs
  a Reader without an artifact — not constructible; dropped in his line too).
- Converter tooling: `convert_nvfp4_quasar.py` + `recipe_nvfp4_quasar.py` +
  `inventory_nvfp4_quasar.py` (his, auto-merged); `test_quasar_nvfp4_converter.py`
  moved to `tests/convert/qwen3_8_27b/` (t22 layout).

**Artifact:** `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` @ `d8e6fbfa3e` downloaded to
`~/.local/share/ninfer/models/mirko-quasar-nvfp4/qwen3_8_27b_nvfp4.ninfer`
(17,555,331,072 B; SHA256 verified against the published `SHA256SUMS`; manifest
matches the contract: 1268 objects / 1262 tensors / 6 resources, NVFP4 256,
bf16 control 96, MTP module pinned to his recipe revision).

**T-Q2 — Phase A ship (engine + our model):**
`ninfer-ship.sh --branch t24-quasar-a --tag t24quasar-c178fae9` — full gates
(G1 push, G2 build, G4 free-GPU ctest vs baseline, G5 tag/container match,
G6 battery + REPLAY + 4XX-WATCH, G7 auto-rollback). Ships the T22-converge
engine (subsumes the failed 19:45 T22 reship — its G1 push failed on a remote
divergence; its pre-check was GREEN) + the QUASAR profile. Lane downtime
~8–12 min (restart window + model load).
Verdict so far: pre-check build pending (t24quasar-precheck image).

**T-Q3 — Phase B (A/B on his artifact):** quadlet swap only — same `:quasar`
image as Phase A, no rebuild:
- add `Volume=%h/.local/share/ninfer/models/mirko-quasar-nvfp4/qwen3_8_27b_nvfp4.ninfer:/workspace/models/mirko_quasar.ninfer:ro`
- `Exec` → `ninfer-serve /workspace/models/mirko_quasar.ninfer … --model-id qwen3.8-27b`
  (his artifact's public model id; the `nvfp4` weights id resolves to
  `Qwen38Nvfp4Quasar` via T-Q1's content detection)
- battery `--model qwen3.8-27b` against the SAME baseline JSON.
**Decision policy (user-approved):** B is kept only if (a) every battery verdict
PASSes and (b) decode tok/s ≥ Phase A's measured tok/s (fresh and 8k). Any hard
FAIL (load failure, 4xx, battery gate) → auto-revert to our artifact. If B is
within ±5% of A → keep A.
**T-Q4 — record + keep winner:** winner pinned (quadlet left as-is = winner);
loser's model dir kept (NOT deleted); ADOPTION.md verdict entry with both
batteries' numbers; the engine port stays in the tree regardless of outcome
(profile + converter serve future QUASAR artifacts).

**T-Q2 ship 21:30 CEST — G1–G5 GREEN, G6 battery pass=14 fail=2, ship script
crashed before G7 (no auto-rollback ran):**
- G1 push, G2 build (`4b4c1198`, tags t24quasar-78d74df5/:quasar/:latest),
  G4 free-GPU ctest **PASS** (rc=0, 106 tests, skips within baseline),
  G5 restart + tag/container match **PASS** — all green.
- G6 battery: 12/14 gates PASS (vision x3, replay x4, schema, tool-call/reject/
  retry, xhigh, think, quality, soak 5/5, 4xx-watch). The 2 FAILs are
  DECODE-FRESH (128.6 vs gate 132.4) + DECODE-8K (6.9 vs gate 130.5 tok/s) —
  the 8k probe ran while the user's active OMP session was hammering the same
  lane (R6 caveat signature: ~3-7 tok/s under load vs ~140 idle). No engine
  regression evidence.
- Crash root cause: the pipeline script `ninfer-ship.sh` was edited (the `-j4`
  ctest patch) at 21:50:14 WHILE the 21:30 ship was still executing it —
  bash re-read the byte-shifted file and hit a phantom syntax error at the
  G6 echo line (line 228, `($BV)` parens), killing the script before G7.
  The ctest gate itself is unaffected (it ran in the podman container from
  the mounted source tree).
- Process fixes applied: (a) the G6 echo line de-parenthesized;
  (b) `NINFER_PIPELINE` env override added so timer services run a
  snapshot copy of the pipeline dir (in-flight edits can no longer corrupt a
  running ship); (c) rule: never edit pipeline files while a ship is
  `is-active` — verify with `systemctl --user is-active` first.
- Lane left on `4b4c1198` (new engine, our model): ctest green + 12/14
  battery gates green + the 2 decode fails are load artifacts.
**Resolution path (scheduled):** 23:15 CEST `ninfer-qab-phasea.timer` runs a
quiet-window battery on the same engine (model qwen3.8-27b-quasar); any gate
FAIL in the quiet window -> auto-rollback to `28623fdc` + ABORT marker.
23:30 CEST `ninfer-qab-phaseb.timer` runs the A/B swap (gated on the clean A
reference from 23:15; aborts on the ABORT marker). NOTE: both windows need a
QUIET session (no active OMP chat on the lane) or the decode gates will fail
again under load.

**T-Q2 clean-window resolution (23:15–23:32 CEST) — GREEN:** the scheduled quiet-window
battery (detached chain, auto-rollback on gate fail) passed all 16 gates (log
`~/.local/share/ninfer/logs/battery-qab-phase-a-clean-2026-09-04.log`): vision x3,
replay 10/10, xhigh, think, quality, soak 5/5, 4xx-watch, plus DECODE-FRESH=147.1 and
DECODE-8K=140.7 tok/s. The 21:30 G6 decode fails are confirmed load artifacts (active
OMP session on the lane: ~3–7 tok/s under load vs ~140 idle). A-clean reference written
to `quasar-ab/A-clean.txt` (`AFRESH=147.1 A8K=140.7`); Phase B unblocks from it.

**T-Q3 Phase B (23:55–00:07 CEST) — KEEP B (QUASAR-QAT wins):** quadlet swap to Mirko's
artifact on the same `t24quasar-78d74df5` engine image (no rebuild), battery 16/16 PASS
(log `battery-qab-phase-b-2026-09-04.log`), decode fresh 147.1→151.9 tok/s (+3.3%), 8k
140.7→150.2 tok/s (+6.8%) — clear win both probes, no ±5% tie rule engaged. Lane now
pinned to `mirko-quasar-nvfp4/qwen3_8_27b_nvfp4.ninfer` with public model id
**`qwen3.8-27b`**; our artifact's id `qwen3.8-27b-quasar` is no longer served — clients
pinned to the old id must use the new one (or get an alias entry on the same :8002
base_url). Rollback path intact: A-state quadlet backup at
`ninfer-nvfp4.container.bak-ab`, our model dir kept (T-Q4: loser not deleted).

**quasar-ab/phase-b.sh bug fixes (all pre-relaunch, 23:46–23:55):** four latent bugs fixed
in `~/.local/share/ninfer/quasar-ab/phase-b.sh`:
1. model-id swap python `str.replace` silently no-opped — the search string's first
   occurrence was a *comment* line above the Exec line, so the live flag never changed
   (lane restarted on A's artifact under B's expectations); now targets the unique
   Exec-line context and asserts the replacement happened.
2. `$BATTERY` referenced but never defined (crash under `set -u`) — defined
   (`$DIR/pipeline/ninfer-battery.sh`).
3. `toks()` called but never defined — B's decode numbers would have silently zeroed and
   forced KEEP_A; defined (copies the battery's `N tok in T s = R tok/s` regex).
4. `grep -c … || echo 0` double-count — on zero matches `grep -c` prints `0` *and* exits 1,
   so the fallback also fired (`0\n0`), making `[ "$BFAIL" != "0" ]` always true and
   forcing a false revert-to-A on a perfect run; replaced with an explicit file check.
(The double-count pattern in phase-a-clean.sh was already fixed at 23:30.)
**Consumer rebind (2026-09-05):** lane public id is now `qwen3.8-27b` (Mirko's artifact); every consumer
pinned to the old `qwen3.8-27b-quasar` id was rebound: OMP (`models.yml` QUASAR entry merged into the
`qwen3.8-27b` entry, dropping the stale "official" row; `config.yml` ocpp-review/scout/sonic overrides;
`gvs-ideator/manager/worker` + `ninfer-template-review` agent frontmatters), Hindsight quadlet (restarted,
0 restarts), aistock (opentrade quadlet + `api.py` LANE_MODEL + Dockerfile .env, image rebuilt via
opentrade-build; OWUI quadlet `DEFAULT_MODELS` + `webui.db` stock-analyst row + `ui.default_models`;
`lane-ensure.json` + `lane_ensure.py` DEFAULTS + pycache purge), pipeline defaults (`ninfer-battery.sh`,
`decode-gate.py`). Verified: lane probe OK on new id, `lane_ensure.py` rc=0 ready, opentrade
`/lane-status` serving_model:true, OWUI `/api/models` lists `qwen3.8-27b`.

## Audit 2026-09-05 — post-T-Q re-evaluation (fresh fetch, T28–T30 proposed)

**Window:** 2026-09-04 → 2026-09-05. **Base:** lane on `t24-quasar-a` (T22 engine + QUASAR-QAT
model `qwen3.8-27b`, INT8 KV, 225280 ctx, MTP3 + lm-head-draft, C=4, vision, preserve-thinking,
built-in 8 GiB host-KV parking). Tree: 215 own commits, **1 behind upstream/master**
(`ad0f3d38 chore: add project funding information` — docs-only).

**Fetch deltas:**
- **dylan/experimental** `7dd98fdc → a68667b9` (4 commits): `81f26a7f` feat: enable numerically
  validated vision with dflash2; `b2b1b7e4` fix(dflash): stop extra-accept past a completed tool
  call; `75faa830` fix(dflash): keep p-less only at hop 0 and isolate packed verify; `a68667b9`
  fix(kv-cache): reuse disk entries with vision. → **T28**.
- **dylan/qwen4** (new branch, `d5863070`): Qwen4 architecture verifier — off-lane (we serve
  qwen3.8-27b); watch for future model support.
- **gzenz/local/combined** `08636ed3 → d205c52a` (19 commits): host-KV safety-net hardening —
  pinned-entry eviction, spill guards, pre-check eviction feasibility, safety-find count cap,
  compact_prefix fixes, session-key → response_id fix, `d205c52a` materialize real rewrite
  checkpoint at finish (Option A) (= the #170 halt class — already fixed in-tree via T22's
  `b8786751`, so this is a fork-side parallel, not a gap). gzenz now 76 ahead / 25 behind
  master. → re-derives **T24**.
- **mirko** (QUASAR engine repo, first fetch in this clone): `feat/dynamic-mtp` (4 perf commits:
  `a1e04606` price adaptive widths by context depth, `08d0d444` eliminate wide MTP decode cliff,
  `f52125a2` tune wide MTP projection, `0e6dd8bd` vectorize e8 packed decode);
  `perf/nvfp4-swiglu-m16n256` (`81e685fc` optimize swiglu for small batches — decode-band
  micro-opt); `feat/kvarn-production` (41 commits: KVaRN mtp3 score production pipeline);
  `fix/mtp-greedy-parity` (`56dfda80` = our Tier 7 content, his line); `integration/upstream-master`
  (`44495ec8` — he is absorbing upstream). His lines are 261–357 behind our lane (different
  base) — adopt by cherry-pick, not merge.
- **eason / cometkim:** no movement (fully upstreamed, dormant).
- **md / upstream PRs:** #167 + #160 (the T23 TMA pair) still OPEN upstream; #173 (rk2v4-e8
  compressed KV) REJECTED per the KV precision floor (E8 family); #163 (serve progress timings)
  + #162 (llama.cpp-compatible /v1/models metadata) small watch items (the latter is useful for
  OWUI display); #152 (automatic shared-prefix write at the system frontier) watch (agent
  workload); #148 (Responses API) watch; #159 CLOSED unmerged.
- **Upstream issues:** #174 (Q4G64 full-vocabulary MTP proposal head — proposal only; pairs
  with T28/T29), #172 (resumable output-limit generations; pairs with #169), #171 (NIAH judge
  strategy — eval tooling), #170 resolved in-tree (T22 `b8786751`), #169 (output-limit
  truncation stops the agent loop) + #168 (strict:true opt-out) open — watch.

**In-tree re-verification (lane):** `b8786751` (#170) IN, `719d56ef` (#158) IN, `a140e7ae` IN;
`ad0f3d38` ABSENT (docs-only delta, rides the next convergence). Queued branches:
`t23-tma-pair @ a05618f7` (2 own commits) and `t18-dylan-wave2 @ 269cf431` (3 own) are both
7 behind the lane — the QUASAR port + docs moved the lane past them, so **re-stack onto
`t24-quasar-a` before shipping either**.

### New tiers (numbered after T27)

**T28 — dylan dflash2 wave (PROBE-GATED, new 2026-09-05).** dylan's dflash/dflash2 line is
maturing on the active experimental branch: `81f26a7f` enables numerically validated vision
with dflash2, `b2b1b7e4` + `75faa830` fix the verify acceptance boundary (extra-accept past a
completed tool call; p-less only at hop 0; isolated packed verify), `a68667b9` reuses disk KV
entries with vision. This is the successor to the T8 question (spec decode beyond MTP3) via
dylan's active line instead of the dead cometkim/taylor-shift lines (native dflash is already
in-tree via T22). Probe design: cherry-pick the dflash2 enablement set onto a re-stacked
branch → free-GPU ctest → A/B vs the MTP3 QUASAR baseline (151.9/150.2 tok/s): gate on
acceptance (drafts/round) as a first-class metric + net decode at 98k/225k quiet window (R2)
+ tool-call boundary regression (the extra-accept fix is exactly our agent-loop class). Merge
only if net decode beats MTP3.

**T29 — Mirko dynamic-MTP decode wave (candidate, new 2026-09-05).** Mirko's `feat/dynamic-mtp`
prices adaptive MTP widths by context depth + eliminates the wide-MTP decode cliff + tunes the
wide MTP projection (+ `0e6dd8bd` e8 packed decode vectorization). Our lane runs fixed
`--draft-tokens 3` + `--lm-head-draft` at C=4; adaptive widths are the decode lever on the
exact silicon Mirko ships (5090). Cherry-pick the 4 perf commits (+ `81e685fc` small-batch
SwiGLU from `perf/nvfp4-swiglu-m16n256`) onto a re-stacked branch → decode A/B vs the
151.9/150.2 baseline (fresh + 8k, quiet window) + quality battery. Numbers are from his line
(QAT recipe included) — re-verify on our artifact.

**T30 — Mirko KVaRN line (TRIAGE, new 2026-09-05).** `feat/kvarn-production` (41 commits):
KVaRN mtp3 score production (balanced score production, pipelined keys/paired-value decode,
score-warp reuse, low-width fused projection tuning). Large port on a diverged base. Adopt
only if triage shows it precision-neutral + decode-positive on our profile; otherwise watch.
If it turns out to be KV compression, the precision floor still applies (no lower-precision
KV without E2E quality evidence) — pairs with issue #164.

### TIERED PLAN (re-evaluated 2026-09-05, round 3)

| Tier | What | Status (2026-09-05 r3) |
|---|---|---|
| 1 | decode & serve quality (#55 #67 #69 #65 #57 #61) | DONE 08-23 (`tier1`, PR #1) |
| 2 | xhigh track (Sharp v22.3.1 + `reasoning_effort` kwargs) | DONE 08-23 (`tier2`, PR #2) |
| 3 | community cherry-picks + Wave B1 decode perf | DONE 08-24 (`tier3`/`tier3-waveb`, PR #3/#4) |
| 4 | upstream convergence + agent-workload (C1–C3, host-KV opt-in) | DONE 08-24 (`tier4`, PR #5) |
| 5 | Wave C (MoE-decode perf, response_format json, Sharp v22.3.2) | DONE 08-24 (`tier5`, PR #6) |
| 6 | portability (SM-count persistent grids) | DONE 08-25 (`tier6`, PR #7) |
| 7 | MTP width-invariant greedy verification | DONE 08-26 (re-adopted, PR #8; first adopt reverted — 50% decode regression) |
| 8 | cometkim DFlash2 probe | CLOSED — probe rolled back 08-31 (4 boot bugs); superseded: native dflash in-tree via T22, dflash2 now lives on dylan's active line → **T28** |
| 9 | dylan XAttention prefill | DEFERRED — DFlash2 decision + prefill/TTFT evidence first |
| 10 | upstream/dev sync | RESOLVED — dev merged into master 08-29, consumed by T12 |
| 11 | quasar re-verification on the T10 merge | CLOSED — rolled back (vision 400); superseded by T12 |
| 12 | upstream convergence wave | DONE 08-30 (lane `t12-4567363e`, battery 16/16) |
| 13 | #98 wave (#107/#97/#72 + pressure fixes) | FAIL 08-31 → DEFERRED — `3d9fda22`/`5e4bf313` are 503-bad under load (auto-rollback); only the 3 bench/docs commits worth; skip until the lane-perf wave lands |
| 14 | host-KV content cache (#73) | PROBE DONE 09-05 — T14 probe confirmed idle prefix loss (45 s/120 s evictions) → port recorded as **T31** |
| 15 | gzenz NVFP4 KV + YaRN (2.12, 400k ctx) | SHIPPED 09-02 in tree (`t15-yarn`) — **not live**: the T-Q quadlet re-pinned INT8 KV 225280 ctx; config flip pending if long-context agent work is the priority |
| 16 | upstream convergence wave 1 | DONE 09-02 (absorbed in the `t15-yarn` line; pulled md micro-opts + GDN fix for free) |
| 17 | pv-f16acc (md) | DONE 09-02 (`t15-yarn`); T25c 64k-needle PASS 09-04 — the 35B caveat did not reproduce on our 27B lane |
| 18 | dylan wave 2 (GDN chunked-prefill precision) | SHIPPED 09-05 (`t18gdn-49400365`, image `e858f88b`, battery 16/16, GDN coverage gate green) — **live lane** |
| 19 | `gated_delta_net_snapshot` op + tests (dylan) | NOT STARTED — candidate if the MTP/draft-state path is wanted (T18 shipped cleanly; no new signal) |
| 20 | watch: open upstream PRs | WATCH — no in-window activity; #167/#160 still OPEN upstream (in-tree via T23 cherry-pick); #152 shared-prefix, #148 Responses API, #163 timings, #162 metadata |
| 21 | watch: upstream issues | WATCH — #169 output-limit, #168 strict:true, #164 KV tail, #166 context-cache 503 unchanged; new #176–#181 cluster → **T32** |
| 22 | upstream convergence wave 2 (`863aa8a5`: #170 fix + tool parser + prefix reuse) | DONE 09-04/05 — LANDED via the T-Q ship (`t24-quasar-a` subsumes `t22-converge` @ `1eff9672`) |
| 23 | md TMA prefill pair (#167 fp8-A8 TMA + #160 nvfp4 activation scales) | SHIPPED 09-05 (`t23tma-bb535075`, image `60c00b73`, battery 16/16) — prefill-side wave, decode gates held |
| 24 | gzenz host-KV safety net | SUPERSEDED → **T31** (T14 probe justified; pick set re-derived from gzenz/local/combined @ `46275617`) |
| 25 | probe wave | PARTIAL — T25c 64k-needle PASS (validates T17); T14 probe completed 09-05 → T31; remaining probes per the 09-04 plan |
| 26 | watch slot | WATCH — upstream-PR watch set (re-mapped each audit) |
| 27 | watch slot | WATCH — community-fork watch set (re-mapped each audit) |
| 28 | dylan dflash2 wave (vision-validated dflash2 + verify-boundary fixes) | BLOCKED 09-05 — no 27B dflash possible (dflash 35B-only, artifact carries no DFlashPayload); dylan line grew again in-window (`631333aa`) but non-portable; revisit on a 27B dflash artifact |
| 29 | Mirko dynamic-MTP decode wave (adaptive widths + small-batch swiglu) | DECIDED 09-05 — T29a port required (`81e685fc` swiglu scheduling conflict; `f52125a2` mechanical); T29b levers no-win (dt4: 8k -2.1%; prefill-chunk 4096: -4.97%) |
| 30 | Mirko KVaRN line (mtp3 score production) | DEFERRED 09-05 — revisit only once a Mirko port lands cleanly (T29a showed his base conflicts with the lane's routing) |
| 31 | gzenz host-KV safety-net port (idle-prefix-loss justified) | NEW 09-05 — pick set re-derived from `46275617` (5 rewrite-checkpoint commits added in-window); gated on the host-KV re-enable decision (pairs with #166) |
| 32 | upstream prefix/context-cache cluster (#176–#181, #142) | NEW 09-05 — WATCH: 6 fresh issues (planner charging, cache saturation, materialization budget, single-slot eviction); #181 independently mirrors our T14 finding; adopt upstream fixes when they land (convergence-wave style); may shrink T31 scope |

**Sequencing (2026-09-05, round 3):** (1) **T31** — gzenz host-KV safety-net port from `46275617`,
gated on the host-KV re-enable decision; (2) **T29a port** — mechanical gdn micro-opt; needs the
swiglu scheduling decision; (3) **T25** remaining probes (if any); (4) **T32** adopt-on-merge
watch; (5) T15 config flip only if long-context sessions outrank the 8–14% native-context decode
penalty. Blocked/deferred: T28 (no 27B dflash), T30 (until a Mirko port lands cleanly), T9/T13
(deferred), T19 (candidate, no signal).

## t23tma-bb535075 ship (2026-09-05) - t23-tma-quasar @ bb535075

Image `60c00b73c4e3` (tags: `t23tma-bb535075`, :quasar, :latest); previous
`:quasar` `4b4c1198791f` retained as rollback target.
- Free-GPU ctest: rc=0, skips within baseline (6 expected).
- Battery: 16 PASS / 0 FAIL: VERDICT UP: PASS VERDICT IMAGE: PASS VERDICT MODELS: PASS VERDICT LEDGER: PASS VERDICT WARMUP: PASS VERDICT VISION: PASS VERDICT VISION-HIST: PASS VERDICT VISION-POISONED: PASS VERDICT REPLAY: PASS VERDICT THINK-SMOKE: PASS VERDICT XHIGH: PASS VERDICT DECODE-FRESH: PASS VERDICT DECODE-8K: PASS VERDICT QUALITY: PASS VERDICT SOAK: PASS VERDICT 4XX-WATCH: PASS
- State: lane `ninfer-nvfp4` runs the new image; :quasar/:latest pinned (verified match).

## t18gdn-49400365 ship (2026-09-05) - t18-gdn-quasar @ 49400365

Image `e858f88b907e` (tags: `t18gdn-49400365`, :quasar, :latest); previous
`:quasar` `60c00b73c4e3` retained as rollback target.
- Free-GPU ctest: rc=0, skips within baseline (6 expected).
- Battery: 16 PASS / 0 FAIL: VERDICT UP: PASS VERDICT IMAGE: PASS VERDICT MODELS: PASS VERDICT LEDGER: PASS VERDICT WARMUP: PASS VERDICT VISION: PASS VERDICT VISION-HIST: PASS VERDICT VISION-POISONED: PASS VERDICT REPLAY: PASS VERDICT THINK-SMOKE: PASS VERDICT XHIGH: PASS VERDICT DECODE-FRESH: PASS VERDICT DECODE-8K: PASS VERDICT QUALITY: PASS VERDICT SOAK: PASS VERDICT 4XX-WATCH: PASS
- State: lane `ninfer-nvfp4` runs the new image; :quasar/:latest pinned (verified match).

## Round 2 (2026-09-05) — T23 + T18 shipped; T29 wave and T14 probe decided

### T23 — t23tma-bb535075 (branch t23-tma-quasar @ bb535075) — SHIPPED, LIVE
Upstream PR #167 (fp8 A8 GEMM operands through TMA) + PR #160 (NVFP4 TMA activation-scales
tile-contiguous) cherry-picked onto t24-quasar-a; ops-side tree byte-identical to prepared
t23-tma-pair. ctest rc=0 (skips within baseline); battery 16/16. Decode gates held as expected
for the prefill-side wave (+1.4% @chunk 1024, +2.6% @4096; bitwise-identical outputs).

### T18 — t18gdn-49400365 (branch t18-gdn-quasar @ 49400365) — SHIPPED, LIVE
Dylan GDN chunked-prefill accuracy fix (269cf431) cherry-picked on T23. ctest rc=0 with the GDN
coverage gate verified: ninfer_gated_delta_net_test ran and passed (2.73 s); full GDN family
green (gating, gating_proj, input_proj x3, replay_record, replay_fold, replay_records); 106
tests, 0 failures. Battery 16/16. Expected effect: FP16 private normalized Q/K + chunk
workspaces; state rel-err -79.2%, output rel-err -52%.

### T29a (Mirko kernel micro-opts) — PORT REQUIRED, NOT PICKED
- f52125a2 (gdn wide-MTP projection): conflicts mechanical only (pure additions:
  nvfp4_gdn_input_w4a4.cu aliases + dispatch branches; test_gdn_input_proj.cpp cases).
- 81e685fc (nvfp4 swiglu small-batch): semantic conflict - removes the lane's T<=3
  SmallTFusedA16 fused route and extends the small-batch threshold (max_tokens >= 2 vs >= 4)
  in nvfp4_linear_swiglu_plan.cpp. A scheduling choice, not a merge.
- Per plan: wave recorded as "conflicts with the lane's TMA/scale routing - needs a port, not a
  pick". No branch created; no lane change.

### T29b (quadlet config levers) - both measured no-win
- 4.2 --draft-tokens 3 -> 4: artifact-legal (mtp_num_hidden_layers=1; engine unrolls the single
  MTP layer; kMaximumMtpDraftTokens=5). Restart clean; memory gate passed (free-after-startup
  2.08 GiB, ~9 MiB VRAM delta). Battery 16/16: DECODE-FRESH 154.4 (+1.6% vs 151.9) but
  DECODE-8K 146.8 (-2.1% vs 150.2) - fails the keep rule (both >= +1%) -> reverted to 3.
  (Battery JSON 139.4/137.4 is the regression floor; the keep rule uses the live references.)
- 4.3 --prefill-chunk 1024 -> 4096: 65k-prompt TTFT 6.44 s -> 6.12 s = -4.97% (below the 5%
  keep bar, single sample, borderline); probe decode flat-to-worse. The md 35B -19.8% does not
  transfer (quiet 27B prefill already ~10k tok/s). Flag removed.
- No flag survived; quadlet left at T18 ship state.

### T14 probe - idle prefix loss confirmed -> T24 justified
- Probe: probes/t14-prefix-retention.py - 76k-token shared prefix, A/B 32-token suffixes, TTFT
  triples, quiet window, capacity pressure ~282k tokens vs the 225280 pool.
- hot: [7.35, 7.35, 0.10, 0.10] s - A/B cold first pass; warm = 0.10 s. Cross-conversation
  non-sharing is by design (PrefixReusePath::PrivateEndpoint is per-sequence,
  request_plan_impl.h:508).
- 45 s idle: [46.56, 0.18, 15.48, 0.14] s - A evicted (re-prefill 46.6 s, then 15.5 s), B
  retained.
- 120 s idle (true idle): [7.33, 10.17, 0.09, 0.10] s - both evicted.
- Rule "any post-idle TTFT >= 10 s" fires (46.56 / 15.48 / 10.17) -> T24 justified; matches the
  third-party #98 benchmark (45 s coin-flip, 22 s re-prefill).
- T31 (next tier, not started this round): port the gzenz host-KV safety-net pick set from
  d205c52a (spill guards, eviction-feasibility pre-check, safety-find cap, session-key ->
  response_id mapping).

### Closures
- T28 (dylan dflash2) - BLOCKED, no work this round: DFlashConfig::supported=false
  (config.h:74), kMaximumDFlashDraftTokens=0; dflash is 35B-only in the tree; the QUASAR
  artifact carries no DFlashPayload (model_view.h:99). Watch for a 27B dflash artifact.
  Future-round note: the full dflash2 wave (3 of its 4 new commits) and the full dynamic-mtp
  branch (4 commits) both edit src/targets/qwen3_6/impl/runtime/program_impl.h +
  layouts_impl.h - a future port of both is one ordered sequence with a conflict pass between
  them, never parallel waves.
- T30 (Mirko KVaRN, 41 commits) - deferred: revisit only once a Mirko port lands cleanly;
  T29a established his base conflicts with the lane's routing, so the bar is a port, not a pick.

## Round 3 (2026-09-05) — 12h re-audit (upstream + forks, 2026-09-04T20:33Z..09-05T08:33Z)

Lane state: `t18-gdn-quasar` @ `cd47dc7f` (image `e858f88b`), **222 ahead / 1 behind**
upstream/master (the behind commit is `ad0f3d38`, a non-code funding chore).

### Upstream (Neroued/ninfer)
- Zero in-window commits on master/dev/feat/kv-nvfp4-k8v4; zero PRs created/updated/merged.
- **#167/#160 (T23 pair) still OPEN upstream** — already cherry-picked in-tree; no re-sync
  needed.
- #173 (rk2v4-e8 sub-floor KV, 208 B/head-token) unchanged; remains rejected by the KV floor.
- **6 new issues (prefix/context-cache planner cluster)**: #176 materialization search 5 ms
  budget cap; #177 private continuation cache permanent saturation; #178 planner charges
  private_transition_loss on already-unreachable checkpoints; #179 physically-infeasible shared
  captures without pressure_evidence; #180 proposal: opt-in rolling context-cache retention;
  **#181 single-slot turn checkpoint evicts the conversation prefix cache (full re-prefill per
  chat turn)** — an independent upstream echo of our T14 idle-prefix-loss finding. Plus #142
  updated (sibling prefix gap, prompt_cache_breakpoint). → recorded as **T32** (watch/adopt).

### Forks (in-window)
- **dylan**: `631333aa` on experimental+master ("perf(runtime): aggregate C<=4 verify kernels")
  — T28 family, dflash2/35B-only numbers (DFlash C2/C3/C4 161→220 tok/s), non-portable — T28
  stays BLOCKED. New `dylan/qwen4` branch (tip `2f51a0be`, 6 in-window commits): Qwen4 model
  target (sparse-MoE/QSA/PLE/GGML, NVFP4-G16 KV, architecture verifier, batched chunked
  prefill) — different model than the pinned qwen3.8-27b; watch only.
- **gzenz**: `local/combined` `d205c52a`→`46275617`, 5 new commits — all in
  `src/targets/qwen3_6/impl/runtime/` (program_impl.h / request_plan_impl.h): rewrite-checkpoint
  materialization at finish + start_sequence, capture group at the reuse_base frontier,
  capture-group derivation from source, preserve rewrite flag. No tests. → **T31 pick set
  re-derived from `46275617`**.
- **md**: dormant (newest 09-04T03:02Z, pre-window); T23 source branches unchanged.
- **eason**: dormant (newest 08-22).
- **mirko**: no in-window commits (newest tip `c853622c` kvarn-production 09-04T10:12Z,
  pre-window).
- **cometkim**: dormant.

### Untracked forks (in-window)
- 6 pushed: kybrcore / Little-Star888 / Sha1rholder (mirror `ad0f3d38`), igorls (PR-author
  re-push), andrewleech/ninfer-v100 (off-lane v10.0 Volta port) — all skipped.
- 1 unique: sunnyyangyangyang `d8e2a27a` "raise prompt vision envelope to full context (262144
  tokens)" — single commit on the upstream tip; one-line envelope change, only meaningful at
  full native context (lane pins 225280) — watch.

### Tier table update
- T23/T18 → SHIPPED (live lane = T18 image); T25 → T14 probe completed; T24 → SUPERSEDED by
  T31; T28 → BLOCKED; T29 → DECIDED (T29a port / T29b no-win); T30 → DEFERRED; T31 → re-derived
  from `46275617`; **T32 → new watch tier** (#176–#181 cluster). T26 = upstream-PR watch,
  T27 = community-fork watch (re-mapped).
- No new tracked fork remotes needed this window.

## Round 4 (2026-09-05) — T31 host-KV safety-net port: 2 latent bugs, NOT shippable yet

Branch `t31-hostkv-quasar`. Ported the gzenz host-KV safety net (justified by the T14
idle-prefix-loss probe). Commits: `db49bd6a` (port), `c5ee0e81` (session-key fallback → lane
Candidate/CatalogEntry API), `92bca578` (frontier fix + test fakes). Both T31 ship attempts
failed and auto-rolled back; the live lane stays on the T18 baseline (`e858f88b` =
`t18gdn-49400365`).

**Ship attempt 1 (`t31hostkv-c5ee0e81`):** G4 ctest FAILED — `test_resource_manager` did not
compile (the T31 API added members the test fakes lacked: `FakeShortlistKey::identity_tag`,
`FakeAdmissionCandidate::set_session_key`, `FakeProgram::safety_net_restore_count`). Rollback.

**Ship attempt 2 (`t31hostkv-92bca578`):** G4 ctest **PASSED 106/106** (frontier fix + test
fakes verified). G6 battery FAILED on a second, distinct bug → rollback to T18 baseline.

### Bug 1 — "completed prefill did not reach the admitted prompt frontier" — FIXED, verified
`engine_core.h resolve_prefill_progress`: the safety-net restore advances the staged prefill
base *after* admission, so the runtime reports more reuse than the committed plan and computes
a shorter frontier. The invariant compared `computed_prompt_tokens` against the **committed**
suffix, so a legitimate restore fataled the engine (all requests 503). Fix: compare against the
runtime's actual reuse (`progress.summary.reused_prompt_tokens`, guarded to never exceed
prompt and never fall below committed via the summary check). No-op for a normal prefill.
Verified by ctest 106/106.

### Bug 2 — "sequence StateImage entitlement is inconsistent" — DIAGNOSED, NOT fixed
`program_impl.h:11241 reserve_state_entitlement` throws when `owned > slots`:
- `owned = sequence_exclusive_state_resources(seq).device.state_slots` — the sequence's
  exclusive device-resident StateImage handles (read/write/rewrite/anchors).
- `slots = state_slots`, a **parameter of `start_sequence`** — the entitlement the request
  planner reserved for this request.
- Only called from the two resident-reuse paths: `PrivateEndpoint` (10505) and
  rewrite-checkpoint-restore (10570).
- Trigger: a request that **reuses a resident prefix** and carries a **vision image state
  slot**. The resumed sequence owns an extra device StateImage slot (the image) the planner's
  `state_slots` did not budget → `owned > slots` → fatal → engine fails all requests → 503
  cascade. Battery repro: `VISION-HIST` (32 msgs, image in msg 1, reusing a resident prefix
  shared with the battery's earlier vision probes).
- Root cause (unconfirmed): the planner's `state_slots` undercounts the vision image state
  slot for a resumed vision sequence; the fix lives in the caller/planning code that computes
  the entitlement. Needs the runtime state values (the e2e/battery window) to confirm + fix.
  **Not** attempted blind — a wrong "fix" risks masking a genuine inconsistency this check
  exists to catch.

**Status:** T31 NOT shippable. Bug 1 fixed + verified; Bug 2 needs the e2e window (GPU free +
no active session on the lane) to confirm the undercount and land a verified fix. Image
`t31hostkv-92bca578` parked on disk. Resume: re-verify lane health/traffic + GPU-free, launch
the ephemeral `t31-debug` container (never created — nothing to clean up), run e2e/battery,
fix Bug 2, clean ship. (See Round 5b rev 2: both bugs fired across the two 92bca578 ship
attempts — Bug 2 killed the verdict-deciding G6 battery, Bug 3 fataled live production
traffic pre-ship.)

## Round 5 (2026-09-05) — entitlement contract pinned by a GPU test (guard for Bug 2)

Commit `5bf447a9` on `t31-hostkv-quasar`: extracted `sequence_exclusive_state_resources` /
`state_exclusive_to_sequence` / `reserve_state_entitlement` out of `ProgramImplCore` into
free functions over `(StateImageStore&, SequenceState&)` in
`impl/runtime/state_image_entitlement.h` (members became delegators; semantics preserved
exactly — including the `owned == slots` no-op early return). New GPU-gated test
`ninfer_qwen3_6_state_image_entitlement_test` pins the Bug 2 invariant: exclusive-ownership
accounting (primary pair, long anchors, borrowed reads excluded, per-owner reference
equality) and the reservation contract (re-affirmation is a no-op; budget-below-owned — the
exact Bug 2 / VISION-HIST trigger — is rejected as inconsistent; multi-step growth rejected;
`owned + 1` materializes exactly one destination).

Verified on-GPU in a disposable buildstage container (`--device nvidia.com/gpu=all`, lane
left serving — test allocates a few MB): **PASS 0.20s, no skip**; full engine compiled
alongside (both 27b/35b variants). Consequence: any fix for Bug 2 (planner `state_slots`
undercount for resumed vision sequences) must preserve this contract — the test will fail
on a regression that loosens the inconsistency check or breaks single-destination growth.
T31 status unchanged: still NOT shippable until Bug 2's undercount is confirmed in the e2e
window and fixed.

The `owned > slots` throw itself is **intentional and stays** — it exists to catch exactly
this class of accounting mismatch. Bug 2 is the planner's `state_slots` undercount for a
resumed vision sequence (the check fires correctly on a real undercount); the fix belongs in
the caller/planning code, not in relaxing this check.

**Bug 2 e2e probe (running 2026-09-05):** `~/.local/share/ninfer/t31-bug2-probe.sh` —
waits for the `t31debug-329d8ac4` image (built from the branch with the diagnostic throw
messages: the inconsistent/destination throws now print `owned=`/`slots=`/`reserved=`),
then in a quiet window (journal-based: no in-flight requests, 15-min abort if traffic
persists) stops the lane, boots the t31debug image as the ephemeral `t31-debug` container
(same quadlet flags, `:8002`), replays the VISION-HIST sequence (fresh image request,
then 32-message histories with image in msg 1 and DIFFERENT final questions per request -
identical payloads take the response-replay fast path and never re-execute the vision
state), captures the `owned`/`slots` values from the engine log, and restores the lane
(trap-guaranteed, VRAM-free check, tag/container match + smoke probe at the end).

**Run 1 (19:42-19:49 CEST): NOT-REPRODUCED.** All gates + restore verified (the script
itself works: build 2m15s from layer cache, quiet-window gate waited out the in-flight
session, VRAM free-based wait, restore = readiness + smoke + tag match on T18 baseline).
The repro missed: hist #2 was byte-identical to #1, so the engine took the 100%
`response replay` fast path (`[inspect] prefix HIT (rewrite)`, `[safety-spill] SKIP: no
endpoint state image`) - the vision state was never re-executed, so the entitlement check
was never reached. Run 2 uses differentiated tails (fresh execution forced on prefix
reuse).

**Run 2 (19:54-20:01 CEST): NOT-REPRODUCED.** Differentiated tails forced fresh
execution (all 200, `cache 0%`), but the shortlist rejected every candidate before
inspect (`key MISS`, `tag_match=-1`, `candidates=1`, no `[safety-find] match=hit`) -
the probe's synthetic 32-message histories share no resident prefix, so the
resume path was never entered. Both runs confirm: the VISION-HIST shape is not the
trigger.

## Round 5b (2026-09-05, rev 2 - CORRECTED) - the 12:05 window was NOT the G6 battery;
both bugs are live ship blockers on e6cec0284

The original Round 5b forensics analyzed the 12:05:38-12:06:54 journal window and
concluded "all G6 probes that completed returned 200; the ship fatal was external
traffic (Bug 3), not a G6 probe regression; Bug 2 is not the ship blocker". That is
RETRACTED. Ground truth (battery log header + ship log + podman + journal
cross-check):

- The ship job that decided the verdict started 12:18:13 (`ship job (log phase) start
  2026-09-05T12:18:13+02:00`); G4 ctest with the lane stopped 12:21:32 (gate PASS);
  G5 restart 12:27:44 (container 729da28d, :quasar re-tagged to e6cec0284, listening
  12:27:52, PID 2442448).
- The verdict-deciding G6 battery ran 12:27:52+ (battery log: `=== battery start
  2026-09-05T12:27:52+02:00 ... lane already healthy - no restart ===`). The battery
  log file for an earlier attempt was overwritten by this run's file (same
  tag+date name, no per-run timestamp in the path).
- The 12:05:38-12:06:54 window belongs to container 9fe32b0b (PID 2367301) - an
  earlier t31hostkv-92bca578 engine instance from a pre-ship live window (the
  12:18:13 ship job did not exist yet). Every request in that window is by definition
  live external traffic (OMP sessions on this lane); the original "REPLAY fixture"
  labels for req#6/#7/#8 were a misattribution - e.g. req#7 is a streaming 64k-xhigh
  32-msg session turn, the shape of live OMP chat, not a captured REPLAY payload.

**Pre-ship live window (12:05:25-12:06:54, container 9fe32b0b) - Bug 3 on production
traffic:**

```
12:05:38  req#2  16 tok -> 200 (12.5s queue behind req#1 prefill)
12:05:39  req#3  129 tok media 1 -> 200 (output 60)
12:05:39  req#4  625 tok 32 msg media 1 -> 200 (output 52)
12:05:40  req#6  live session, 310 msg, 168,716 tok -> prefilling
12:05:44  req#7  live stream session, 32 msg, 168k tok, 5 media, 14 tools -> prefilling
12:05:45  [compact-prefix] bulk [safety-spill] OK x5 resident prefixes to host-KV
          (index=0: frontier=73818 ckpt_frontier=72727 ledger=73819 identity=73819)
12:06:53  req#8  live session, 311 msg, 74,031 tok, admitted 12:06:53.2
          [safety-find] entries=5 match=hit frontier=72727
          [restore] frontier=72727 entry=pinned checkpoint=0
          [restore] KV+state copied, syncing transfer_stream
          -> 12:06:53.796 FATAL "completed prefill did not reach the admitted
             prompt frontier" (engine_core.h) -> engine fatal, all requests 503
```

Bug 3 firing on real user traffic - the strongest Bug 3 signal yet: no probe, no
fixtures, just a 74k-token session continuation restored from the host-KV safety
net. (This window's small vision-shaped probes - 16/129/625/200 tok - all returned
200; the window's tail went 503 after the fatal, battery-shaped: 313/71/72/2-msg
requests.)

**Ship G6 battery (12:27:52+, container 729da28d, PID 2442448) - Bug 2 kills the
battery:**

```
12:27:52  WARMUP (16 tok) -> 200 (133ms)
12:27:52  req#1  EXTERNAL live OMP session: 124 msg, 184,545 tok, 5 media, 14 tools,
          stream, 64k xhigh -> prefilling CONCURRENTLY with the battery
12:27:53  battery VISION single (129 tok, media 1) -> 200 at 12:28:59 (t=66.3s:
          queued behind the 184k external prefill + first-vision warmup)
12:28:59.375  req#4  EXTERNAL live OMP stream admitted (127 msg, 5 media, 14 tools)
12:28:59.409  req#3  battery VISION single done (output 60) -> its StateImage
              becomes resident
12:28:59.465  req#5  battery VISION-HIST (32 msg, media 1, 625 tok) admitted
12:28:59.4xx  ninfer: engine fatal error, failing all requests:
              sequence StateImage entitlement is inconsistent   (Bug 2)
              [journal order: this line FOLLOWS req#5's start -> req#5 is the
               trigger; req#4 was in-flight collateral]
12:28:59.620  req#4 + req#5 -> 500 internal error (both in flight at the fatal)
12:28:59+  every remaining probe 503 "inference engine is unavailable":
          VISION-POISONED, REPLAY 0/10, THINK-SMOKE, XHIGH, DECODE x2, QUALITY,
          SOAK x5
```

Battery verdict: pass=7 (UP, IMAGE, MODELS, LEDGER, WARMUP, VISION, 4XX-WATCH) /
fail=9 (VISION-HIST, VISION-POISONED, REPLAY, THINK-SMOKE, XHIGH, DECODE-FRESH,
DECODE-8K, QUALITY, SOAK) -> G6 rc=1 -> G7 SHIP VERDICT FAIL -> rollback, verified
on e858f88b. Trigger = the battery's own VISION-HIST probe (req#5): admitted
12:28:59.465 - 56ms after the battery's VISION single (req#3) completed and its
StateImage became resident - and the journal's fatal line immediately follows
req#5's start line. The battery log captured the entitlement error text on the
probe's own response (http=500, t=0.2s) - the Round 4 VISION-HIST repro firing
inside the live G6 battery. req#4 (external 127-msg/5-media OMP stream) was
in-flight collateral, not a co-trigger: it was admitted .375, BEFORE req#3's
StateImage was resident, so it could not have hit the resident-reuse throw path
at admission, and no fatal line appears in its .375-.465 setup window; both
req#4 and req#5 show 500 at .620 only because "failing all requests" 500s
in-flight requests.

Corrections to the original Round 5b claims:

- "VISION-HIST PASSED in G6" - wrong for the verdict-deciding G6 (it received the
  500 entitlement). The 200 was from the pre-ship window's earlier probe sequence
  on the other engine instance.
- "the ship fatal was external traffic (REPLAY fixtures)" - wrong as G6
  attribution: the G6 fatal was the Bug 2 entitlement fatal in the VISION-HIST
  phase. The external-traffic fatal was real but belonged to the pre-ship live
  window (Bug 3) - a separate engine instance and event.
- "Bug 2 is NOT the ship blocker" - RETRACTED. Round 4's framing is restored: both
  bugs are live on e6cec0284 and both are ship blockers (Bug 2 killed the
  verdict-deciding G6 battery; Bug 3 killed a live production session pre-ship).

Still valid from the original Round 5b (mechanism, unchanged):

- Bug 3 mechanism: admission sealed as fresh Root (committed_reused=0); runtime
  transparently restores the spilled checkpoint and reports runtime_reused; the
  spill entry's exec_frontier (73,818) includes the spilling request's OUTPUT
  tokens (72,727 prompt + 1,092 output - 1) while its KV checkpoint is 72,727
  (ckpt_frontier); if the runtime reports reuse from exec_frontier (or the ledger,
  73,819) instead of the checkpoint frontier, the expected suffix is short by
  ~1,092 tokens and the `progress.complete` check throws.
- The shared 72,727-token prefix is the OMP system prompt (every OMP session and
  captured fixture on this lane shares it).

Gate gaps found:

- 4XX-WATCH reported PASS ("zero engine rejections since battery start") while the
  engine fataled with 500/503 - it only watches 4xx. Extend it to fail on
  `engine fatal error` / 5xx engine-unavailable.
- The "quiet window" is advisory only: live OMP traffic ran concurrently in both
  windows (184k/127-msg streams during G6; 72k/168k/74k turns pre-ship). Enforce
  it: verify the lane has zero active requests before G6, or fatal attribution
  and decode gates are confounded.

Current state (20:1x CEST): lane on the T18-gdn baseline (e858f88b =
t18gdn-49400365 = :quasar = :latest) after the G7 rollback; healthy (/v1/models
200, chat probe 82ms, decode 170.5 tok/s). The 19:48:49->20:00:53 stop was a clean
managed restart (`server stopped`), not a crash; engine back at 20:01:15.

Next (both bugs must be fixed before any T31 ship):

- Bug 2: the `329a8ac4` entitlement refactor is proven insufficient - the same
  throw fataled the G6 battery. Confirm the state_slots undercount with runtime
  state values in a controlled e2e window (no concurrent traffic), then fix the
  planner's vision state budget for resumed sequences.
- Bug 3: reconcile the restore-reported reuse to the checkpoint frontier
  (ckpt_frontier=72,727), not exec_frontier (73,818) / ledger (73,819); the
  instrumented throw (`b3bf63a4`, image t31debug-b3f63a4) prints
  computed=/prompt=/committed_reused=/runtime_reused=/expected_suffix= to nail
  which frontier the runtime reports.
- Re-ship gate: unmodified battery in an ENFORCED quiet window + the 3-fixture Bug 3
  repro (`~/.local/share/ninfer/t31-b3-probe.sh` + `probes/t31b3-repro.py`), plus a
  74k-restore live-traffic soak.

## Round 6 (2026-09-05, evening re-audit) — gpillon DFlash2 line, md perf-branch flood,
ninfer-fusion, chat-template switch plan (T33–T40)

Window: 2026-09-05T08:33Z → ~19:00Z. New tracked remote added: `gpillon`
(https://github.com/gpillon/ninfer). Highest tier before this round: T32 → new tiers T33–T40.

### Where we stand (roll-up)

- **Live lane:** T18-gdn image `e858f88b` (= `:quasar` = `:latest` = `t18gdn-49400365`), artifact
  `mirko_quasar.ninfer` (identity `qwen3.8-27b`/`nvfp4`, recipe `qwen3_8_27b_quasar_nvfp4-v2`,
  17,555,331,072 B, sha `93181637…`), INT8 KV @ 225,280 ctx, C=4, MTP3 + `--lm-head-draft`,
  vision, `--preserve-thinking`, Sharp v22.4.0 template. Decode baseline 151.9 fresh / 150.2 @8k.
- **Shipped/in tree:** T1–T7, T12, T15 (in tree, not live), T16, T17, T18 (live), T22, T23.
- **Parked/blocked:** T31 (2 bugs, image parked), T13 (503-bad under load), T9, T19, T25 remainder,
  T28 (see below — now unblocked in principle), T29b (no-win), T30 (deferred).
- **Upstream remaining:** `t31-hostkv-quasar` is **237 ahead / 1 behind** `upstream/master`; the one
  behind commit is `ad0f3d38` (funding chore, docs-only). **No in-window upstream code commits** —
  there is nothing left to converge on code-wise. Open upstream PRs: **#183 (NEW)**, #173
  (rejected by KV floor), #167/#160 (already in-tree via T23), #163, #162, #152, #148, #107, #97,
  #84/#59 (Windows, off-lane), #61, #54.

### gpillon/ninfer — the DFlash2 line that actually runs (our T8 lineage, continued)

Fork chain: **Neroued → natpate/ninfer-windows → cometkim/ninfer → gpillon/ninfer.** So this is the
direct continuation of the line our T8 probe rolled back on 2026-08-31. Default branch
`gpillon/coding` @ `a00648cb`; the artifact-matching branch is `feat/dflash2-local` @ `43b03ea5`.
Divergence vs our lane: 378 ours / 99–103 theirs. No in-window commits (newest 2026-09-03).

**The T28/T8 blocker is genuinely removed — but by a DIFFERENT mechanism than we assumed.** Our
27B target keeps DFlash **v1** unsupported on their branch too; they added a *separate* module:

```
ours   src/targets/qwen3_6_27b/impl/config.h:73-91
       DFlashConfig{ supported = false; ... };  kMaximumDFlashDraftTokens = 0
theirs src/targets/qwen3_6_27b/impl/config.h:79-124
       DFlashConfig{ supported = false; ... }        // "DFlash v1 never ships on this target"
       DFlash2Config{ artifact_module = true; execution = true; layers = 5;
                      block_size = 8; feature_rows = 25600; hidden = 5120;
                      intermediate = 17408; query_heads = 32; kv_heads = 8;
                      head_dim = 128; local_capacity = 2048; }   // all-SWA-2048 drafter
       kMaximumDFlashDraftTokens = 0;  kMaximumDFlash2DraftTokens = 7
       kNativeContext = 1048576                      // cometkim 1M envelope — OFF-LANE, do not port
```

**The artifact question — the module is GRAFTED, not reconverted.** `gpillon/…-dflash2-NInfer` ships
`qwen3_8_27b_nvfp4full-v2.ninfer` (18.07 GiB, sha `abb1e120…`) = cometkim v1's 1,259 base objects
**byte-for-byte** plus **66 appended DFlash2 objects** (34 NVFP4 weight matrices + 32 BF16
norm/conv-base tensors), produced by `tools/artifact/graft_dflash2_module.py` (`41231998`). The
drafter is a 5-layer sliding-window(2048) block-diffusion model at target hidden width 5120,
`target_layer_ids = [5,19,33,47,61]`, `selector_rank = 256`, `selector_top_k = 16`,
`block_size = 8`. Module quantized weight-only (rel. Frobenius err max 0.0959 / mean 0.0950).

**Consequence for our constraint (keep QUASAR weights, add DFlash2 on top): the mechanism exists,
but the tool as written refuses our artifact.** `graft_dflash2_module.py:_validate_source_artifact`
hard-checks `identity.model_id == inventory_nvfp4full.MODEL_ID` **and**
`identity.weights_id == inventory_nvfp4full.WEIGHTS_ID` and requires the object list to equal the
module-less `nvfp4full` image; ours is `qwen3.8-27b`/**`nvfp4`** (quasar recipe). The module payload
itself is base-independent (it is the drafter's own tensors at target width, encoded by
`convert_nvfp4full.materialize_dflash2_object` from a `--dflash2-model` checkpoint directory), and
the tool's own comment states *"Shape/format/layout equality is what makes the payloads
transferable"*. **The drafter checkpoint is publicly available upstream of gpillon:
`z-lab/Qwen3.8-27B-DFlash2` (241k downloads; also `incoai/Qwen3.8-27B-DFlash2`)** — so we do NOT
need gpillon's nvfp4full artifact at all; we can encode the module ourselves against QUASAR.

Why this is accuracy-safe: speculative decoding is **verified** against the target model, so a
drafter trained on nvfp4full hidden states grafted onto QUASAR weights costs *acceptance rate*, not
output quality — provided greedy/verification parity holds (the T7 lesson: gate on drafts/round,
never on net tok/s alone). Module is validated at load but not materialized on device unless
`--spec dflash2` is selected, so it costs no VRAM when unused. The card confirms the module runs
with **BF16 or INT8 group-64 KV** — no hyperquant dependency, so our KV floor is satisfied.

**Counter-evidence that forces a probe rather than an adopt:** `ninfer-fusion`'s own
`docs/SPEC-STRATEGY.md` measures *its* DFlash2 path losing to MTP3 at every context — acceptance
24.7% → 12.7% (1.5K → 12K), decode 70.6 → 37.5 tok/s vs MTP3 43–47% / 96.9 → 83.2 tok/s. gpillon
reports native acceptance **3.4–3.7 tok/round** at `block_size 8`. Different implementations,
opposite conclusions → the probe must measure acceptance at 1.5K / 8K / 32K on OUR artifact.

**Bonus finding — a T31 correctness input (see T34).** gpillon carries two silent KV-corruption
fixes. `f4b128c6 fix(kv-ram-cache): stop offering rewrite-checkpoint restores from host RAM`
(26 lines, 1 file) documents that a rewrite-checkpoint restore sourced from a host-RAM record
**answers one request with another request's state** — reproduced with strictly sequential sibling
traffic, verified not a race, not wrong record selection ("the selected record's ledger and identity
verify token-for-token"), and reproducing on `aa8b5dd7` *before* their host-RAM work. Their fix
restricts host-RAM records to append-at-frontier reuse only; measured cost on a 135-request trace =
11 requests (~8%) fall back to `full_reset`. They call it explicitly *"a mitigation, not a cure —
the underlying packed-checkpoint defect remains"*. Also `eaf2037b` (hyperquant exact-key side
store — off-lane KV) and `ac60331d test(kv): compare MTP RAM restores to VRAM`.

Other gpillon content: host-RAM KV tier with probation/protected eviction, prefix-reuse admission,
**tagged request lanes** (`@main`/`@agents`/`@classifier`) so a long-lived conversation is not
evicted by short-lived traffic, streaming/tool-call hardening, adaptive MTP verification-width
calibration. Off-lane: Windows port, 1M-context envelope, hq-e8-2b hyperquant KV (sub-floor).

### Astrangemaninhere/ninfer-fusion — WATCH

Standalone re-imported mirror (not a GitHub fork; no shared ancestry → cherry-pick only), branch
`main` @ `c0f2d27`, 40 commits, created 2026-09-01. On-lane hardware/model (5090 sm_120a, Qwen3.8-27B
NVFP4) and it does touch `src/targets/qwen3_6_27b/**`. Headline is **sub-floor KV compression**
(per-layer 4-bit E8Kv, NVFP4-tier KV, rANS entropy cold pool, NVMe cold tier) shipping *perplexity
only, no E2E quality* → **REJECT** under the KV floor. It also enables 27B DFlash **v1**
(`DFlashConfig::supported = true`, `kMaximumDFlashDraftTokens = 7`, new
`WeightsProfile::Qwen38Nvfp4DFlash2` requiring its own converter run + new artifact) — the nominal
T28 trigger, but its own numbers say DFlash2 loses to MTP3 everywhere, and it would require
abandoning our QUASAR artifact. Everything else (YaRN, GDN, host-KV, prefix-sharing) is already
in-tree via T15/T18/T22/T31 or arriving upstream via #152. `RESEARCH-EXTERNAL.md` is V100/vLLM
notes — off-lane.

### md (MichaelDementii) — 48 branches, one cumulative stack, mostly OFF-LANE

md pushed a large perf stack checkpointed across many branches, all based on `upstream/master`
`ad0f3d38` (237 behind our tip). **Decisive triage axis: md edits `src/targets/qwen3_6/**` — the
35B-A3B sparse-MoE target — plus shared `src/ops/**`. Our 27B is the separate DENSE
`src/targets/qwen3_6_27b/**` target, which md leaves untouched except `6870d530`.** md's decode
numbers are measured on `qwen3_6_35b_a3b.ninfer` (~3B active experts, ~770 tok/s MTP3 baseline vs our
~150) on a 500W-capped vast.ai 5090, and its hot path is the small-T sparse-MoE route (43% of md's
round) which our dense 27B does not have → **md decode tok/s do not transfer.**
Already absorbed by us: decode-fusions, sigmoid-gate (`ed505ebc`, `b3814f6c`), q/k-rmsnorm fuse
(`d3278b79`), fused-TMA-SwiGLU (`00369f63`), pv-f16acc (T17), our own T18 GDN + T23 TMA.
On-lane shortlist: (1) **draft window k=3→5, measured +17.3% with higher acceptance** — a zero-code
serve-flag probe (+ tiny 27B-aware enabler `6870d530`, 15-token window); (2) `8767dac7`
decode-softmax-fold (shared `src/ops/softmax_attention`, clean port); (3) `c735909b` + `16c66809`
nvfp4-TMA prefill extensions **measured on our exact 27B-nvfp4 model** (prefill −2.79…−3.71%) —
extends T23; (4) `ce71f787` MTP sampled-draft (probe); (5) `6870d530`.
REJECT: all `sparse_moe` branches (no MoE on 27B), `weight-stream-evict(-nc)` (evict-first measured
zero), `draft-head-narrow` (35B head, net −1.7%), `q6-head-simt-threshold` (net-neutral in-round).

### gzenz FORCE-PUSH — T31 pick set invalidated

gzenz force-pushed at 17:30Z: `local/combined` `46275617` → **`5f23c37e`** (rebase/squash, merge-base
`d17a5f1f`). The **5 rewrite-checkpoint commits our T31 port was derived from
(`46275617`/`54ddff5c`/`93144e91`/`fb3ba1b8`/`906847d8`) were DROPPED and re-squashed into
`d00e5f0b`.** Net content is close (22 files, +358/−105), so the port is salvageable, but **T31's
pick set MUST be re-derived from `5f23c37e`** before any further T31 work.

### Other tracked forks

- **mirko**: `feat/kvarn-production` `c853622c` → **`114b0fcb`** (+2 KVaRN commits, incl. an in-window
  greedy-parity fix) — feeds deferred T30.
- **dylan**: `experimental` **`cdd1b6c1` "accelerate C1-4 speculative decode"** — NEW and on-lane
  (27B-NVFP4, our exact C=4 concurrency) → probe candidate (T40). `dylan/qwen4` (13 commits) is a
  different model → off-lane.
- **eason, cometkim**: no in-window commits (cometkim's line now continues as gpillon).

### Upstream PRs/issues — new in window

- **PR #183 + issue #182 (NEW): `--chat-template FILE` operator-managed chat-template override.**
  Upstream is converging on the capability our fork already has as `--chat-template-file` (T2 era).
  Adopt-on-merge and align flag naming; directly relevant to T37.
- **Issue #184 (NEW, 16:34Z): a disconnected streaming client still holds its slot during context
  materialization** — on-lane risk at C=4 (a dropped OWUI/OMP stream can hold 1 of 4 slots).
- T32 cluster (#176–#181, #142) unchanged otherwise; #181 still mirrors our T14 finding.

### Chat-template switch (the operator ask) — T37

Goal: drop the Sharp template and run the model's own default at xhigh. **Verified mechanism: simply
REMOVE `--chat-template-file` from the quadlet Exec.** It is *not* a hard-fail and *not* a generic
compiled default:

- `frontend.cpp:compile_chat_template` (243-251): non-empty path → `compile_jinja(...)` (arbitrary
  jinja via minja, capabilities probed by scanning the text for `reasoning_effort`/`terse`); empty
  path → `CompiledChatTemplate::resolve(resources.chat_template_jinja)`.
- `chat_template.cpp:685-693` `resolve()` sha256-hashes the **artifact-embedded** template and
  accepts exactly two pinned digests: `kThinkingToggleTemplateDigest` or
  `kReasoningEffortTemplateDigest`; anything else throws `unsupported frontend/chat_template.jinja`.
- Our artifact embeds `frontend/chat_template.jinja` = 8,952 B, sha
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` (extracted directly from
  `mirko_quasar.ninfer`) = **`kReasoningEffortTemplateDigest`** = the official Qwen3.8-27B template
  (also on host as `templates/chat_template.qwen-base.bak`). So omitting the flag loads the
  **compiled-in ReasoningEffort semantics** — thinking/tools/media rendered in C++
  (`RenderBuilder` + `append_media_placeholder`), no jinja.
- xhigh is native there: `capabilities()` (708-714) reports `xhigh = true`,
  `default_effort = XHigh`; `preserve_thinking.value_or(effort_template)` (814) keeps our
  `--preserve-thinking` behavior. `validate_tokenizer_config` (218-227) already passes (artifact
  self-consistent: embedded template == `tokenizer_config.json.chat_template`).
- **Correction to the record:** the live template is Sharp **v22.4.0 "froggeric"** (29,063 B, sha
  `180e7015…`), not the v22.1 the quadlet comment and ADOPTION.md:76,322 claim.

Three behavioral deltas to A/B (this is why it is PROBE-FIRST, not a straight flip):
1. **default effort flips medium (Sharp) → xhigh (official)** — longer thinking, more output tokens,
   higher latency on *every* thinking turn regardless of accuracy;
2. the Sharp **`terse` lead + tool-error-escalation** instructions are lost (official has no
   `terse`; the kwarg stays in the server allowlist but is inert on the compiled path);
3. think-block bytes differ (`<think>…` vs `<think>\n…\n\n\n`), though
   `prompt_ends_in_open_reasoning` (1055-1077) matches both, so open-reasoning detection is
   template-agnostic.

Media contracts hold identically (the compiled path is the C++ clone of `append_media_placeholder`
plus the strict count check), so the vision gates should pass — that is what makes this low-risk.

Procedure (config-only; no rebuild, no image retag, rollback = one file + flag):
```
1. cp -a ~/.local/share/ninfer/templates/chat_template.jinja \
         ~/.local/share/ninfer/templates/chat_template.sharp-v22.4.0-live.bak
2. edit ~/.config/containers/systemd/ninfer-nvfp4.container: DELETE
   `--chat-template-file /workspace/templates/chat_template.jinja` from Exec
   (leave the template Volume mount; fix the stale "live file = v22.1" comment)
3. systemctl --user daemon-reload && systemctl --user restart ninfer-nvfp4.service   (~2-3 min reload)
4. curl -s http://127.0.0.1:8002/v1/models  -> 200 with id qwen3.8-27b; boot journal must show
   NO `unsupported frontend/chat_template.jinja` and NO tokenizer_config mismatch
5. ninfer-battery.sh --tag <current-running-image-ref>   -> all 15/16 verdicts PASS, 4XX-WATCH zero
6. xhigh A/B (below). Rollback: restore the .bak and re-add the flag, daemon-reload + restart.
```
A/B method: run one fixed request set under both templates with
`chat_template_kwargs:{"reasoning_effort":"xhigh"}` and compare (a) `reasoning_content` char length
(thinking depth), (b) QUALITY gate 200-word probe word count + terseness (directly exposes the lost
`terse` lead), (c) `probes/decode_tps.py` fresh + 8k vs `logs/quasar-baseline-2026-08-26.json`
(151.9 / 150.2 — guards against a spec-decode/MTP fallback induced by the render change),
(d) `probes/t25c-needle-64k.py` (template-agnostic long-context correctness),
(e) `probes/t31b2-repro.py` / `t31b3-repro.py` as regression sentinels, plus a one-off throwaway
xhigh-correctness script (~10 math/logic/needle questions, score = exact-answer rate + mean
`reasoning_content` length + decode tok/s).
**Decision rule:** adopt the default template only if the A/B shows no accuracy drop at xhigh AND
the full battery is green (esp. XHIGH, VISION-*, REPLAY, 4XX-WATCH); otherwise keep Sharp.

### New tiers

**T33 — DFlash2 drafter grafted onto the QUASAR artifact (PROBE-FIRST, top priority).** Operator
constraint: the lane KEEPS the QUASAR-quantized weights (better accuracy); DFlash2 is added *on top*
as a drafter module. Work: (a) cherry-pick the DFlash2 op layer from `gpillon/coding` — all NEW
files, so conflict-free: `a5064a9d` (two-tap dynamic conv + selector transition scores),
`b5f6a15a` (top-k + selector path walk), `6d962dba`/`92cdac97` (selector swizzle + `[K,P,L]`
lattice addressing fixes), `9990b6b2` (context norm gains are plain `w` — the fix that reproduced
3.4–3.7 tok/round); (b) engine integration `b4087269` (four `[d0,d1]` gather bugs fixed) +
`34a33720` (engine-signature adaptation) + `1953f2f4` (27B `DFlash2Config`, new
`qwen3_6_27b/impl/load/bindings.{h,cpp}`, `speculative_options`, `startup_features`, layouts) +
`a14f0033` (GDN replay records + drafter wide-swiglu split); take `kMaximumDFlash2DraftTokens = 7`
but **NOT** `kNativeContext = 1048576` and **NOT** the hq/hyperquant KV routes; (c) converter: add
DFlash2 tensor specs to our quasar inventory and a graft variant that validates
`qwen3.8-27b`/`nvfp4` + recipe `qwen3_8_27b_quasar_nvfp4-v2` instead of `nvfp4full`, mirroring
`6b10d57e` + `41231998`; (d) fetch drafter `z-lab/Qwen3.8-27B-DFlash2`, encode the module, graft onto
a COPY of `mirko_quasar.ninfer` (never in place), verify sha + object count 1,259+66. Gates:
**acceptance rate (drafts/round) measured at 1.5K / 8K / 32K** — must beat MTP3's observed
~65% of 3 (~1.95 accepted/round) and must not collapse with context the way ninfer-fusion's DFlash2
did (24.7%→12.7% by 12K); decode A/B vs 151.9/150.2; greedy/verification parity (T7 lesson);
full battery 16/16; INT8 KV retained. Kill switch: `--spec mtp` still works on a grafted artifact,
and the module costs no VRAM unless `--spec dflash2` is selected.

**T34 — host-KV restore correctness (reframes T31).** gpillon's `f4b128c6` shows the packed
rewrite-checkpoint restore from host RAM serves one request with another's state — silent
contamination on the same path T31 enables. **Therefore T31 must NOT relax the completed-prefill
frontier invariant; our Bug 3 fatal is plausibly that invariant correctly refusing a corrupt
restore.** Adopt the shape of their mitigation (offer host-RAM records for append-at-frontier reuse
only; leave the VRAM-resident checkpoint restore path untouched) and port `ac60331d`
(MTP RAM-vs-VRAM restore comparison test) as a permanent guard. Expected cost ≈8% of restores fall
back to `full_reset`. Skip `eaf2037b` (hyperquant, off-lane).

**T35 — draft-window k=3→5 (zero-code probe).** md measures +17.3% with *higher* acceptance at k=5.
Our lane pins `--draft-tokens 3`. Probe: serve-flag A/B at 3 vs 5 (and the `6870d530` 15-token-window
enabler if the ceiling blocks it), gated on acceptance/round + decode fresh/8k + battery.
Cheapest plausible decode win in this round.
**RESULT (2026-09-06, PROBE COMPLETE — REVERTED to k=3).** k=5 ran 01:26–09:08 CEST: quiet
battery 16/16 PASS (all gates green, 4XX zero-reflections); decode fresh 152.5 tok/s vs k=3
140.6 (+8.5%); decode 8k 144.8 vs k=3 155.4 (−6.8%); acceptance fresh 2.1–3.5/round (k=3 ~1.9)
but 8k only 1.3–1.6/round (k=3 long-ctx held ~1.7–2.1). Plan rule: adopt only if acceptance AND
both decode numbers improve AND battery green — 8k decode did not improve → REVERT. Reverted in
the 09-06 09:08 restart (same restart applied the host-KV arena bump, operational note below).
k=3 baseline re-confirmed live: fresh 140.6 / 8k 155.4 — the T33/T36 comparison baseline stays on
k=3.

**T36 — md dense-lane ops wave.** `8767dac7` decode-softmax-fold; `c735909b` + `16c66809` nvfp4-TMA
prefill extensions (27B-nvfp4-measured, extends T23); `ce71f787` MTP sampled-draft (probe).

**T36 build record (2026-09-06) — wave ready, probe absorbed.** Wave branch `t36-mdops-quasar` tip
`29e628a7` (pushed to `gevil`), five commits on `533e93fc`:
- `29e628a7` = re-pick of `8767dac7` (split-KV decode softmax takes the prompt kernel's arithmetic)
- `e96df2c9` = re-pick of `c735909b` (TMA grid walked in token groups)
- `0243d3db` = re-pick of `ce71f787` (MTP sampled-draft; 18 files, +436/−52). Conflict resolutions:
  kept our `CausalAttentionExecutionEnvelope` + state-source/destination-slot plumbing (the merged
  `TargetVerifyFrameView` and `target_verify_accept` use them; md's `lanes` field dropped), kept our
  `increment_token_counts` blocks in the sampler, took md's `out_prob`/support plumbing and his
  comments.
- `cbf8a72d`, `d61a8489` = ops docs (host-KV arena note; T35 revert + arena 8→32 GiB record).
- `16c66809` content (tile-continuous activation scales) already present in base `533e93fc`
  (symbol check: `nvfp4_make_tma_2d` / `nvfp4_tma_load_2d` / `nvfp4_w4a4_tma_route` counts equal in
  base and wave) — its pick is a no-op, not re-picked.
- **Probe absorbed:** `ce71f787` arrived inside the wave as `0243d3db` (the earlier wave build pulled
  it in), so a separate probe branch was a 2-line no-op (created, verified empty against the wave,
  deleted). Consequences: (a) the sampled-draft behavior ships WITH the wave image — no longer
  independently rejectable at pick level; if the post-ship acceptance gate fails, the revert is a
  `0243d3db`-removal rebuild, not a pick drop. (b) The acceptance-rate gate is unchanged and is
  measured on the post-ship image: MTP3 with draft head, acceptance drafts/round at 1.5K/8K/32K
  vs the MTP3 baseline, decode A/B vs 140.6/155.4, greedy parity, battery 16/16, INT8 KV retained.
- G4-equivalent host build (buildstage-merge container, python3 pre-installed per G4 style): PASS
  2026-09-06 (fresh /build, `BUILD_TESTING=ON`, full cmake configure + build to 100%, all test
  binaries linked, ~4.5 min). Ship additionally gated on a quiet window (user decision).

**T37 — chat-template switch to the artifact-embedded ReasoningEffort template at xhigh
(PROBE-FIRST).** Full plan above. Config-only, engine-native, one-file rollback.

**T37 result (2026-09-06) — ADOPTED.** The switch executed 2026-09-05 21:42 CEST (quadlet:
`--chat-template-file` removed; the Sharp v22.4.0 file kept as `chat_template.sharp-v22.4.0-live.bak`
and the mount retained for one-line rollback). Live engine verified on the artifact-embedded
template (no flag in the container's `/proc/1/cmdline`; `engine ready`; no
`unsupported frontend/chat_template.jinja`; xhigh kwargs accepted).

- **A (Sharp v22.4.0, pre-switch 21:40):** decode 139.2 fresh / 149.8 @8k; xhigh 10-question
  accuracy 10/10; quality 238 words; total thinking 2366 chars.
- **B (embedded @xhigh):** accuracy 10/10; quality 216 words; thinking 2935 chars (native xhigh
  default); 64k needle FOUND; quiet-window decode 140.6 fresh / 155.4 @8k (A/B ratio 1.01 / 1.04 —
  render change is decode-neutral).
- **Contamination correction:** the 21:42 battery (14/16: DECODE-FRESH/8K fail) and the 22:06 B
  decode capture (14.8 / 41.0 tok/s) were contention artifacts — this session's own OMP requests
  ran on the same lane during those windows (the T31 pattern). The stored script's
  `RECOMMEND: KEEP-SHARP` line was driven solely by those numbers and is invalidated.
- **Quiet-window re-battery (2026-09-06 00:43, `pipeline/logs/battery-2026-09-06.log`): 15/16** —
  all functional gates PASS: DECODE-FRESH 166.0 tok/s (gate 132.4), DECODE-8K 149.3 tok/s
  (gate 130.5), REPLAY 10/10, VISION ×3, XHIGH, THINK-SMOKE, QUALITY, SOAK 5/5, 4XX-WATCH zero
  rejections. Sole FAIL = LEDGER ("no engine-ready marker in boot ledger window") — a window
  artifact only: the boot (21:42) predates the battery's ledger window by ~3 h; UP/WARMUP/
  4XX-WATCH confirm the engine healthy.
- **Decision (plan 1f): adopt** — accuracy ≥ Sharp, decode within gates, battery functionally
  green, native xhigh, no external template file. New decode baseline on this render path
  (record only — the ship pipeline's `quasar-baseline-2026-08-26.json` is NOT overwritten and
  remains the `--refresh-baseline` target): **fresh 140.6 / 8k 155.4 tok/s** (2026-09-06, quiet,
  battery method: coastal-erosion, thinking off, max 512). Later tiers gate decode against
  these numbers on the embedded-template render path.

**T38 — upstream chat-template override + streaming-slot watch.** PR #183 / issue #182
(`--chat-template FILE`): adopt-on-merge and reconcile our `--chat-template-file` naming; issue
#184 (disconnected stream holds a slot during materialization): on-lane at C=4 — adopt the upstream
fix when it lands, or reproduce if we see slot starvation.

**T39 — Astrangemaninhere/ninfer-fusion (WATCH).** Sub-floor KV REJECT (perplexity-only evidence);
its 27B DFlash v1 profile needs a new artifact and its own data says DFlash2 < MTP3 — superseded by
T33's graft-onto-QUASAR approach. Re-check only if it publishes E2E quality for the KV tiers.

**T40 — dylan `cdd1b6c1` "accelerate C1-4 speculative decode" (PROBE).** On-lane (27B NVFP4, C=4).
Verify portability (dylan's line historically carries 35B-only dflash assumptions), then decode A/B.

### Status changes this round

- **T28 → SUPERSEDED by T33**: the blocker is not "no 27B dflash artifact" but that DFlash **v1** is
  permanently off the 27B target; DFlash**2** is a separate module and is graftable onto our own
  artifact. dylan's dflash2 work stays non-portable.
- **T31 → pick set INVALID (gzenz force-push, re-derive from `5f23c37e`) and approach REFRAMED by
  T34** (do not relax the frontier check; restrict host-RAM reuse instead).
- **T30 → still deferred**, but `114b0fcb` adds a greedy-parity fix worth reading if T30 revives.
- **T32 → extended**: #184 added to the cluster watch; #183/#182 split out as T38.
- **T8 → CLOSED, lineage note**: the cometkim line now continues as gpillon and is the T33 source.

### TIERED PLAN (re-evaluated 2026-09-05, round 6)

| Tier | What | Status (2026-09-05 r6) |
|---|---|---|
| 1 | decode & serve quality (#55 #67 #69 #65 #57 #61) | DONE 08-23 (`tier1`, PR #1) |
| 2 | xhigh track (Sharp v22.3.1 + `reasoning_effort` kwargs) | DONE 08-23 (`tier2`, PR #2) |
| 3 | community cherry-picks + Wave B1 decode perf | DONE 08-24 (`tier3`/`tier3-waveb`) |
| 4 | upstream convergence + agent-workload (C1–C3, host-KV opt-in) | DONE 08-24 (`tier4`) |
| 5 | Wave C (MoE-decode perf, response_format json, Sharp v22.3.2) | DONE 08-24 (`tier5`) |
| 6 | portability (SM-count persistent grids) | DONE 08-25 (`tier6`) |
| 7 | MTP width-invariant greedy verification | DONE 08-26 (re-adopted; first adopt reverted, −50% decode) |
| 8 | cometkim DFlash2 probe | CLOSED 08-31 (4 boot bugs) — lineage continues as gpillon → **T33** |
| 9 | dylan XAttention prefill | DEFERRED — prefill/TTFT evidence first |
| 10 | upstream/dev sync | RESOLVED 08-29 (consumed by T12) |
| 11 | quasar re-verification on the T10 merge | CLOSED — rolled back (vision 400) |
| 12 | upstream convergence wave | DONE 08-30 (battery 16/16) |
| 13 | #98 wave (#107/#97/#72 + pressure fixes) | DEFERRED — `3d9fda22`/`5e4bf313` 503-bad under load |
| 14 | host-KV content cache (#73) | PROBE DONE 09-05 (idle prefix loss @45s/120s) → **T31** |
| 15 | gzenz NVFP4 KV + YaRN (400k ctx) | IN TREE 09-02, **not live** (quadlet pins INT8 @225,280) |
| 16 | upstream convergence wave 1 | DONE 09-02 |
| 17 | pv-f16acc (md) | DONE 09-02; T25c 64k-needle PASS 09-04 |
| 18 | dylan wave 2 (GDN chunked-prefill precision) | SHIPPED 09-05 — **live lane** (`e858f88b`) |
| 19 | `gated_delta_net_snapshot` op + tests (dylan) | NOT STARTED — no new signal |
| 20 | watch: open upstream PRs | WATCH — #183 NEW; #167/#160 in-tree; #163/#162/#152/#148/#107/#97 open |
| 21 | watch: upstream issues | WATCH — #184 NEW (stream holds slot); #164/#166/#168/#169/#142 unchanged |
| 22 | upstream convergence wave 2 | DONE 09-04/05 (landed via the T-Q ship) |
| 23 | md TMA prefill pair (#167 + #160) | SHIPPED 09-05 (`t23tma-bb535075`) — extended by **T36** |
| 24 | gzenz host-KV safety net | SUPERSEDED → T31 |
| 25 | probe wave | PARTIAL — T25c PASS, T14 done; remainder open |
| 26 | watch slot | WATCH — upstream-PR watch set (re-mapped) |
| 27 | watch slot | WATCH — community-fork watch set (re-mapped; +gpillon, +ninfer-fusion) |
| 28 | dylan dflash2 wave | **SUPERSEDED by T33** — DFlash v1 is permanently off the 27B target |
| 29 | Mirko dynamic-MTP decode wave | DECIDED 09-05 — T29a port pending; T29b levers no-win |
| 30 | Mirko KVaRN line | DEFERRED — `114b0fcb` adds a greedy-parity fix if revived |
| 31 | gzenz host-KV safety-net port | **BLOCKED** — 2 bugs (B2 entitlement, B3 frontier); pick set INVALID (re-derive from `5f23c37e`); approach reframed by **T34** |
| 32 | upstream prefix/context-cache cluster (#176–#181, #142) | WATCH — +#184; #181 mirrors our T14 finding |
| 33 | **DFlash2 drafter grafted onto the QUASAR artifact** | **NEW — PROBE-FIRST, top priority.** Engine port from `gpillon/coding` + quasar-side graft using `z-lab/Qwen3.8-27B-DFlash2`; keeps QUASAR weights; gate on acceptance/round @1.5K/8K/32K |
| 34 | **host-KV restore correctness (reframes T31)** | **NEW — ADOPT the mitigation shape**: host-RAM reuse = append-at-frontier only; do NOT relax the frontier invariant; port `ac60331d` as a guard |
| 35 | **draft window k=3→5** | **REVERTED 09-06** — probe complete: battery 16/16, fresh +8.5% but 8k −6.8% + long-ctx acceptance degraded → plan rule: revert to k=3 (baseline frozen for T33/T36) |
| 36 | **md dense-lane ops wave** | **NEW — PORT-CANDIDATE**: `8767dac7` decode-softmax-fold; `c735909b`+`16c66809` nvfp4-TMA (27B-measured); `ce71f787` sampled-draft probe |
| 37 | **chat template → artifact-embedded ReasoningEffort @xhigh** | **ADOPTED 09-06** — live since 09-05 21:42; quiet battery 15/16 (LEDGER window artifact only); decode-neutral vs Sharp (140.6/155.4); new render-path decode baseline recorded |
| 38 | **upstream `--chat-template FILE` (#183/#182) + stream-slot (#184)** | **NEW — WATCH/adopt-on-merge**; reconcile flag naming with our `--chat-template-file` |
| 39 | **Astrangemaninhere/ninfer-fusion** | **NEW — WATCH**; sub-floor KV REJECT (perplexity-only); its DFlash2 < MTP3 by its own data |
| 40 | **dylan `cdd1b6c1` C1-4 speculative decode** | **NEW — PROBE**: on-lane 27B NVFP4 at our exact C=4 |
| 41 | **wall-time-to-accurate-answer (T2A) research & plan** (W0–W6; supersedes the token-ranking) | **NEW 09-06** — research complete on the live lane (P0–P6 sequencing); harness-side levers (W0/W1/W2/W3/E/W4) + engine-side half = the T33 gpillon cluster |

**Sequencing (round 6).** (1) **T37** chat-template switch — config-only, no build, immediate operator
value, and it must settle BEFORE T33/T35 so decode A/Bs are measured against a stable render path. — **settled 2026-09-06: ADOPTED** (result block above).
(2) **T35** draft-window probe — zero code, largest cheap decode upside. (3) **T33** DFlash2-on-QUASAR
— the substantial wave; engine port + converter graft + acceptance-gated probe. (4) **T34** fold into
T31 before any T31 re-ship (and re-derive the pick set from `5f23c37e`). (5) **T36** md dense ops
wave. (6) **T40** dylan spec-decode probe. (7) **T38**/T32/T39 watch. Deferred: T29a, T30, T9, T13,
T19, T15 config flip.

## Operational event 2026-09-06: host-KV arena 8 → 32 GiB + T35 revert (single restart 09:08 CEST)

- **Context.** User symptom: one ~178k-token OMP session, every turn full re-prefill — journal:
  `cache 0 (0.0%)` on ~178–180k prompts, TTFT 1m8s–1m48s, "host" column tiny — while warm
  follow-ups hit 95–100% cache (req#506 95.5% in the k=3 era; post-boot req#4 99.8%, req#83
  100.0% at TTFT 4.3s). NOT a T35/k=5 regression: the same `cache 0` re-prefills appear in the
  k=3-era journal (00:50–00:57, req#501–507). Arena semantics verified in the T18 source:
  `--host-kv-mib` is ONE byte pool (`host_kv_capacity_bytes = mib << 20`, serve_options.cpp:239)
  with a state-count cap of 8 ("host 8 states" is that cap — identical at fresh boot, not fixed
  slabs); 8 GiB ≈ 2× one 2.8 GiB (178k-token) state, so single-session fit is NOT the issue.
- **Change.** Quadlet `--host-kv-mib 8192 → 32768` (user request: was 32G before; MemAvailable
  32 GiB pre-pin → 10 GiB post-pin) + `--draft-tokens 5 → 3` (T35 revert, above). Backup:
  `ninfer-nvfp4.container.bak-2026-09-06-hostkv32g`. Boot 09:08:26 CEST: `host KV pinned |
  32.0 GiB | 13.4s`, `host 8 states, 32.0 GiB KV`. `/v1/models` 200; post-restart follow-ups
  already hit 99.8–100% cache (TTFT 4.3–7.9s).
- **Root cause: UNCONFIRMED.** Candidates: (a) multi-generation / parked-lane arena contention at
  8 GiB; (b) restore-path fallback — T31/T34 class (host-RAM rewrite-checkpoint restore is the
  documented broken path upstream); (c) state-identity mismatch (this session's requests
  alternate 167-message / 1-message forms — different rendered prefixes can't match); (d) T14
  idle expiry before parking. Decisive test: this session's NEXT turn after ≥1 min idle — expect
  high cache % + TTFT in seconds. NOTE: this build logs no park/restore events at info level
  (zero such lines in the journal since boot) — the observables are the done-line
  `cache N (X%, private endpoint)` field and TTFT. If re-prefill persists at 32 GiB → (b)/(c):
  escalate to T31/T34. ALSO watch 500s on vision turns (StateImage-entitlement class, the 09-05
  Bug 2 500): the bigger arena raises restore frequency and this session carries 5 vision media.
- **Corruption-exposure note (plan step 5a):** T18's restore path validates entries (corrupt entry
  → loud `ResidentLost` → full reset; frontier-entitlement invariant rejects loudly — the 09-05
  Bug 3 fatal is that rejection) and ran at 8 GiB since 09-05 with no silent corruption; the
  bump changes capacity, not validation semantics.

## t36mdops-29e628a7 ship (2026-09-06) - t36-mdops-quasar @ 29e628a7

Image `eaa60d8cb205` (tags: `t36mdops-29e628a7`, :quasar, :latest); previous
`:quasar` `e858f88b907e` retained as rollback target.
- Free-GPU ctest: rc=0, skips within baseline (6 expected).
- Battery: 16 PASS / 0 FAIL: VERDICT UP: PASS VERDICT IMAGE: PASS VERDICT MODELS: PASS VERDICT LEDGER: PASS VERDICT WARMUP: PASS VERDICT VISION: PASS VERDICT VISION-HIST: PASS VERDICT VISION-POISONED: PASS VERDICT REPLAY: PASS VERDICT THINK-SMOKE: PASS VERDICT XHIGH: PASS VERDICT DECODE-FRESH: PASS VERDICT DECODE-8K: PASS VERDICT QUALITY: PASS VERDICT SOAK: PASS VERDICT 4XX-WATCH: PASS
- State: lane `ninfer-nvfp4` runs the new image; :quasar/:latest pinned (verified match).

## t36mdops-29e628a7 REGRESSION — rolled back 11:37 CEST (supersedes the "shipped" entry above)

- **Verdict: REGRESSED, rolled back.** The 16/16 battery result is a battery-vs-real-traffic
  gap, not a ship PASS. The image failed under real traffic at 09:28:55 UTC (11:28:55 CEST):
  engine fatal `materialization preparation state is invalid` on a 1.45M-token xhigh vision
  request with a 137,887-token compacted prefix (req#26, 110-message session, 14,487-token
  prefix match). All subsequent requests 503'd (`failed during prepare`) until the user's
  manual restart at 11:29; the engine kept running and serving after the restart (health
  endpoint 200), but the fatal class could recur on the same traffic shape, so the lane was
  rolled back to `e858f88b` (t18gdn-49400365) at 11:37.
- **Root cause (evidence-backed, hypothesis pending T31/T34 resolution).** The wave's base is
  `t31-hostkv-quasar` @ `533e93fc` — the T31 branch, i.e. T31's **unshipped, unvalidated
  host-KV safety-net code was carried into the wave base and shipped with it** (base-selection
  error; the plan assumed a pure T18-lineage lane branch). The fatal site
  (`program_impl.h` `materialization preparation state is invalid`) **pre-dates the wave** —
  verified present in the T18 image source (`49400365`), so the invariant check itself is not
  new; what is new is the T31 machinery that routes this traffic into it: the
  `[compact-prefix]`/`[safety-spill]` path. On the 1.45M-token request the spill was SKIPPED
  (`no endpoint state image`, `[shortlist] HIT` reuse 137,887 / frontier 141,480 /
  ckpt_frontier=0), and the materialization of the 141,480 frontier then tripped the
  pre-existing invariant → fatal. T18 has no compact-prefix path at all, so this traffic
  shape could not reach the fatal there (it would full-reprefill or be rejected at
  max-context 225,280). This is the third sibling fatal of the T31 Bug 2/3 family
  (`sequence StateImage entitlement is inconsistent`, `completed prefill did not reach the
  admitted prompt frontier`) — the safety net has at least three fatal checks in one class,
  and real 1.4M-token compacted-prefix vision traffic exercises a fourth shape the battery
  never sends (battery VISION = 5 media at most; VISION-HIST is single-shot).
- **Corrections to earlier records (this session):**
  - `2b4c050b`/`51e50654` claim "16c66809 already in base": **wrong for the TMA scale file.**
    File-level diff (`git diff 16c66809 t36-mdops-quasar -- src/ops/linear_swiglu/nvfp4/`): the
    wave's `nvfp4_linear_swiglu_u444_tma.{cu,cuh}` do not contain md's post-change
    tile-contiguous-scale content; the wave files instead hold pre-existing WIN32 portability
    code. 16c66809's functionality is **absent from the wave** (a ~25-line TMA micro-optim).
    No correctness impact; the wave's measured prefill gain (-2.7%..-3.7% at exact-27B) is
    therefore partly unaccounted-for. Consequence for any re-wave: re-pick 16c66809 properly.
  - The earlier "16c66809 no-op" symbol check only covered `src/ops/linear/nvfp4/` (a
    different, not-shipped file set) and was insufficient.
- **What the user experienced:** "current version is unstable and broken" — the 503 storm on
  trivial 2-message requests after the 09:28 fatal (engine process stayed up but failing
  everything), persisting through the 11:29 manual restart because the restart re-launched
  the **same** image (all three tags pointed at `eaa60d8c`; the restart was container-only,
  not a rollback).
- **State after rollback:** `:quasar`/`:latest` → `e858f88b907e` (T18); `t36mdops-29e628a7`
  tag retained on `eaa60d8c` for forensics. Verified: image id match, `/v1/models` 200,
  chat probe 200 (118.7 tok/s decode), unit active post daemon-reload.
- **Next (before re-attempting md's perf wave):**
  1. Re-wave from a base **without** T31's in-tree work (e.g. T18 line `t24-quasar-a`
     + the md picks), or first fix T31 Bug 2/3 (fold into T34, pick set re-derived from
     gzenz `5f23c37e` + gpillon `f4b128c6`).
  2. Add a battery gate for the compacted-prefix / huge-context vision shape (≥1.4M-token
     xhigh vision request, or at minimum a 100k+ token compacted-prefix restore probe) —
     the battery's VISION/VISION-HIST shapes cannot expose it.
  3. Re-pick 16c66809 (conflict-resolved, not `--3way -X theirs`-assumed-absent).
  4. The 4XX-WATCH gate remains blind to 5xx/engine-fatal (T31-era gap, still open):
     a mid-battery fatal must fail the battery, not pass.

### gzenz StateImage fix found (2026-09-06, answers the user's "saw a StateImage fix yesterday" memory)

- **`gzenz/fix/checkpoint-stateimage`** @ `1b11452c` (2026-09-05 17:30), already merged to
  `gzenz/master` as **PR #1** (`c17de1cc`, 2026-09-06 11:10). Exactly the fatal class that
  broke the T36 wave; NOT in our tree or the live image (ancestry-verified; no remote commit
  touched the fatal string itself — pickaxe-verified — the hardening is behavioral):
  1. **Worker recovery**: `catch(const std::logic_error&)` in the engine worker loop
     (`engine_core.h` ~2023, above the existing OOM catch) — converts this fatal class
     ("stale checkpoint state image" / `materialization preparation state is invalid`) into
     per-request failure + `recover_from_oom_locked` worker recovery, with a
     `kOomMaxRecoveries` livelock guard, instead of `fail_all` + 503 storm.
  2. **Slot-budget guard** (`program_impl.h` rewrite-restore path): if the new
     `rewrite_state` would exceed `state_slots`, release it and proceed without a rewrite
     checkpoint for the turn (drops the checkpoint rather than fatale).
  3. **Restore-after-consume**: removes the identity check that blocked rewrite restore
     after consume; allows restore when the endpoint is evicted; removes the
     `!rewrite_checkpoint` guard so the checkpoint tracks the latest turn boundary
     (`chat_template.cpp`).
  4. 8-phase E2e test suite (checkpoint-advance, tool-calling restore, responses-tools,
     reasoning-effort).
- **Also confirmed by the scan:** gzenz `d00e5f0b` (our T31 port's re-squashed base) contains
  the exact `[safety-spill] SKIP: no endpoint state image` / `SKIP guard` conditions seen in
  the 11:28 crash log — the spill-SKIP fallthrough into the pre-existing fatal is the
  wave's exposure. Upstream StateImage fixes `b8786751`/`da49c0d6`/`a140e7ae` are already in
  the live image tree (`49400365`); dylan/md/eason/cometkim/gpillon have no
  StateImage-relevant commits in the window.
- **Adoption implication:** when T31/T34 is re-derived, base it on gzenz `d00e5f0b` **+
  `1b11452c`** (worker recovery catch + slot guard + restore-after-consume). With the
  worker catch, this fatal class degrades to a per-request failure instead of a lane-wide
  503 storm — the difference between "unstable and broken" and "one request failed".

## Round 7 (2026-09-06, ~14:25 CEST) — re-audit (gzenz delta), gpillon fork audit, T31 wave PAUSED, T41 wall-time tier

Window: 2026-09-06 11:37 CEST (t36 rollback) → 14:25 CEST. Highest tier before this round:
T40 → new tier T41.

### Re-audit deltas

- **gzenz — 1 new commit** since the t31rev wave was built (the wave picked the `0d4ee9b7` squash):
  `1d2497c9` "wrap bad_alloc retry in try-catch, fail ONE request instead of nuking"
  (`program_impl.h`, +19/−8, 13:55 CEST today, on `fix/materialization-root-fallback`). Wraps the
  OOM-retry in the materialization path in a try-catch: if the retry also OOMs (device state slots
  exhausted), `abort_transaction()` → `ContextTransactionStatus::Aborted`, failing the ONE request
  instead of nuking the worker. Pairs with the worker-recovery catch (`1b11452c`): it closes the
  remaining `bad_alloc` escape on the restore path. **`t31rev-quasar` is now one commit behind**
  gzenz's fallback line — pick `1d2497c9` before re-deriving/re-shipping.
- **gzenz/master:** unchanged (`c17de1cc`) — the new work is on the fix branch, not master.
- **upstream:** `ad0f3d38` (funding-info chore, docs-only) — the only new commit since round 6.
- **gpillon:** no new commits since round 6 (tips still `a00648cb` `gpillon/coding`, `43b03ea5`
  `feat/dflash2-local`).
- **cometkim/dylan/eason/md/mirko:** no on-lane movement this window.

### Lane state (live, after operator restart 14:06 CEST)

- Image `e858f88b` (`t18gdn-49400365`, T18-gdn); engine ready 17.8 s; `/v1/models` 200.
- Quadlet: artifact-embedded template @xhigh (T37; no `--chat-template-file`), `--spec mtp
  --draft-tokens 3` (T35 reverted), `--host-kv-mib 32768` (09:08 arena bump), INT8 KV 225,280, C=4,
  vision, preserve-thinking.
- Decode baseline (live render path, k=3, quiet): **140.6 fresh / 155.4 @8k tok/s**.

### T31/T34 — wave built + gated, NOT shipped — **PAUSED (operator directive 09-06)**

- `t31rev-quasar` @ `54b120ef` = 3 re-derived picks on the T18 tip `49400365`: `0b0ac916` (host-KV
  safety-net unit, from `d00e5f0b`), `db47e3fc` (checkpoint advance / restore-after-consume /
  worker recovery, from `1b11452c`), `d0807ece` (materialization root-fallback + host-KV e2e
  phases 1–9, from `0d4ee9b7`; gzenz's BLAKE3 signature-verification subsystem deliberately NOT
  ported — our tree's opaque-signature design is preserved; e2e phase 10 dropped with it)
  + `54b120ef` docs (`THINKING_WALLTIME_PLAN.md`).
- Gates passed pre-pause: conflict resolution clean (zero markers), G2-equivalent build PASS,
  ctest rc=0 within baseline, quiet-window battery green on the T18 image, host-KV e2e phases 1–9
  ported and passing in-container.
- **Why paused:** gzenz keeps landing fixes (09-05 17:30Z force-push, the 4-commit squash into
  `0d4ee9b7`, and `1d2497c9` today). Shipping now risks another re-derivation cycle; the stable
  T18 lane stays live and untouched.
- **On resume:** (1) pick `1d2497c9` onto `t31rev-quasar`; (2) re-check gzenz branch movement;
  (3) rebuild + free-GPU ctest + quiet-window battery + compacted-prefix probe; (4) ship via the
  supervised pipeline (G2-safe gate, non-lane shipwatch supervisor).
- Note: this wave sits on the **T18 tip**, not on the t36-mdops base — deliberate. The T36
  regression came from the wave base carrying T31's unshipped host-KV code (base-selection error);
  this wave isolates the T31 fix set. The T36 md picks remain separately evaluable in
  `t36-mdops-quasar` (rolled-back image `eaa60d8c` retained).

### T41 — wall-time-to-accurate-answer (T2A) research & plan — **NEW TIER**

Objective shift (operator): optimize **wall time to an accurate answer**, not thinking tokens.
`T2A = queue + prefill·(1−cache_hit) + thinking_tokens/decode + P(zero-yield)·retry`.
Measured on the live lane (n=244 OMP turns + journal, 09-06): the enemy is (a) re-prefill on cache
miss (140k prompt: 44 s miss vs 0.3 s hit), (b) queueing behind any other consumer (one effective
slot; waits 36–90 s), and only then (c) thinking tokens (41–161 on constrained tasks; 17–18k on
hard open-ended, 84–152 s). Full data, method warnings (never benchmark on the interactive lane;
nonce every A/B prompt — response replay is real), and primary sources:
`THINKING_WALLTIME_PLAN.md` (committed on this line this round; supersedes the old
"reduce overthinking" token-ranking in this file).

| # | Lever | Class | Expected |
|---|---|---|---|
| W0 | lane exclusivity for the interactive session (bench/subagent fan-out off-lane or to cloud models) | ops policy | removes the 36–90 s queue term — largest single win |
| W1 | prefix-cache hit engineering (immutable→volatile prompt ordering; kill/coarsen head clocks; ≥0.95 cache gate) | prompt ordering | 44 s → 0.3 s on long turns, zero model risk |
| W2 | context budget vs KV pool (pin one 140k session; compaction, subagent offload, per-project MCP gating; answer the host-KV restore question) | config+hygiene | protects W1; lifts prefill rate + MTP acceptance |
| W3 | thinking-shape stop-rule prompt guidance (no caps; xhigh held) | prompt layer | 17k → 6–9k thinking tokens on the hard slice |
| E | one-clause xhigh template edit ("consider alternatives where a check has failed") | template (full battery + canary) | targets the enumeration tail |
| W4 | zero-yield insurance: measure P(empty) at n≥30 first; adopt `frequency_penalty 0.3` only if non-zero | sampler (flagged as such) | removes the retry tail |
| W6 | MTP/draft sweep on thinking-heavy traces (acceptance collapses to 40.7% on 5.9k-token thinking turns) | spec decode | −20–30 s on hard turns, zero accuracy risk |

Red lines: no hard reasoning-token caps; no temperature change; no effort downgrade; nothing
adopted at n<30 / easy-slice-only / medium-effort / llama.cpp-only evidence; never benchmark on
the interactive lane; nonce every A/B prompt. Sequencing: P0 log-parser/dashboard → P1 W0+W1 →
P2 W2 → P3 W6 → P4 W4 → P5 W3 then E → P6 TRS/template-family (only if hard-slice thinking still
dominates).

**Convergence with T33 (gpillon):** gpillon's agentic cluster attacks the same T2A terms on the
engine side — host-RAM KV tier with active-lane sibling sharing (W1/W2), tagged request lanes
`@main`/`@agents`/`@classifier` (W0), adaptive MTP verification width (W6). T41's harness-side
levers (W1 ordering, W2 compaction, W3 stop-rules) and T33's engine-side cluster are the two halves
of the same plan.

### gpillon/ninfer — fork-changes.md audit (read in full, 09-06)

Driving problem per the maintainer doc: **agentic coding traffic latency** — subagent bursts
sharing most of the prompt serialize on a single prefill lane, producing multi-second TTFT
ladders. The cluster (T33 source; cherry-pick only, never merge — diverged base):

- **Host-RAM KV tier**: finished/active GPU lanes snapshotted to pinned RAM so later requests
  restore instead of re-prefill. Two-tier probation/protected eviction (content lineage = hash of
  leading tokens; a successful restore marks the lineage hot). Active-lane sharing for identical
  concurrent requests (≥2048-token match; one snapshot serves every sibling in the burst — claim
  tracking instead of a pinned flag; per-request remembered snapshot bases). System+tools
  shared-prefix boundary capture (split at the first user message, 4096-byte-block aligned, no new
  state slot). `KVRamCache::capture_bytes()` is the single sizing authority (preflight can never
  disagree with the capture). `PreserveExisting` admission mode fails cleanly instead of evicting
  (checked against real allocator first-fit span geometry, not aggregate free bytes).
- **Two silent-corruption fixes** (root-caused from symptoms, not patched around):
  (a) rewrite-checkpoint restores from host RAM answered one request with another request's state
  — the T34 source; mitigated by offering host-RAM records for append-at-frontier reuse only; the
  underlying packed-checkpoint defect is still open (next suspects `pack_slot_to_host` /
  `unpack_slot_from_host`); (b) hyperquant side-store restore corruption — off-lane for us (INT8 KV).
- **Tagged request lanes** (`@main`/`@agents`/`@classifier`): a trailing suffix on the wire model
  id becomes a `RequestClass`; `@main`-owned retained lanes are ranked LAST among eviction victims
  (reorder, never exclude) so short-lived agent/classifier bursts recycle younger lanes first
  instead of evicting the long-lived main conversation; untagged traffic = `Agents`, schedules
  exactly as today. Log schema v14 (`class=` field).
- **Adaptive MTP verification width**: batch-stable window from request-local accepted-prefix
  survival (sustained score + fresh tail evidence, to avoid widening churn on mixed streams), EWMA
  per-(batch-size, ctx-band, width) round-cost calibration, K5–K8, exact-width CUDA Graph /
  ReplaySSM state preserved. Verified against production traces where a flat cost curve left the
  selector unable to narrow with acceptance swinging 40–96%.
- **Serve robustness**: tool-call XML leak (malformed/unclosed `<tool_call>` remainder was appended
  to visible content raw; a second bug replayed it on stream teardown → content-length invariant
  crash in tool-capable streaming); warmup decoupled from `--pending-timeout-ms` + fail-fast
  (already in-tree via #79 — verify on merge); **100%-CPU decode fix** (`cudaDeviceScheduleAuto`
  resolves to spin on this box, so per-round `synchronize()` busy-waited a host core; now
  `cudaDeviceScheduleBlockingSync` at context creation + event-routed waits).
- **Off-lane**: 1M-context envelope, hyperquant KV, Windows/laptop-GPU portability.

**Operator directive (09-06): adoption starts with the gpillon fork** — the biggest win for
agentic coding on this lane. T33 re-scoped as the entry point: the agentic-serving cluster
(host-RAM KV tier + tagged lanes + adaptive MTP + tool-call/warmup/CPU fixes) alongside the
DFlash2 drafter graft onto QUASAR (hard constraint unchanged: the lane keeps QUASAR weights;
spec decoding is verified, so the drafter costs acceptance rate, not quality). DFlash2's
acceptance gate is unchanged: beat the incumbent at 1.5K/8K/32K, greedy/verification parity,
battery 16/16. The cluster's corruption-fix half is T34's guard shape, so the two tiers land
together.

### Correction — t36mdops REGRESSION entry figures (journal-verified)

The t36mdops REGRESSION entry (above) says "a 1.45M-token xhigh vision request with a 137,887-token
compacted prefix (… 14,487-token prefix match) … frontier 141,480". Journal ground truth (09-06
11:28:55 CEST, req#26, 110-message session, 5 media, 14 tools):
`[prefix-match] SIZE FAIL: count=141487 prompt=145367 resident=137887`. Correct figures:
**prompt 145,367 tokens; frontier/count 141,487; resident prefix 137,887** (the last one was already
correct in the entry). "1.45M" was a 10× digit-transcription error; "141,480" and "14,487" were
typos of 141,487. The trigger was a 110-message session whose 145,367-token prompt SIZE-FAILed
against the resident 137,887 prefix (spill → `[safety-spill] SKIP: no endpoint state image` →
materialization fatal), not a single 1.45M-token request.

## T33 Wave Plan (2026-09-06) — gpillon fork adoption: Wave A agentic cluster, Wave B DFlash2 graft

Re-scoped per operator directive (09-06): T33 = full adoption of the gpillon/ninfer agentic
coding-traffic stack (the fork's driving problem) plus the DFlash2 drafter graft, executed as
two sequential waves on one branch, **off the T18 tip (`49400365`)** — NOT off the T31 line
(the base-selection error that killed the T36 wave: its base carried unshipped T31 host-KV
code). Cherry-pick only (diverged base; no merge). QUASAR artifact byte-identical in both
waves (Wave A = runtime only; Wave B grafts 66 new objects onto the 1,259-object base, base
tensors unchanged).

### Wave A — agentic serving cluster (no artifact change)

Pick set, chronological order in `gpillon/gpillon/coding`:

| # | Pick | What |
|---|---|---|
| 1 | `de386ad6` | system RAM KV cache for finished chats (tier base; cherry-pick of dylan 14329810) |
| 2 | `f144f052` | KV RAM used size + copy times in serve logs |
| 3 | `27665883` | two-tier probation/protected eviction by content lineage |
| 4 | `7bdee888` | active-lane prefix sharing for identical concurrent requests |
| 5 | `96371a3d` | one shared sibling snapshot instead of per-sibling capture |
| 6 | `f4b128c6` | **T34 guard**: stop offering rewrite-checkpoint restores from host RAM (append-at-frontier only) |
| 7 | `68b12497` | disable the RAM tier while an exact-key side store is in use |
| 8 | `eaf2037b` | record format v2 (hyperquant side-store carry; off-lane route, format preserved) |
| 9 | `07aeac2d` | preserve coding-agent prefix state (system/tools shared boundary) |
| 10 | `2065ed38` | capture dynamic shared-prefix boundaries |
| 11 | `2728ace4` | exact-size RAM captures + PreserveExisting admission |
| 12 | `093c1fdd` | sibling-prefix overlap telemetry (no behavior change) |
| 13 | `7a4634b5` | tagged request lanes @main/@agents/@classifier (@main-owned lanes evicted last) |
| 14 | `5f014910` | tool-call XML leak fixes (2 bugs incl. stream-teardown crash) |
| 15 | `6a1b62c5` | decouple warmup from client-facing request deadline |
| 16 | `27417ca2` | warmup fail-fast + auto kv-capacity bounds |
| 17 | `adf494c2` | block host sync — fixes 100% CPU decode (cudaDeviceSchedule spin) |
| 18 | `c2708ec8` | adaptive MTP verification widths |
| 19 | `9d86436c` | price adaptive widths by context depth |
| 20 | `9bef0f73` | calibrate round-cost model from measured round duration |

Conflict policy: our commits touch the same areas (`src/ops/softmax_attention.*` 6,
`src/ops/linear/nvfp4/*` 10, T23/T36 TMA work) — adapt gpillon's hunks to our signatures on
conflict; no reformatting; no new flags beyond theirs (RAM tier inert at `--host-kv-mib 0`,
tagged lanes schedule untagged traffic as today, adaptive MTP opt-in).

Wave A gates: host build PASS in the buildstage container (same path as the t31rev/t36 waves)
→ free-GPU ctest within T18 baseline (+ new kv-ram-cache unit tests) → quiet-window battery
16/16 → cache-hit-rate A/B (sibling sharing changes restore behavior) → greedy parity →
supervised ship via ninfer-ship.sh (mandatory quiet window, non-lane shipwatch). INT8 KV
@225,280, k=3, embedded template @xhigh unchanged.

### Wave B — DFlash2 drafter graft (only after Wave A is shipped and stable)

Engine picks (audited order, round 6): `a5064a9d → b5f6a15a → b4087269 → 34a33720 →
9990b6b2 → 92cdac97 → 6d962dba → 1953f2f4 → a14f0033`; keep `kNativeContext = 262144`; skip
`2dc34a79` (drafter SWA KV through the RAM tier — decide once Wave A's tier is live); add
serve telemetry `5ebbb1ab`/`fff3d6eb` if acceptance profiling needs it.
Quasar side: port the converter/graft (`graft_dflash2_module.py` equivalent) — graft the 66
objects (34 NVFP4 + 32 BF16 norms) from the `z-lab/Qwen3.8-27B-DFlash2` bf16 checkpoint onto
the QUASAR artifact; verify the 1,259 base objects byte-identical + 1,325 total; quasar
identity override in `validate_source_artifact`.
Wave B gates: acceptance rate beats the MTP3 incumbent (~1.95/round @ k=3) at **1.5K/8K/32K**
with no context collapse (fusion counter-evidence: 24.7% → 12.7% by 12K — the
context-collapse check is mandatory), greedy/verification parity, battery 16/16, INT8 KV,
QUASAR base byte-identical. `--spec dflash2` is the kill switch (module not materialized
without the flag).

### Decision rule
Wave A: adopt only if the quiet-window battery is green and no TTFT/decode/cache-hit
regression. Wave B: adopt only if acceptance beats MTP3 at all three contexts. Otherwise
keep the lane as-is and record the results here.
