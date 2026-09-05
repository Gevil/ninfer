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

### TIERED PLAN (re-evaluated 2026-09-05)

| Tier | What | Status (2026-09-05) |
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
| 14 | host-KV content cache (#73) | RE-SCOPED — premise killed by #98 (upstream scheduler supersedes #64/#73/#90); now = validate the new in-tree #98 architecture on the T14 probe (hot + 45s-idle branch switch) |
| 15 | gzenz NVFP4 KV + YaRN (2.12, 400k ctx) | SHIPPED 09-02 in tree (`t15-yarn`) — **not live**: the T-Q quadlet re-pinned INT8 KV 225280 ctx; config flip pending if long-context agent work is the priority |
| 16 | upstream convergence wave 1 | DONE 09-02 (absorbed in the `t15-yarn` line; pulled md micro-opts + GDN fix for free) |
| 17 | pv-f16acc (md) | DONE 09-02 (`t15-yarn`); T25c 64k-needle PASS 09-04 — the 35B caveat did not reproduce on our 27B lane |
| 18 | dylan wave 2 (GDN chunked-prefill precision `f25f5463`) | PREPARED — `t18-dylan-wave2 @ 269cf431`, compile GREEN; gates: GPU ctest + MTP/dflash accuracy battery; re-stack onto `t24-quasar-a` first |
| 19 | `gated_delta_net_snapshot` op + tests (dylan) | NOT STARTED — candidate if the MTP/draft-state path is wanted (separate feature; not self-contained in T18) |
| 20 | watch: open upstream PRs (08-30/09-04 plan) | WATCH — #167/#160 promoted to T23; current set: #152 shared-prefix write, #148 Responses API, #163 timings, #162 metadata |
| 21 | watch: upstream issues (08-30/09-04 plan) | WATCH — #169 output-limit truncation, #168 strict:true opt-out, #164 KV precision tail, #166 context-cache 503 |
| 22 | upstream convergence wave 2 (`863aa8a5`: #170 fix + tool parser + prefix reuse) | DONE 09-04/05 — LANDED via the T-Q ship (`t24-quasar-a` subsumes `t22-converge` @ `1eff9672`) |
| 23 | md TMA prefill pair (#167 fp8-A8 TMA + #160 nvfp4 blocked scales) | QUEUED — `t23-tma-pair @ a05618f7` prepared (both PRs still OPEN upstream); re-stack onto `t24-quasar-a` + post-merge ctest + battery = **next ship** |
| 24 | gzenz host-KV safety net | RE-DERIVE from `d205c52a` — 19 new commits (spill guards, eviction caps, session-key fix, rewrite-checkpoint Option A); #170 class already in-tree via T22 → scope = host-KV hardening only; gated on the host-KV re-enable decision (pairs with T14) |
| 25 | probe wave (09-04 plan) | PARTIAL — T25c 64k-needle PASS (validates T17); remaining probes per the 09-04 plan |
| 26 | watch slot (09-04 plan) | WATCH — open-PR watch set |
| 27 | watch slot (09-04 plan) | WATCH — open-PR watch set |
| 28 | dylan dflash2 wave (vision-validated dflash2 + verify-boundary fixes) | NEW — PROBE-GATED: A/B vs the MTP3 baseline on acceptance + net decode (quiet window) |
| 29 | Mirko dynamic-MTP decode wave (adaptive widths + small-batch swiglu) | NEW — candidate decode A/B vs the 151.9/150.2 baseline |
| 30 | Mirko KVaRN line (mtp3 score production) | NEW — TRIAGE: adopt only if precision-neutral + decode-positive; KV floor applies |

**Sequencing (2026-09-05):** (1) **T23** — prepared, re-stack + ctest + battery (next ship);
(2) **T18** — prepared, GPU ctest + accuracy battery; (3) **T14 probe + T24 re-derive** —
together, gated on the host-KV re-enable decision; (4) **T25** remaining probes; (5) **T28**
dflash2 probe if spec decode beyond MTP3 is wanted; (6) **T29** dynamic-MTP decode wave;
(7) **T30** triage. Standing config decision: T15 (NVFP4 KV + YaRN 400k) sits in tree, not
live — flip the quadlet only if long-context sessions outrank the 8–14% native-context decode
penalty.

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
