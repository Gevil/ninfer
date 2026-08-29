# NInfer upstream-master merge plan — 2026-08-29

**Scope:** plan only (no merges executed). Resolves the 2026-08-26 "MERGES PARKED" decision in `ADOPTION.md` once the user approves execution.
**Repo:** `~/Work/Personal/ninfer` (remotes: `upstream` = Neroued/ninfer, `gevil` = GitHub fork, `dylan`/`eason`/`md`/`cometkim` = fork remotes).
**Forks audited:** all remotes fetched 2026-08-29; extra forks probed via scratch refs `refs/audit/*`.

---

## 1. Current state (facts)

| Item | Value |
|---|---|
| Upstream master | `feaf4dd0` (08-20) → `ce09aee5` (08-29): **89 commits, 421 files, +72 169 / −20 955 lines** |
| `gevil/qwen3.8-nvfp4full` | `1b9aef3b` (= tier6 mainline tree) — **89 behind, 151 local commits** |
| `gevil/master` | `37bc977f` (quasar-nvfp4 merged) — **89 behind, 169 local commits** |
| `gevil/tier4..tier7`, `tier3(-waveb)` | all 89 behind (stale; do not merge from these) |
| Lane working copy `~/containers/ninfer-nvfp4` | `fork/nvfp4full-merged` @ `0e044c01` — 89 behind, 52 local |
| Live quadlet `ninfer-nvfp4` | image `ninfer-nvfp4:quasar` (built 08-26 from `master`/`quasar-nvfp4` @ `0f61e1db`, battery 9/8 PASS, decode within −5%); quadlet `Image=ninfer-nvfp4:quasar` |
| Groupwise quadlet `ninfer` | `ninfer:latest` — currently stopped (port 8002 shared) |
| Lane config | Qwen3.8-27B, MTP3 + vision, 225 280-slot INT8 KV @ C=2, Sharp v22.3.2 template (bind-mounted, no rebuild on swap) |

**merge-tree (naive, full merge `upstream/master` into each mainline): 70 conflicts each.** 11 are modify/delete: upstream deleted files we patched — `src/serve/{openai_schema.{cpp,h}, anthropic_schema.cpp, tool_call_parser.{cpp,h}}`, `src/ops/launcher/gqa_attention{.h,_prefill.cu}`, `include/ninfer/ops/gqa_attention.h`, `src/runtime/engine/concurrent_executor.h`, `tests/test_responses_schema.cpp`.

## 2. What the 89 upstream commits contain

1. **Serve protocol-adapter re-architecture** (the dominant conflict source). `openai_schema.*`/`anthropic_schema.cpp`/`tool_call_parser.*` deleted; replaced by `openai_chat_*` / `openai_responses_*` (new full Responses API) / `anthropic_messages_*` / `http_transport` / `console_log` split. Also: cancel abandoned streaming (`ce09aee5`), thinking-signature validation + request ids (`6d0a6f1b`), neutral tool message names (`f8fc0f50`), text-part tool results (`79c292bc`), readiness gated on warmup (`ab82f886`), warmup kept out of context cache (`123bf1a1`).
2. **Runtime re-architecture**: program = resource authority (`020ca885`), paged-KV + state-image physical containers (`9d4cc6f9`, `c648a132`), **prefix-caching resource scheduling + default host context cache** (`f7bcd2ba`, `3a011c70`), measured context-cost scheduling (`dda31c75`), MTP prefill token kept on device (`02bd904e`), host-work timing (`94d0ef49`), qwen thinking-budget control (`a87d1fc6`).
   → **PR #73 (dylan, content-addressed host KV) was CLOSED UNMERGED 08-27, superseded by upstream's own host-context-cache work.** Our 4 port commits (`7fa78bec`, `cd8b4be4`, `b5ecbd1b`, `0d45cfce`) collide with this.
3. **Attention rework**: FP8 KV cache (`6183c9be`), **INT8 KV Hadamard rotation** (`17a7275f` — touches our INT8-KV path), softmax-attention ownership consolidation (`a58a946c`), GDN recurrent unification (`e9c7046f`), GDN prefill conv written straight into Q/K/V (`92bb06eb`).
4. **Ops perf** (md's wave partially landed): MoE prefill weight staging ×8 codes (`8d343527`), FP8 W8A16 vocabulary GEMM (`1fc1cb76`), sparse-MoE gather-lifetime fix (`3a61ef3f`).
5. **Frontend perf/correctness**: one-pass media placeholder expansion (`8ba4bfd0`), one-encode prompt boundaries (`189a0a12`), BPE merge linearization (`92b1a721`), in-place media resize (`e196f09f`), malformed UTF-8 repair (`4cece118`), oversized-prompt rejection at tokenization (`2fb1a504`).
6. **Model-target changes**: `src/targets/qwen3_6_27b` package.h +54 lines (new profile surface) and `package.cpp` — **collides with our quasar/nvfp4full profile registrations**.
7. **Dev branch**: 2 commits ahead of master (`0b5b7c9a` logprob reduction, `11e76d8d` perplexity corpus) — eval tooling, not needed for the lane.

## 3. Merge plan (phases)

### Phase 0 — preparation (½ day)
- Confirm no active session/orchestration on the lane (PARKED gate, per skill). `git status` clean (commit or stash the dirty `ADOPTION.md` first — it holds the parked-audit record).
- Baseline: record current image digests (`:quasar`, `:tier6-*`), `journalctl` KV ledger line, decode probe (fresh-context + 98k) for regression comparison.
- Scratch clone strategy: do the merge in a clone of `~/Work/Personal/ninfer` (or use it directly on a new branch — the working copy is clean apart from ADOPTION.md).

### Phase 1 — merge upstream/master into `qwen3.8-nvfp4full` (2–4 days)
Branch `upstream-2026-08-29` from `1b9aef3b`; `git merge --no-ff upstream/master`.
Per-area conflict policy:

| Area | Policy |
|---|---|
| `src/serve/*` (11 files + deleted-file rework) | **Adopt upstream adapter architecture wholesale.** Re-land our local serve patches as new commits on top (see keep/drop table §4). This is the biggest rework block. |
| `src/runtime/*`, `src/core/paged_kv_cache*` | **Adopt upstream.** Drop our PR #73 port (4 commits, superseded); keep PR #64 host-KV park only if upstream's host-context-cache does not cover it (check `2190c4a1` bidirectional host-KV-swap bench first — upstream appears to have first-class host KV now; likely drop #64 too and verify `--host-kv-cache-mib` UX parity). |
| `src/ops/*` (gqa_attention deleted, small_t, sparse_moe_prefill, q5/q6 rowsplit, GDN recurrent) | Adopt upstream; verify our tier3/4/5 cherry-picks against the new tree (SwiGLU W4A4 `55036778`, staged split-KV reduce `0d5efa28`, GEMV cp.async `e481ccd8`, decode-suite) — re-apply only the ones not superseded by md's landed wave. |
| `src/targets/qwen3_6_27b/*`, `qwen3_6_35b_a3b/*`, `qwen3_6/impl/*` (29+2 files) | Resolve keeping our **nvfp4full + quasar profile registrations** while taking upstream's new package.h surface. This is where the quasar converter/profile work (`303dbcaa` et al.) lands. |
| `src/frontend` (chat_template.cpp, processor.*) | Adopt upstream frontend perf; re-apply our **vision-pad stray-marker strip** (tier1/2 `caec0dee`) and **starts_in_reasoning fix** (`f68b1106`) on top. |
| `tests/*` | Adopt upstream's rewritten suites (`test_openai_schema` +1 210, new `test_openai_responses` +976, `test_resource_manager` +1 945); adapt our local test changes (vision-pad, tool-params) to the new APIs. |
| `docs/`, `tools/` | Take upstream (they deleted `concurrent-inference-architecture.md`, added TTFT campaign + `tools/ninfer_serve` client libs). |

**Verification gates (per commit batch, not just at the end):**
1. Container build (buildstage pattern, CUDA 13.1.2, `-DCMAKE_CUDA_ARCHITECTURES=120a`) — must be clean.
2. Host ctest suite (5 serve suites + frontend; Qwen tokenizer bind-mount per skill).
3. Free-GPU host-KV ctest (5/5 pattern).
4. Tag the resolved merge: `merge upstream/master (ce09aee5) (local)`.

### Phase 2 — battery + image ship (1 day)
- Build `ninfer-nvfp4:upstream-ce09aee5` from the merged tree; run the standard battery (225 280 ctx, decode @ fresh + 98k, thinking-mode request with `max_tokens ≥ 256`).
- **Gate: decode ≥ quasar baseline −5%** (tier7 precedent: a big merge halved decode and was rolled back the same day).
- Check the boot ledger line: upstream's default host-context-cache provision changes VRAM accounting — free-after-startup must stay ≥ ~1.0 GiB at 220K/int8/C=2 (was 1.40 GiB).
- On PASS: push `gevil` branch, open/update the GitHub PR (current Gevil PRs #1–#8 are all merged tier-PRs; this is a new "upstream convergence" PR). On FAIL: park, keep `:quasar` live, file the regression per battery log.

### Phase 3 — quasar line (1 day)
- `master` (quasar) = `qwen3.8-nvfp4full` merge + re-apply/verify the 4 quasar commits (`c4761d16`, `8d88bf26`, `303dbcaa`, `0f61e1db`). If the quasar profile registration merged cleanly in Phase 1, this is mostly re-verification: rebuild `:quasar` from the new master, battery again (gate: −5%), retag + restart, ledger check.

### Phase 4 — candidate cherry-pick wave (post-merge; each = own tagged commit, per skill)
See §5/§6 for the ranked list. After the master merge, "89 behind" forks become cherry-pick-cheap (no rebase onto an old base needed).

## 4. Local serve-patch keep/drop table (re-land audit during Phase 1)

| Local patch (commit) | Upstream status | Decision |
|---|---|---|
| `0e498b5b` reasoning_effort kwarg (Sharp steering) | No upstream equivalent; upstream added thinking-budget control (`a87d1fc6`) + signature validation | **KEEP** — re-land in `openai_chat_request.cpp`/`openai_common.cpp`; may compose with the new thinking-budget knob (check overlap) |
| `102ab113` tool-arg schema typing (PR #65 port) | **Superseded**: upstream `0e4cdf84` "type tool arguments by schema" | **DROP** (verify upstream behavior covers our test cases) |
| `f91b7c85`/`51ea7d86` warmup decoupling (PR #79 port) + `5c9e9dbc` fail-fast | Upstream `ab82f886` readiness-on-warmup + `123bf1a1` warmout-of-cache | **DROP**, verify equivalence in battery |
| `677b9e8c` cached-tokens usage (PR #55) | PR #55 still OPEN upstream | **KEEP** (re-land; drop once upstream merges #55) |
| `325986ea` image token budget (PR #61) | PR #61 still OPEN upstream | **KEEP** (re-land in the new request path) |
| `0e659f73`/`a8cdc1a6` tool-param typing base picks | Superseded by `0e4cdf84` | **DROP** |
| `a568115b` json_object/json_schema response_format | Unknown — check `412b11f3` "close OpenAI compatibility gaps" | **VERIFY**, keep if not covered |
| `d19efff1` terse kwarg (Sharp v22.3.2) | None upstream | **KEEP** (our template contract) |
| `b189de0a`/`41ef241f` `--chat-template-file` override | Unknown — grep upstream for chat-template-file | **VERIFY**, keep if not covered |
| `e7430984`/`3ab8e8e8` max_model_len / n_ctx on /v1/models | Unknown — check `412b11f3` | **VERIFY** |
| `b2aec4d2` et al. `--webui` in-process WebUI | None upstream (Windows-port PRs touched webui updater) | **KEEP** (re-land; Windows guard from `7f90fc12` included) |
| `cd8b4be4`/`0d45cfce`/`b5ecbd1b`/`7fa78bec` PR #73 host-cache ports | #73 closed unmerged, superseded by `f7bcd2ba`/`3a011c70` | **DROP** — adopt upstream host context cache; keep only if a UX gap remains (`--kv-host-cache-mib` opt-in vs upstream default-on) |
| `0bb10f9a`/`967c2ba5` PR #64 host-KV park | #64 closed; upstream host KV appears first-class now (`2190c4a1` bench) | **VERIFY** then likely DROP |

Agent-critical regressions to cover in the battery (OWUI/aistock/opentrade depend on these): `chat_template_kwargs.reasoning_effort`, `preserve_thinking`, tool-call parameter round-trip (string-typed params), image content parts with `--preserve-thinking` (the 08-23 vision-pad 400 storm), `/v1/models` contract fields.

## 5. Open-PR audit (Neroued/ninfer, 14 open)

| PR | Title | Relevance | Verdict |
|---|---|---|---|
| **#107** (koloved) | qwen3.8 wire-format detect NVFP4 artifact profile (+24/−5, 5 files) | **HIGH** — routes Ostfralla W8+NVFP4 artifact (18.3 GB, faster on 5090, 262144 ctx) to the Qwen36Nvfp4 profile; same fix as gzenz `69ca6788`. Touches `qwen3_6_27b/package.cpp` — resolves in Phase 1 conflict anyway | Cherry-pick **after** master merge (conflict file already being resolved); optional if we don't want the Ostfralla artifact |
| **#72** (iamwavecut) | On-demand vision residency (`--vision-residency overlay`) (+1938/−81, 51 files) | **HIGH for this lane** — tested on sm_120 5090 32GB qwen3.8-27b: int8 → +2.35 GiB free-after-startup (our 1.40 GiB slack → room for more KV), bf16 → +36 288 tokens. Opt-in, defaults unchanged | Tier item **after** master merge: dedicated branch, own battery at our 220K/int8 config. Biggest single-VRAM win available |
| **#113** (md) | NVFP4 fused SwiGLU at every 256-token block (1 file, 1 commit on master) | **MED-HIGH** — 1.06–1.37× where newly applicable; our model is NVFP4 27B | Cherry-pick post-merge |
| **#112** (md) | W8 row-split decode 8 codes/thread (1 file, 1 commit on master) | **MED** — 1.09× GEMM shapes; relevant to W8 profile paths (quasar/groupwise) | Cherry-pick post-merge |
| **#89** (igorls) | Size persistent grids from active device SM count | **ALREADY PORTED** — Tier 6 (`gevil` PR #7) | Verify our port matches upstream after merge; no action |
| **#37** (kimsey0) | Ollama-compatible chat endpoints (+1871, 18 files) | LOW-MED — useful if aistock/opentrade ever need Ollama endpoints; OWUI already speaks OpenAI | Defer |
| **#35** (danielfparkernz) | Compressed-KV E8 lattice for Blackwell (+2717, 26 files) | LOW now — we have local port branch `pr-35`; INT8 KV already covers 220K; E8 = future longer-context | Park; revisit when context > 225K is needed (MirkoCovizzi's dynmtp branch extends it, §6) |
| **#55**, **#54**, **#61** | cached-tokens, terminate-exception, image budget | Already in our tree (Tier 1–2 ports) | No action; they remain open upstream — once upstream merges them, our re-landed copies should converge |
| **#97** | Container build C++/CUDA cache | Tooling only | Skip |
| **#84**, **#59** | Windows ports (2 of them) | We're Linux | Skip |
| **#48** | bench request-log schema v10 | Small bench fix | Adopt with the master merge if bench tooling matters |

## 6. Fork audit (recently pushed, 2026-08-22 → 08-29)

| Fork (pushed) | Findings | Verdict |
|---|---|---|
| **MirkoCovizzi/ninfer-rtx5090-mobile** (08-29) | Same GPU as us. `feat/dynamic-mtp` (16 ahead): **adaptive MTP verification widths**, "eliminate wide-MTP decode cliff" (`08d0d444`), MTP pricing by context depth, laptop-5090 MTP tuning + the E8 compressed-KV port (same lineage as PR #35) + vectorized E8 decode. `fix/mtp-greedy-parity` (3 ahead): **"make greedy verification width-invariant"** (`56dfda80`/`7d566547` — note: `7d566547` is exactly what our reverted Tier 7 PR #8 merged; the fix landed and was reverted for a 50% decode regression — any re-adoption must be gated on the same battery). `feat/quasar-nvfp4-converter` — he's independently building quasar-NVFP4 converter work (cross-check source for our quasar profile). `perf/nvfp4-swiglu-m16n256` (11 ahead) overlaps PR #113 | **Most interesting fork**. Cherry-pick candidates post-merge: MTP greedy parity fix (battery-gated), adaptive-MTP widths (Tier item), swiglu (compare vs #113). Not a merge base (89 behind) |
| **dylanbrodiefafard/experimental** (08-27, moved to `13a2e5d0`) | 9ec0d8c4 **quantize MTP layer weights to NVFP4 from BF16** (direct VRAM win for our MTP3 lane); e3f40a24/20f622f6 nvfp4 prefill/decode-band tuning; 48d18570 **don't re-tokenize whole conversation per turn** (TTFT for long agent chats); d98f3fdb checkpoint pinning (issue-#62 agent line); 2c7785d9 packed speculative verify; 583d8e10 "block host sync to fix 100% CPU during decode" | Cherry-pick wave candidates post-merge (all on an old base): MTP-NVFP4 quant, no-re-tokenize, packed-verify, CPU-spike fix |
| **MichaelDementii** (new branches 08-28/29) | PR branches = #112/#113 (see §5); `perf/gdn-conv-direct-qkv` already landed in master (`92bb06eb`); `perf/consolidated-kernel-stack` (38 commits) = research superset of his wave (TMA-L2 walk, 15-token draft window, fp16 PV tiles) — mostly superseded now | Skip the stack; adopt #112/#113 |
| **gzenz/ninfer** (08-28) | `windowed-mtp-experiment` (25 ahead): **windowed MTP draft attention** (`c3831d22`), host-KV evicting-restore (`15ddfe5a`), admission-retry on KV exhaustion (`35c3162d`…), `/stats` endpoint + request-log rotation (`f640f7b8`) | Cherry-pick candidates post-merge (89 behind): windowed MTP (Tier item), stats endpoint (small, useful for the dashboard) |
| **scchow/ninfer** (08-28) | `upgrade/sharp-on-upstream` (15 ahead / 18 behind): **salvage tool calls emitted in the reasoning channel** (`6edae87f`, `a8809317`), TTFT/ITL metrics in OpenAI responses (`3f6edb0e`), stray think-close strip | Salvage fix = candidate (overlaps our local pr-10 `7aa92003` and gzenz `770184c7` — pick the best of the three during the merge); TTFT/ITL metrics useful for battery tooling |
| **potatohog/ninfer** (08-25) | `rotation` branch: **5 unmerged INT8-Hadamard fixes** (`f8e7e200`, `6b97640a`, `608a4dc7`, `e098b185` + test) — not in upstream (only the base feature `17a7275f` is). Our lane runs INT8 KV | **Review against upstream's hadamard feature during the merge**; if the fixes address real defects in `17a7275f`, cherry-pick (correctness on our KV path) |
| **mr-september/ninfer-sharp** (08-25) | `fix/stray-think-close` (4 commits) — same stray-think-close problem as scchow's, older base | Superseded by scchow; skip |
| **Don-Chad/ninfer-3090** (08-29) | sm_86 RK8/V4 paged quant for RTX 3090 (38 commits, 126 behind) | Not our GPU (sm_120) — skip (reference only) |
| natpate, SV8ARJ, AmerM137 | Windows ports | Skip |
| geoffwatts (v100), bratnieks (AMD), JCraigWasTaken (gfx906) | Other architectures | Skip |
| knoopx, SolettaSolaris, potatohog-master, eason, md-master | Mirrors / fully upstreamed / dylan-line mirrors (`33627e8` template-reasoning-effort = known overlap, already decided) | Skip (re-audit of eason confirms: master still `feaf4dd0`, develop unchanged `9168239b`) |

## 7. Execution order (when approved)

1. **Phase 0** prep + baseline capture.
2. **Phase 1** upstream merge into `qwen3.8-nvfp4full` with the §4 keep/drop policy → clean build + ctest.
3. **Phase 2** battery gate (−5% decode) → ship merged image (parked `:upstream-ce09aee5` until PASS).
4. **Phase 3** quasar re-verification → retag `:quasar` from merged master → restart + ledger.
5. **Phase 4** cherry-pick wave, ranked by value/risk:
   a. potatohog INT8-Hadamard fixes (correctness, our KV path)
   b. #107 profile detection (small, unlocks Ostfralla artifact)
   c. #113 NVFP4 SwiGLU + #112 W8 rowsplit (decode perf, single commits on master)
   d. dylan: MTP-NVFP4 quant (`9ec0d8c4`) + no-re-tokenize (`48d18570`) + packed-verify (`2c7785d9`)
   e. scchow reasoning-channel salvage (agent robustness)
   f. Tier items: #72 vision overlay (+2.35 GiB headroom), MirkoCovizzi adaptive-MTP, gzenz windowed-MTP — each own branch + full battery (tier7 precedent: decode-gate before acceptance)
6. GitHub: push merged `qwen3.8-nvfp4full` + `master` to `gevil`, open the convergence PR; update `ADOPTION.md` (unpark record) and this file.

## 8. Risks & gates

- **Serve rework is the biggest unknown** (protocol adapters deleted/rewritten; our 8 local patches must re-land). Mitigation: re-land one patch per commit, battery per batch; agent-critical contract list in §4 is the regression net.
- **Decode regression risk** (tier7: −50% on a big merge, rolled back same day). Gate: battery decode ≥ baseline −5% before any retag.
- **VRAM re-accounting**: upstream default host-context-cache + our 220K/int8 config → verify ledger free-after-startup ≥ 1.0 GiB (was 1.40 GiB).
- **INT8 Hadamard correctness** (`17a7275f` + potatohog's unmerged fixes) — our KV path; ctest + a long-context probe required.
- **Quasar profile registration** lives in the most-changed target files (`qwen3_6_27b/package.*`) — resolve carefully; `verify_structure` quasar fix (`8d88bf26`) must survive.
- Rollback path at every step: current images (`:quasar`, `:tier6-*`) parked; quadlet retag + restart is the only live change; no destructive ops on `gevil` branches.

## 9. Tiered execution table (aligned with ADOPTION.md Tier 8/9/10)

Renumbering: ADOPTION.md's parked 08-26 proposal defined T8 = DFlash2 probe, T9 = dylan XAttention (deferred), T10 = upstream dev→master convergence (WATCH). T10's trigger is now met — dev's work landed in master as part of the 89-commit wave. New tiers continue from T11.

| Tier | What | How (method) | Target | Gate | Est. |
|---|---|---|---|---|---|
| **T8** (defined 08-26) | cometkim DFlash2 probe — 5 cherry-picks + nvfp4full-v2 artifact, free-GPU A/B vs 141.7 tok/s @98k baseline | Probe branch on the **current** tree (before T10 — the decision needs only the old base); 90-file DFlash22DraftModel line | scratch `tier8-dflash2` | buildstage free-GPU A/B; net decode must beat baseline | hours |
| **T10** (trigger met) | Upstream master sync: 89 commits `feaf4dd0`→`ce09aee5` (421 files) | One `git merge --no-ff upstream/master` from `qwen3.8-nvfp4full` (`1b9aef3b`) → branch `upstream-2026-08-29`; per-area conflict policy (§3/§4): adopt serve re-architecture + re-land local patches commit-by-commit, drop #73 port, adopt new runtime, keep nvfp4full+quasar profiles | `qwen3.8-nvfp4full` | clean container build (120a) + host ctest + free-GPU host-KV ctest | 2–4 d |
| **T11** | Quasar re-verification on merged tree | Merge T10 result into `master`; rebuild `ninfer-nvfp4:quasar`; retag + restart | `master` | battery −5% decode gate + ledger free-after-startup ≥ 1.0 GiB (host-context-cache re-accounting) | 1 d |
| **T12-A** | Quick wins + correctness | Individual cherry-picks, each its own tagged merge commit: potatohog INT8-Hadamard fixes (5, our KV path), PR #107 profile detection, PR #113 NVFP4-SwiGLU, PR #112 W8 rowsplit | `qwen3.8-nvfp4full` | build + ctest + decode spot-check | 1 d |
| **T12-B** | Agent-workload wave | Cherry-picks: dylan `9ec0d8c4` MTP→NVFP4 quant (VRAM), `48d18570` no-re-tokenize (TTFT), `2c7785d9` packed verify, `583d8e10` CPU-spike fix; scchow reasoning-channel salvage | `qwen3.8-nvfp4full` | build + agent contract battery (§4 list) | 1–2 d |
| **T12-C** | Tier items (each own branch + full battery, tier7 precedent) | #72 vision overlay (+2.35 GiB headroom); MirkoCovizzi adaptive-MTP widths; gzenz windowed-MTP + `/stats`; dylan checkpoint pinning | dedicated branches → `qwen3.8-nvfp4full` | full battery, −5% gate, ledger | 2+ d each |
| **T9** (deferred 08-26) | dylan XAttention prefill (13 ahead / 35 behind) | Revisit only after T8 decision + prefill bottleneck evidence; if T10 merge lands first, re-resolve on new tree | — | gated on T8 | — |
| **Hold** | PR #35 E8-KV, #37 Ollama adapter, Don-Chad 3090 (sm_86), cometkim 768k presets (lane is 225k on 32 GB), tier7 fast-path fix (fallback if T8 probe fails) | — | — | — | — |

Ordering rationale: T8 first (hours, decides the spec-decode direction the 08-26 parking was waiting for, and gates T9) → T10 (unblocks everything: every fork/PR candidate is "89 behind", so post-merge cherry-picks are conflict-cheap) → T11 (keep the live lane green) → T12-A→B→C (cheap wins first, battery-gated Tier items last).
## 10. T10 execution record (2026-08-29)

Branch `upstream-2026-08-29` (from `qwen3.8-nvfp4full` @ 1b9aef3b):

- **7e771891** `Merge upstream/master (ce09aee5)` — the 89-commit wave (421 files, +72k/−21k).
  Policy per §3/§4: adopt the serve protocol-adapter re-architecture (`openai_chat_*`/
  `openai_responses_*`/`anthropic_messages_*` replace `openai_schema`/`anthropic_schema`/
  `tool_call_parser`), the new runtime (resource authority + prefix-cache scheduling +
  default host-context cache), ops consolidation. Keep: nvfp4full + quasar profiles,
  all tier ops perf picks (files upstream did not touch stayed ours), ADOPTION docs,
  `tools/convert/qwen3_8_27b` nvfp4full converter. Drop: our #73 host-KV port tests
  (upstream deleted the mechanism they superseded).
- **Re-lands (in order):**
  | commit | what |
  |---|---|
  | 839ea7d1 | Jinja chat template override `--chat-template-file` (quadlet dependency) |
  | 81e32cc9 | honor reasoning effort in custom Jinja templates |
  | dbc6e2f2 | Sharp template kwargs (`reasoning_effort`/`terse`) + structured output (`response_format`) |
  | 57e355fb | `--webui`/`--webui-dir` in-process stock WebUI serving |
  | 333b339e | `starts_in_reasoning` derived from rendered prompt (fc4f6cae) |
  | d39d050b | strip stray vision-pad markers (caec0dee, extended: source-coord strip shifts every boundary/span field) |
  | f7b7e96b | tool-call checkpointing, BPE-stable placement (281cf9ec = dylan 673d695c) |
  | 159cf61d | drop stray think-close markers in tool-capable content (e4beff22 = PR #86 port) |
  | d0d8cfc5 | literal vision tokens kept out of media binding (6da2efef, adapted to RenderBuilder; fragment literal-spans shifted per inserted break byte) |
  | 1f739774 | skip empty think wrappers on past assistant turns (16f405d4) |
  | 75244b0a | terminate handler names the escaping exception (PR #54) |
  | d03b0a72 | drop duplicate declarations left by re-lands (`chat_template_path`, `supports_terse_`) |
- **Verified superseded (no re-land):** PR #55 cached_tokens (upstream native:
  `openai_chat_response` reports `prompt_tokens_details.cached_tokens` from
  `prefix_cache_hit_tokens`); PR #79 warmup decoupling (upstream warmup uses
  `DeadlinePolicy::UnboundedStartup`); PR #61 image token budget (replaced by upstream
  media admission: `--media-cache-mib`/`--media-live-mib`); e7430984/3ab8e8e8
  max_model_len/n_ctx (upstream exposes `max_model_len` natively in /v1/models).
- **Deferred:** ca0763ba WebUI dialect (built-in WebUI off by default; OWUI is the lane UI;
  re-land only if the in-process WebUI is ever enabled on the lane); PR #65 string-typed
  tool params (upstream now requires string `arguments`; verify OWUI tool-call round-trip
  in the T11 battery — if it 400s, re-land 102ab113 against `openai_chat_request`).
- **T11 quadlet delta (verified against merged `serve_options`):** only
  `--kv-host-cache-mib 32768` → `--host-kv-mib 32768`. Every other quadlet flag
  (`--spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking --max-context
  225280 --kv-capacity 225280 --kv-dtype int8 --max-concurrency 4 --pending-timeout-ms
  900000 --default-max-tokens --chat-template-file`) is present in the merged tree.
- **Gate:** container build + serve-layer ctest running
  (`~/.local/share/ninfer/logs/t10-buildstage-*.log`, `t10-serve-ctest-*.log`).
