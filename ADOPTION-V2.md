# ADOPTION-V2.md — adoption record re-baselined on `quasar-master`

**As of 2026-09-09.** Branch `v2/adoption` (off `quasar-master` @ `f7727926`), docs-only.
All SHAs, line numbers, ahead/behind counts, apply results, and PR/issue states in this file
were measured on 2026-09-09 after `git fetch --all --prune` (see Appendix A for the harnesses).

---

## 1. Purpose and supersession

The lane's code baseline changed on 2026-09-08: the ~275-commit fork lineage was replaced by
**pure upstream master + one commit** (`quasar-master` @ `f7727926`), and upstream now ships its
own dflash2 speculative-decode engine (which the lane now runs). That voids part of the old tier
ledger, makes the fork's `master` no longer describe the running code, and re-opens the feasibility
of the gpillon agentic-coding cluster that was previously parked as a "weeks-long hand-port".

This file is the **adoption authority** as of the `quasar-master` baseline. It:

- supersedes the operational tier ledger (blob `033194ae89d4bc89f2288cc64e228ce1ec0418d7`, 761
  lines, on `t31rev-quasar` / `t33-gpillon-quasar` / `t36-mdops2-quasar`, and committed on
  `t42wave-quasar` as `aec57b08b0121d3ce876e29d643772fc6695225c`);
- re-frames the round ledger (blob `6e53e2a6443063be7be3b7e54759c371d7277a4a`, 2541 lines, Rounds 1–22,
  tiers T1–T49) as **historical** and reconciles every tier against the new baseline (§5).

Neither source file is deleted, edited, or moved. They remain citable by `branch:path`.

## 2. Baseline (what the lane runs)

| | |
|---|---|
| Code baseline | `quasar-master` @ `f7727926` = upstream `b88c0f6f` + `f7727926` (`targets(qwen3_6_27b): support Qwen38Nvfp4* weights profiles for quasar artifacts`, 1 file: `src/targets/qwen3_6_27b/impl/load/bindings.cpp`, +103/−3: `WeightsProfile { Qwen38Nvfp4, Qwen38Nvfp4Full }` family switch + 3 MTP3 kv-workspace curve cases) |
| Upstream state | `b88c0f6f` = upstream master **and** dev tip (live `ls-remote` 2026-09-09); `quasar-master...upstream/master` = 1 ahead / 0 behind |
| Image | Baseline image `localhost/ninfer-nvfp4:qm-f7727926`. **As of 2026-09-09 (end of day), V2-T1 → T2 → T3 → T5 have shipped in sequence on top** (V2-T4 rejected) — the live image is the **V2-T5** candidate build (branch `v2/t5-decode` @ `324a8de3` + the 3 on-path decode picks, image `f8b76e5a4dc2`), retagged to `:quasar`/`:latest`; the earlier V2-T1 tag `:v2t1-ba21e676` is retained. quadlet `ninfer-nvfp4`, port 8002→8080, 2 Volume mounts |
| Artifact | `~/.local/share/ninfer/models/qwen3.8-27b-quasar-dflash2-master/qwen3_8_27b_quasar_dflash2_master.ninfer` — 1334 objects, 19.78 GiB (QUASAR base + 66 master-contract dflash2 objects), sha256 `da5efb3332e00ed5a9d719aa5cc09a4066fa03ab0d1706f6119f2fba8f2ba338` |
| Flags | `--spec dflash2 --draft-tokens 7 --kv-dtype nvfp4 --kv-capacity 225000 --max-context 225000 --host-kv-mib 16384 --max-concurrency 4 --vision --preserve-thinking --default-max-tokens 80000 --pending-timeout-ms 900000 --model-id qwen3.8-27b` — **the V2-T1 ship reset `--max-context`/`--kv-capacity` from the 09-08 256K bump (`262144`) to `225000`; re-bumping to `262144` is a pending follow-up** |
| Probes (2026-09-08) | 189.5 / 140.4 / 51.2 tps @ 1.5k/8k/32k prompt; 8.8 tps @ 120k; dflash2 acceptance 32.6 / 36.2 / 28.2 / 41.7 % |
| Rollback chain | `~/.local/share/ninfer/quadlet-backups/`: `qm256.bak-*` (dflash2 @ 225280/8192), `qmd2.bak-*` (MTP3 on pure master), `qm.bak-*` (T33 incumbent), `qg1.bak-*` — per-window epoch-suffixed names in the directory |
| OMP-mounted pin | `chat_template.jinja` sha256 `180e7015759b2b6b57574d6c2ca5c2d19eb2b05a4aaffa80866f71eb1a1fad1a` (byte-identical across every quadlet change) |

**Upstream dflash2 surface already in the baseline** (this is the engine the lane runs):
`src/ops/attn_input_proj/w8/w8_dflash2_attn_input.cu`, `src/ops/linear_swiglu/w8/w8_dflash2_linear_swiglu.cu`,
`src/ops/common/dflash_rope.cuh`, `src/targets/qwen3_6/impl/runtime/dflash_{context.h,context_impl.h,impl.h}`,
`tools/convert/qwen3_8_27b/dflash2_{inventory,recipe}.py`, `tests/targets/qwen3_6_27b/test_engine_dflash2_real.cpp`,
`tests/convert/qwen3_8_27b/test_dflash2_recipe.py`, `docs/maintainer/qwen3.8-27b-dflash2.md`,
`bench/targets/qwen3_6_35b_a3b/dflash_round_bench.cpp`.

## 3. Record provenance (where the prior records live)

Measured 2026-09-09: `git rev-parse <branch>:ADOPTION.md` for every local branch.

| Blob (full) | Lines | Branch(es) | Content |
|---|---|---|---|
| `6e53e2a6443063be7be3b7e54759c371d7277a4a` | 2541 | `master` (8a75dda3), `t41-upstream-agentic` (d7e44386) | Round ledger: Rounds 1–22, tiers T1–T49; T33 gpillon Wave Plan (line 1827); round-6 TIERED PLAN (line 1496); Round 22 full re-audit (line 2445) |
| `aec57b08b0121d3ce876e29d643772fc6695225c` | 1053 | `t42wave-quasar` (committed 2026-09-09 as 6e575be7) | Tier ledger + the 2026-09-08 evening record (puremaster2-df2 window, gpillon as-is experiment, dflash2 consolidated verdict + CORRECTION, quasar-master cutover, quasar-dflash2 graft, 256K bump, lane cleanup, fork push) |
| `033194ae89d4bc89f2288cc64e228ce1ec0418d7` | 761 | `t31rev-quasar`, `t33-gpillon-quasar`, `t36-mdops2-quasar` | Tier ledger (pre-09-08): T1–T30 TIERED PLAN (lines 739–781) |
| `2eca88b65ecc1d97b71526ef8814422515e74251` | 2185 | `t31-hostkv-quasar` | Adds Rounds 2–16 + t18gdn ship + t36mdops regression |
| `ed73df0818a7c98722616eb639e3ab6daf2bae58` | 830 | `t33-dflash2-quasar`; also the linked worktree `/tmp/t33-wave-b-wt` (same blob, no unique content) | T33 Wave B — QUASAR DFlash2 drafter record |
| `8fd773bc442a11f7672d727defcfe33a91a1478e` | 1527 | `t36-mdops-quasar` | T36 wave record |
| `7b2ad93fc35bbc2bd3136d6c6554ca61447c26ee` | 880 | `t18-gdn-quasar` | T18 GDN record |
| `703337bf70300819b0b65b325db7c38f3b63bdb1` | 753 | `t24-quasar-a` | T22/T24 record |
| `e2bca2fa21266d05ebd84181e791fa4c064c8335` | 429 | `t18-dylan-wave2`, `t23-tma-pair` | T18/T23 record |
| `19dec832a5f6698545fd7798bdb6960498e72a8a` | 519 | `quasar-nvfp4` | Early QUASAR record |
| `8e393259dd3405b95e917d56e7d9f77036c386a2` | 385 | `mtp-sampled-draft`, `qwen3.8-nvfp4full`, `tier8-dflash2`, `upstream-2026-08-29` | Early record |
| `c3c38b61e1d4294e899c60c28f239f50ddb55a1c` | 1100 | `tier13` | T13 record |
| — | — | `quasar-master`, `t33b-dflash2-upstream`, `t33bg2-quasar-port`, `audit/ninferno-*`, `backup/pre-fork-430298a`, `fork/nvfp4full-merged`, `pr-*` | no ADOPTION.md |

Satellite records (unchanged, citable): `~/.local/share/ninfer/{adoption-tier3-record.md,
adoption-waveb1-record.md, quasar-conversion-record.md, t8-dflash2-boot-debug-2026-08-31.md,
tier8-experiment-record.md, t11w-handoff-2026-08-30.md, hostkv32-record.md,
concurrency4-record.md, logs/t33df2h-notes.md, data-driven/ninver-t33df2e-supervisor.md}`;
repo policy docs: `NINFER-UPSTREAM-MERGE-PLAN.md` (re-land policy incl. the serve-patch
keep/drop table) and `AGENTS.md`.

**Important:** `master`'s *code* is **older** than the baseline (274 ahead / 93 behind
`quasar-master`; `b88c0f6f` is not an ancestor of `master`), and its post-2026-09-04 commits are
docs-only. `master` is historical for code; this file replaces it as the record.

## 4. Verdict rule (why patch-ids are not used)

`git cherry quasar-master <branch>` returns **zero** patch-id-matched commits for every lane
branch (measured; e.g. `t42wave-quasar` shows +210 unmatched / 0 matched), because all past
adoptions landed as **merges or adaptations**, not as identical patches. Patch-id equivalence is
therefore not valid evidence in this lineage. Absorption is judged by:

1. **Upstream content/PR-merge state** — a tier is `ABSORBED` when the upstream commits/PRs it
   tracked are inside `b88c0f6f` (checked with `git merge-base --is-ancestor` for named SHAs) or
   their PRs show as merged.
2. **Live PR state** — items tracking *open* upstream PRs are `RE-ADOPT` candidates (content not
   yet in the baseline).
3. **Lane applicability** — items whose value depended on the old MTP/spec stack, the old
   executor, or a reverted decision are `VOID` (with reason) or frozen as *conditional*.

Measured on 2026-09-09.

## 5. Re-baseline ledger, T1 → T49

Old verdicts transcribed from `master:ADOPTION.md` line 1496 (round-6 TIERED PLAN, updated through
Round 22 at line 2445) and the T33 Wave Plan at line 1827. V2 status vocabulary:
`ABSORBED` · `RE-ADOPT` · `VOID (reason)` · `REJECTED (reason)` · `WATCH` · `CARRIED` (config/policy
or research that survives the cutover) · `SPLIT` (partially each).

| Tier | What (one line) | Old verdict (2026-09-08) | V2 status (2026-09-09) |
|---|---|---|---|
| T1 | decode & serve quality (#55 #67 #69 #65 #57 #61) | DONE 08-23 (`tier1`, PR #1) | VOID — cutover superseded the code lineage; behavior re-validated on the new base by battery; #61/#54 still open upstream (§9) |
| T2 | xhigh track (Sharp v22.3.1 + `reasoning_effort`) | DONE 08-23 (`tier2`, PR #2) | VOID — superseded by cutover; the xhigh template policy survives as T37 (CARRIED) |
| T3 | community cherry-picks + Wave B1 perf | DONE 08-24 (`tier3`/`tier3-waveb`) | VOID — superseded; surviving items re-derived as V2-T4/V2-T5 |
| T4 | upstream convergence + agent-workload (C1–C3) | DONE 08-24 (`tier4`) | ABSORBED — upstream portion is inside `b88c0f6f`; host-KV opt-in carried by T14/T31 (WATCH) |
| T5 | Wave C (MoE-decode perf, response_format json) | DONE 08-24 (`tier5`) | VOID — superseded by cutover |
| T6 | portability ops (SM-count persistent grids) | DONE 08-25 (`tier6`) | VOID — superseded by cutover |
| T7 | MTP width-invariant greedy verification | DONE 08-26 (re-adopted; first adopt reverted, −50 % decode) | VOID — MTP stack no longer in lane (dflash2); the *gate* (acceptance-gated adoption) is carried as standing policy §10.5 |
| T8 | cometkim DFlash2 probe | CLOSED 08-31 (4 boot bugs) | VOID — superseded by upstream dflash2 (§2) |
| T9 | dylan XAttention prefill | DEFERRED (prefill/TTFT evidence first) | VOID — dylan prefill line is a dead path for the dense 27B target |
| T10 | upstream/dev sync | RESOLVED 08-29 (consumed by T12) | ABSORBED — inside `b88c0f6f` |
| T11 | quasar re-verification on the T10 merge | CLOSED — rolled back (vision 400) | VOID |
| T12 | upstream convergence wave | DONE 08-30 (battery 16/16) | ABSORBED — inside `b88c0f6f` |
| T13 | #98 wave (#107/#97/#72 + pressure fixes) | FAIL 08-31 → DEFERRED (`3d9fda22`/`5e4bf313` 503-bad under load) | WATCH — 503 evidence stands; #107/#97/#72 still open upstream (§9) |
| T14 | host-KV content cache (#73) | PROBE DONE 09-05 (idle prefix loss @ 45 s/120 s) → T31 | WATCH — host-KV re-enable decision gates T31 (V2-T7) |
| T15 | gzenz NVFP4 KV + YaRN (2.12, 400 k) | IN TREE 09-02, not live (quadlet pinned INT8 @ 225280) | CARRIED → RE-ADOPT (config-level) — YaRN-400k + NVFP4-KV profile is a config decision, not in `b88c0f6f`; the lane meanwhile runs nvfp4 KV @ 262144 (the cutover already realized the KV part) |
| T16 | upstream convergence wave 1 | DONE 09-02 (absorbed in t15-yarn; md micro-opts + GDN fix) | ABSORBED — upstream portion inside `b88c0f6f`; the GDN fix survives as T18 (RE-ADOPT) |
| T17 | pv-f16acc (md) | DONE 09-02 (t15-yarn); T25c 64k-needle PASS 09-04 | SPLIT — the fp16-KV core the picks were made against (`21a0e85f`) is ABSORBED (ancestor of `b88c0f6f`); the md re-picks `15729d9b`, `2e99db7b` are not in the baseline → RE-ADOPT (V2-T4) |
| T18 | dylan wave 2 (GDN chunked-prefill precision) | SHIPPED 09-05 — live lane (marker `e858f8`, no longer resolves post-cutover) | RE-ADOPT — `49400365` (accuracy fix; dylan's GDN line is a dead path upstream, so it is not in `b88c0f6f`) → V2-T4 |
| T19 | `gated_delta_net_snapshot` op + tests (dylan) | NOT STARTED — no new signal | VOID — GDN op is a dead path for the dense 27B target |
| T20 | watch: open upstream PRs | WATCH | CARRIED — re-mapped in §9 |
| T21 | watch: upstream issues | WATCH | CARRIED — re-mapped in §9 |
| T22 | upstream convergence wave 2 (863aa8a5) | DONE 09-04/05 (landed via the T-Q ship) | ABSORBED — `863aa8a5` is an ancestor of `b88c0f6f` (verified) |
| T23 | md TMA prefill pair (#167 + #160) | SHIPPED 09-05 (`t23tma-bb535075`) — extended by T36 | RE-ADOPT — `52fabe3e` (fp8-A8 TMA staging) + `bb535075` (nvfp4 tile-contiguous scales): #167 and #160 are **still open** upstream (verified 2026-09-09) → V2-T4 |
| T24 | gzenz host-KV safety net | SUPERSEDED → T31 | CARRIED — see T31 |
| T25 | probe wave | PARTIAL — T25c PASS, T14 done; remainder open | CARRIED — T25c PASS validated T17; the rest closed by the cutover |
| T26 | watch slot (open-PR watch set) | WATCH | CARRIED — §9 |
| T27 | watch slot (community-fork watch set) | WATCH | CARRIED — §8 |
| T28 | dylan dflash2 wave | SUPERSEDED by T33 | VOID — superseded by upstream dflash2 (§2) |
| T29 | Mirko dynamic-MTP decode wave | DECIDED 09-05 — T29a port pending; T29b levers no-win | VOID — MTP stack no longer in lane; the surviving `81e05f*` item frozen as MTP-conditional |
| T30 | Mirko KVaRN line | DEFERRED — `114b0fcb` (greedy-parity fix) would be needed if revived | REJECTED — sub-floor KV (KVaRN k4v2-g128) with no E2E quality evidence (§10.4) |
| T31 | gzenz host-KV safety-net port | BLOCKED → RE-EVAL 09-08 (B2 entitlement / B3 frontier on the old pick set; gzenz line was `51d6ba6a`) | CARRIED → V2-T7 — re-derive the pick set from `62b857c1` (117 ahead, 2026-09-09); only if host-KV re-enable is approved |
| T32 | upstream prefix/context-cache cluster (#176–#181, #142, #184) | WATCH | CARRIED — watch (§9); #181 mirrors our T14 finding |
| T33 | DFlash2 drafter grafted onto the QUASAR artifact | REJECTED 09-08 (Round 21) — permanently parked (R18/19 decode loss, 182k 9× collapse, 8k parity mismatch; master-rebase line `t33bg2-quasar-port` @ `4008da6d` passed decode gate but failed same-image parity 3/3) | REJECTED-SUPERSEDED — the lane now serves **upstream** dflash2 via the graft artifact (§2); the R18 parity defect is engine-level; revisit only if upstream fixes dflash2 verify-path greedy parity |
| T34 | host-KV restore correctness (reframes T31) | NEW — ADOPT the mitigation shape: host-RAM reuse = append-at-frontier only; port `ac60331d` as a guard | CARRIED — the mitigation is gpillon pick #6 (`f4b128c6`, §6) + `ac60331d`; lands with V2-T8 |
| T35 | draft window k=3→5 | REVERTED 09-06 — probe complete: battery 16/16, fresh +8.5 % but 8k −6.8 % + long-ctx acceptance degraded → k=3 frozen | VOID — probe verdict stands; the draft window is now upstream dflash2 `--draft-tokens 7` |
| T36 | md dense-lane ops wave | RE-SCOPED 09-08 (Round 22): `c735909b` re-pick superseded — upstreamed as `ee9d5192`; surviving wave = `67bf4b78` softmax-fold re-pick; `ce71f787` remains an acceptance-gated probe (T36b) | SPLIT — `c735909b` → ABSORBED (upstream `ee9d5192`, identical file set, in baseline); `67bf4b78` → RE-ADOPT (V2-T4); `ce71f787` (MTP sampled-draft) → WATCH, MTP-conditional |
| T37 | chat template → artifact-embedded ReasoningEffort @xhigh | ADOPTED 09-06 — live since 09-05 21:42; battery 15/16; decode-neutral vs Sharp | CARRIED — host-mounted template + quadlet config; unaffected by the cutover; pinned by sha256 (§2) |
| T38 | upstream `--chat-template FILE` (#183/#182) + stream-slot (#184) | NEW — WATCH/adopt-on-merge | CARRIED — #183 still open upstream (verified 2026-09-09) |
| T39 | Astrangemaninhere/ninfer-fusion | NEW — WATCH | REJECTED — sub-floor KV (perplexity-only evidence); its DFlash2 < MTP3 by its own data |
| T40 | dylan `cdd1b6c1` C1-4 speculative decode | RE-SCOPED 09-08 — standalone probe dropped (file set is GDN-centric) | VOID — dead path for the dense 27B target; the shared `linear.cpp`/`nvfp4_dispatch` slices ride upstream if merged |
| T41 | wall-time-to-accurate-answer (T2A) research & plan (W0–W6) | NEW 09-06 — research complete on the live lane | SPLIT — the research is CARRIED; the engine-side half = the gpillon agentic cluster → **V2-T2/T3/T8/T9** (§6–7) |
| T42 | #211 KV stream-ordering hotfix (fixes #210 agent GPU lockup) | P0 09-08 — vulnerable pattern verified in our engine; cherry-pick into the next image build | **SHIPPED 2026-09-09 (V2-T1; RE-ADOPT realized on the live lane)** — PR #211 (clean 3-file, base exactly `b88c0f6f`; the earlier #213 pollution is gone) landed as `64fa227a` + baseline repair `ba21e676` on branch `v2/t1-stream-kv` (image tag `v2t1-ba21e676`). Verified: battery 15/15 non-decode (REPLAY 4/4, 4XX-WATCH clean) + no decode regression (clean warm-idle, candidate at/above live on both depths). Residual: #210 crash-repro not yet exercised (no reliable repro) |
| T43 | upstream master convergence wave 3 (`a16b6442`→`7f14d963`) | NEW — cherry-pick wave (none in the gpillon line, merge-base `863aa8a5`) | SPLIT — `ee9d5192`, `b158afe2`, `641ef3e7`, `7f14d963` are ABSORBED (all ancestors of `b88c0f6f`, verified); `0f84adaf` (#195 context-cost weights fallback) is **not** in the baseline (#195 still open) → RE-ADOPT (V2-T4) |
| T44 | md MTP/decode ops perf (triage-verified 09-08: `qwen3_6/impl/runtime/` is the shared family base) | NEW — cherry-pick candidates on `a16b6442`, acceptance-gated | SPLIT — decode items `38f52b34`, `61250e89`, `ed150906` → V2-T5; MTP items `505d1af7`, `1f155fed` → V2-T9 (MTP-conditional; note our in-lane copy of `1f155fed` is `fa12e8ef`) |
| T45 | #174 full-vocab Q4G64 MTP proposal head | NEW — PROBE (acceptance-gated; restores structured-output coverage `--lm-head-draft` loses) | CARRIED — MTP-conditional; implementation lives on the reporter's external fork (release `b5f2d1*`, SHA no longer resolvable — re-fetch from the #174 thread before any pick) |
| T46 | cometkim PDL decode chain (`feat/kernel-perf`) | NEW — WATCH→PROBE; +77 %/+56 % claims unverified; 3 force-pushes since | CARRIED → **V2-T6** — re-derive on the post-force-push tip `6c3fdbf4` and verify the claims on our base before any window |
| T47 | nvfp4qat QUASAR-QAT artifact profile (cometkim) | NEW — A/B candidate (low effort); still QUASAR-lineage, so the operator constraint holds | CARRIED → **V2-T6** — post-force-push tip `c17ccc30`; our `f7727926` already carries the `Qwen38Nvfp4*` profile family, so compare before adopting |
| T48 | dylan/experimental agentic-runtime slices | NEW — WATCH (feeds T41): `42c9c7d4` (exit p-less thinking cycles), `a39c5c25` (W4/W6 multi-request decode projection aggregation) | CARRIED — branch is 126 ahead of `b88c0f6f` (tip `42c9c7d4`, 09-07; unchanged on 09-09 fetch); GDN/dflash/qwen4 bulk = dead path; the two slices feed V2-T8 |
| T49 | #208 stability watch — 5090 NVFP4+MTP `cudaErrorIllegalAddress` | NEW — STABILITY (top priority alongside T42); may share the #210 root cause | CARRIED — tracks V2-T1; if it surfaces on this lane, engage upstream #208 with our repro data |

## 6. gpillon agentic adoption (the priority)

Fork state (2026-09-09 fetch: **no movement** on gpillon since 09-03): `gpillon/gpillon/coding`
tip `a00648cb`, 103 ahead / 234 behind the baseline, merge-base `a05746aa`.
The agentic cluster = the 20-pick Wave-A list from `master:ADOPTION.md` line 1837.

### 6.1 Measured feasibility on the new baseline

Harness (read-only; fresh temp index per pick, repo untouched — Appendix A): patch each pick
against the `quasar-master` tree with `git apply --3way`. **Result (per pick, measured in
isolation 2026-09-09):** 17 of the 20 picks fail only on paths that do not exist in the baseline;
picks 15–17 additionally carry genuine 3-way conflicts on existing serve/device files (upstream's
serve/device-layer refactor has diverged from gpillon's base `a05746aa`). The old "29 conflicted
files" figure does not reproduce. What is missing is **substrate**, and what conflicts is the
refactored serve/device layer:

| # | Pick | What (Wave-A line) | Files | 3-way merged | Missing paths (the only failures) | Class |
|---|---|---|---|---|---|---|
| 1 | `de386ad6` | system RAM KV cache for finished chats (tier base; cherry-pick of dylan `14329810`) | 54 | 52 | `src/runtime/engine/concurrent_executor.h`, `docs/maintainer/concurrent-inference-architecture.md` | HAND-PORT |
| 2 | `f144f052` | KV RAM used size + copy times in serve logs | 22 | 15 | `concurrent_executor.h`, `docs/maintainer/concurrent-inference-architecture.md`, `kv_ram_cache.{h,cpp}`, `kv_ram_snapshot.h`, 2 tests | DEPENDENT #1 |
| 3 | `27665883` | two-tier probation/protected eviction by content lineage | 2 | 0 | `kv_ram_cache.{h,cpp}` | DEPENDENT #1 |
| 4 | `7bdee888` | active-lane prefix sharing for identical concurrent requests | 7 | 4 | `concurrent_executor.h` + `kv_ram_cache.{h,cpp}` | DEPENDENT #1 |
| 5 | `96371a3d` | one shared sibling snapshot instead of per-sibling capture | 4 | 1 | `concurrent_executor.h` + `kv_ram_cache.{h,cpp}` | DEPENDENT #1 |
| 6 | `f4b128c6` | **T34 guard**: stop offering rewrite-checkpoint restores from host RAM | 1 | 0 | `kv_ram_cache.cpp` | DEPENDENT #1 |
| 7 | `68b12497` | disable the RAM tier while an exact-key side store is in use | 4 | 3 | `kv_ram_cache.cpp` | DEPENDENT #1 |
| 8 | `eaf2037b` | record format v2 (hyperquant side-store carry) | 6 | 3 | `kv_ram_cache.{h,cpp}`, `tests/test_kv_ram_cache.cpp` | DEPENDENT #1 |
| 9 | `07aeac2d` | preserve coding-agent prefix state (system/tools shared boundary) | 17 | 9 | `HANDOFF.md`, `concurrent_executor.h`, `kv_ram_cache.{h,cpp}`, 4 kv-ram tests | HAND-PORT |
| 10 | `2065ed38` | capture dynamic shared-prefix boundaries | 17 | 8 | 1 `.bat`, 3 stale `*.bak-v2-*` (skip), `concurrent_executor.h`, `kv_ram_cache.{h,cpp}`, `test_kv_ram_cache.cpp` | HAND-PORT |
| 11 | `2728ace4` | exact-size RAM captures + PreserveExisting admission | 16 | 10 | `HANDOFF.md`, `concurrent_executor.h`, `kv_ram_cache.{h,cpp}`, `test_kv_ram_cache.cpp`, `test_kv_ram_cache_opt.cpp` | HAND-PORT |
| 12 | `093c1fdd` | sibling-prefix overlap telemetry (no behavior change) | 3 | 2 | `concurrent_executor.h` | PATH-REMAP |
| 13 | `7a4634b5` | tagged request lanes @main/@agents/@classifier | 16 | 12 | `concurrent_executor.h` + `src/serve/{openai,anthropic,responses}_schema.cpp` (upstream moved the schema files) | HAND-PORT |
| 14 | `5f014910` | tool-call XML leak fixes (2 bugs incl. stream-teardown crash) | 2 | 1 | `src/serve/tool_call_parser.cpp` (upstream moved it to `src/targets/qwen3_6/impl/frontend/tool_call_parser.{cpp,h}`) | PATH-REMAP |
| 15 | `6a1b62c5` | decouple warmup from client-facing request deadline | 2 | 0 | none — but 2 existing files conflict: `src/serve/generation_service.{cpp,h}` | CONFLICT |
| 16 | `27417ca2` | warmup fail-fast + auto kv-capacity bounds | 4 | 0 | none — but 4 existing files conflict: `apps/cli/options.cpp`, `apps/serve/main.cpp`, `src/serve/generation_service.cpp`, `src/serve/serve_options.cpp` | CONFLICT |
| 17 | `adf494c2` | block host sync — fixes the 100 %-CPU decode (adapts dylan's `58383e*`, lineage deleted) | 2 | 0 | none — but 2 existing files conflict: `src/core/device.{cu,h}` | CONFLICT |
| 18 | `c2708ec8` | adaptive MTP verification widths | 60 | 55 | `docs/maintainer/concurrent-inference-architecture.md`, `tools/reference/qwen3_6_27b/{README.md,cli.py,model.py,mtp.py}` | HAND-PORT (MTP-conditional) |
| 19 | `9d86436c` | price adaptive widths by context depth | 12 | 9 | `docs/maintainer/concurrent-inference-architecture.md`, `src/targets/qwen3_6/impl/runtime/mtp_adaptive.h`, `tests/targets/qwen3_6/test_mtp_adaptive.cpp` | DEPENDENT #18 |
| 20 | `9bef0f73` | calibrate round-cost from measured round duration | 2 | 1 | `mtp_adaptive.h` (`program_impl.h` merges) | DEPENDENT #18 |

**Structural blocker, quantified:** 11 of the 20 picks patch
`src/runtime/engine/concurrent_executor.h`, which does not exist in the baseline — upstream's
executor refactor replaced it with `src/runtime/engine/{scheduler.h,resource_manager.h}`
(+ `tests/test_resource_manager.cpp`). That confirms Round 9's "hand-port, not cherry-pick" call
and localises it: the hand-port is about *creating the substrate* (`kv_ram_cache.{h,cpp}`,
`mtp_adaptive.h`, executor re-targeting). The conflict surface is real but bounded: picks 15–17
(7 unique files, 8 per-pick hits: `src/serve/generation_service.{cpp,h}`, `apps/cli/options.cpp`,
`apps/serve/main.cpp`, `src/serve/serve_options.cpp`, `src/core/device.{cu,h}`) 3-way-conflict
because upstream refactored the serve/device layer after gpillon's base `a05746aa` — per-hunk
manual resolution, no substrate required.

**None of the 20 picks touches a `dflash*` path** (verified per pick), so the agentic cluster is
separable from gpillon's DFLASH2 engine by construction.

### 6.2 Sub-tier split (feasibility-ordered)

- **V2-T2 — RE-DERIVED 2026-09-09: pick 17 only.** The §6.1 dry-run measured the conflict *surface* (picks 15–17 3-way-conflict on 7 existing serve/device files). Re-deriving the *content* against the real base: picks 15 (`6a1b62c5`, warmup/deadline decouple) and 16 (`27417ca2`, warmup fail-fast + auto kv-capacity) are **SUPERSEDED by the base** — warmup already uses `DeadlinePolicy::UnboundedStartup` (deadline decoupled; the 60s cap intentionally not adopted) and `apps/serve/main.cpp` already fails fast (`return 1`) on warmup failure (residual = cosmetic usage-text strings). So V2-T2 = **pick 17 (`adf494c2`, the 100 %-CPU blocking-sync fix) only**, adapted to the 2-stream base (gpillon's single-stream `load_stream` → our `transfer_stream`; CUDA 13.1.2 3-arg `cudaInitDevice`). **SHIPPED 2026-09-09** (2nd attempt, supervised retag+restart window; branch `v2/t2-agentic` @ `0faef6d4`, image `408c7df8` = `v2t2-0faef6d4` = `:quasar`). First ship: built + ctest PASS + booted, battery **15/16** — the sole red was the cold **decode-fresh** probe (145.4 vs the 153.9 gate; decode-8k 157.3 passed). Both candidate probes sat ~5–10 % below the V2-T1 baseline (fresh 162.0 / 8k 165.3) → a small **per-round latency from the blocking sync**, a real but modest trade-off against the host-core savings. **User accepted the trade-off → re-shipped.** Verified live: during decode the host `ninfer-serve` process sits at **~0.9 % CPU (state S, sleeping)** — the 100 %-core busy-wait is gone (the fix's value); frees up to 4 cores at `--max-concurrency 4`.
- **V2-T3 — PATH-REMAP:** picks 14, 12 (`5f014910`: remap the parser to
  `src/targets/qwen3_6/impl/frontend/tool_call_parser.cpp` — its test file 3-way-merges;
  `093c1fdd`: drop the `concurrent_executor.h` hunk, keep the 2 merging files).
- **V2-T8 — HAND-PORT cluster:** picks 1–11 + 13 (the RAM-KV tier + tagged lanes), including the
  T34 guard (pick 6). **Design decision recorded 2026-09-09 (§6.5): integrate, not replace** — re-target
  the 12 picks onto the split executor (`scheduler.h`/`resource_manager.h`/`EngineCore`), plug the
  two-tier eviction into `ResourceManager` pressure planning, and share the `--host-kv-mib` pool with the
  `HostKVExtentStore` demotion mirror. Effort: weeks.
- **V2-T9 — MTP-conditional:** picks 18–20 (adaptive MTP widths trio) — only if the lane returns
  to `--spec mtp`; today it runs upstream dflash2.

### 6.3 DFLASH2 exclusion (mechanical, permanent)

gpillon's DFLASH2 engine lives on `gpillon/feat/dflash2` (`62acfe15`) and
`gpillon/feat/dflash2-local` (`43b03ea5`). It is incompatible with the upstream dflash2 engine the
lane runs. Rule: **no `dflash*` path is ever adopted from gpillon; the lane's dflash2 comes from
upstream only** (§10.5). The agentic cluster above is verified dflash-free.

### 6.4 Linux gate

gpillon's `src/serve/webui_update.cpp` is Windows-first; their own
`gpillon/fix/linux-webui-platform-guards` (`4c824c6b`) and our recorded
`/tmp/gpillon-wt-linux.patch` (+33/−1) cover it. None of the 20 picks above touches that file, but
any broader gpillon adoption must carry the fix.

### 6.5 V2-T8 hand-port design note (recorded 2026-09-09)

Grounded by a read-only map of `v2/t3-path-remap` @ `458376c4` + `gpillon/gpillon/coding` @ `a00648cb`.
**Conclusion: integrate, not replace.** The baseline has no `ram_kv`, but already owns both halves the
hand-port touches; the work is re-targeting 12 picks onto a split executor, not building new substrate.

**F1 — the baseline already owns both host-RAM roles.**
- *Logical* eviction substrate (present): `Scheduler<Request>` (`scheduler.h`, admission/round; driven
  from `engine_core.h:1925` `worker_loop` → `build_round_membership`), `ResourceManager<Package>`
  (`resource_manager.h`, cache policy + pressure planning), `admission_policy.h` (backfill/protection).
  Four `RetentionClass` (`contract/types.h:360`: `SharedStable`/`LiveSession`/`RecentPrivate`/`Disposable`)
  with private-retention weights **`SharedStable`=0 / `LiveSession`=16 / `RecentPrivate`=4 / `Disposable`=1**
  (`resource_manager.h:1419-1430`), plus hit-history (`RetentionObservation`), publication-order, and
  degrade-or-evict pressure counters (`pressure_*_owners_{degraded,evicted}`, `include/ninfer/types.h:910-912`).
- *Spatial* host-KV demotion mirror (present): `HostKVArena` (`src/core/host_kv_arena.h`, fixed pinned pool,
  `--host-kv-mib` → `host_kv_capacity_bytes`, default 8 GiB `include/ninfer/types.h:28`, lane 16 GiB; pure
  allocator, no eviction) + `HostKVExtentStore` (`host_kv_extent_store.h`, logical-pages↔host-extents, D2H
  publish / H2D restore; header: "no checkpoint, retention, or scheduling policy"). A per-owner device↔host
  safety net, not a cache.
- `concurrent_executor.h` is **gone**: 0 matches in the baseline tree; scheduling half → `Scheduler`,
  cache-policy half → `ResourceManager`, protection half → `admission_policy.h`, worker thread/loop →
  `EngineCore` (`engine_core.h:1925`/`:2055`).

**F2 — gpillon's RAM-KV cluster (picks 1–11 + 13) is a third, *temporal* role.**
`KVRamCache` (over its own `HostPinnedArena`) caches finished/agentic-chat KV by content hash, with a
distinct two-tier eviction (`evict_unpinned()` FIFO-cold where `live_count==0` vs protected/multi-claim;
`RamCapturePolicy{AllowEviction,PreserveExisting}`), active-lane prefix sharing
(`RamCaptureKind{Terminal,ActiveSibling,SharedBoundary,DynamicBoundary}`), and coding-agent prefix
preservation (`RewriteCheckpoint`/`TurnClosure`, `RequestClass::Agents`, T34 guard = pick 6 `f4b128c6`).

**Decision (integrate):**
1. *Eviction plugs into, doesn't duplicate.* Map capture kinds → existing `RetentionClass`; drive the
   two-tier probation/protected eviction through `ResourceManager` pressure planning (hit-history +
   publication-order → `VictimDisposition{Retained,Evicted}`), not a parallel policy.
2. *RAM pool is shared, not forked.* `KVRamCache` draws on the same `--host-kv-mib`/`HostKVArena` pool as
   the `HostKVExtentStore`; the demotion mirror is the higher-priority tenant, `KVRamCache` yields under
   pressure. A separate pool risks unbounded host-RAM — A/B shared-vs-separate on a multi-lane agentic load.
3. *Wiring goes to `EngineCore::worker_loop`* (`engine_core.h:1925`) + `ResourceManager` — where the
   executor's stats/eviction duties already live; the dropped executor is not resurrected.
4. *The 11-of-20 `concurrent_executor.h` dependency is the bulk of the hand-port:* each pick's executor
   hunk re-targets to its correct half (scheduling → `Scheduler`/`EngineCore`, cache-policy →
   `ResourceManager`, RAM-snapshot/stats → `KVRamCache` + `EngineCore`).

**Gate:** TTFT / decode / cache-hit-rate A/B + battery + greedy parity (V2-T8 is decode-affecting via the
cache-hit path); the shared-vs-separate pool decision settled by a host-RAM-accounting A/B.

**Scoping re-verify (2026-09-09, post-V2-T5):** re-confirmed on the current baseline - no upstream drift
(`quasar-master...upstream/master` = 1/0, tip still `b88c0f6f`), executor surface unchanged
(`concurrent_executor.h` = 0, `scheduler.h`/`resource_manager.h` = 1 each, all 20 gpillon SHAs resolve), and both
F1 baseline halves are present (`Scheduler`/`ResourceManager`/`admission_policy` + `HostKVArena`/`HostKVExtentStore`).
**Magnitude:** the 4 cluster picks total **~+8,000 lines** (`de386ad6` alone +7483/-227 across ~45 files incl. the
executor + serve + qwen3_6 runtime + tests; `f144f052` +284/-59; `27665883` +68; `7bdee888` +158), and the
`kv_ram_cache.{h,cpp}` core is 1463 lines (`kv_ram_cache.h` 314 + `.cpp` 1149) + `kv_ram_snapshot.h` 28 + 4 tests.
This is a **multi-week hand-port, not a quick pick**. Sequencing: (1) port `kv_ram_cache.{h,cpp}` + integrate with
`HostKVArena`/`HostKVExtentStore` on the shared `--host-kv-mib` pool (decision 2); (2) re-target the 4 cluster picks
onto the split executor (each `concurrent_executor.h` hunk -> its correct half: scheduling -> `Scheduler`/`EngineCore`,
cache-policy -> `ResourceManager`, RAM-snapshot/stats -> `KVRamCache` + `EngineCore`); (3) the self-contained picks
(tool-call leak, warmup decouple/fail-fast, block-host-sync, MTP-widths) are independent and can land separately.
**Branch base:** the current lane state (V2-T5), so the T8 picks stack on the shipped T1/T2/T3/T5.

### 6.6 V2-T8 port progress + dependency-surface findings (2026-09-09 evening)

**Committed progress on `v2/t8-agentic` (branched off `v2/t5-decode`):**
- `kv_ram_cache.{h,cpp}` + `kv_ram_snapshot.h` ported verbatim (the hand-port's core; the substrate
  the 4 cluster picks build on). **NOT self-contained** (verified by per-TU syntax check, 09-09
  evening, buildstage container): `RamCaptureSource` pulls in gpillon's paged-KV pool
  (`PagedKVAllocation`/`PagedKVPool`/`PagedKVCache` in `src/core/paged_kv_cache.h`), the runtime's
  core types (GDN `LinearAttentionStatePool`, dflash `CyclicKVCache`, `Tensor`), and
  `runtime::RequestClass` — none of which the baseline provides (its paged-KV cache was restructured
  upstream, a different design). The substrate is a deep integration, not a 3-file port.
- `HostPinnedArena` ported into `core/arena.{h,cu}` (the gpillon-specific first-fit pinned-host
  block allocator the substrate draws on). All five helpers it uses (`cuda_error_message`,
  `free_pinned`, `is_power_of_two`, `checked_add_uintptr`, `align_up_addr`) already exist in the
  baseline's `arena.cu`, so it's a verbatim class + impl port. Interim host-RAM pool; the
  shared-vs-forked decision (§6.5 decision 2) is a later A/B.

**The core difficulty — the `prefix_identity` design divergence.** The baseline's
`ResidentPrefixIdentity` (execution-split: `rewrite_execution_frontiers_` + `equals`/`prefix_equals`
+ the separate `PrefixShortlistDigests` + `append_generated(…, execution_split_after)`) and
gpillon's (packed/hash: `pack`/`unpack`/`packed_bytes` + `PrefixHash128` + `prefix_hash_chain`/
`prefix_hash_at` + the FNV mix helpers) are **divergent designs, not a superset** (conflicting
`matches` impls + helper names). The substrate needs gpillon's packed/hash; the baseline's decoder
(7 files: `engine.cpp`, `frontend.cpp`, `api_impl.h`, `program.h`, `program_impl.h`,
`request_plan_impl.h`, `frontend.h`) needs the baseline's execution-split. **Reconciliation
(decided): keep the baseline's execution-split `ResidentPrefixIdentity` + `PrefixShortlistDigests`,
and ADD gpillon's packed/hash additions** — the shared members (`token_types_`/`positions_`/
`vision_items_`) are identical, so `pack`/`unpack`/`packed_bytes` + `PrefixHash128` + the FNV
helpers + `prefix_hash_chain`/`prefix_hash_at` + the `token_types()`/`positions()`/`vision_items()`
accessors slot in without disturbing the execution-split methods; the substrate's 2-arg
`append_generated` calls work against the baseline's 3-arg signature (3rd arg defaults `nullopt`);
the substrate's `prefix_matches(prompt, std::vector<TokenId>, …)` call converts to the baseline's
`std::span` signature.

**`PrefixReusePath` value divergence:** the baseline's `contract/types.h` uses
`PrefixReusePath::Root` (lines 452, 581); the substrate uses `FullReset`/`AppendAtFrontier`/
`RestoreTurnCheckpoint`/`RestoreResponseCheckpoint` → add the missing enum values (or re-target
the substrate to the baseline's values).

**`concurrent_executor.h` dependency:** the baseline has **no** `concurrent_executor.h` (0 matches);
the 11 cluster picks that patch it (verified per-pick `git show --name-only`) touch it + `admission_policy.{h,cpp}`
+ `api_impl.h` + `runtime.h`. The executor was restructured upstream (the `resource_manager.h`
rename), so each pick's executor hunk re-targets to its correct half (scheduling → `Scheduler`/
`EngineCore`, cache-policy → `ResourceManager`, RAM-snapshot/stats → `KVRamCache` + `EngineCore`).

**Progress + honest scope (per-TU syntax check done 09-09 evening, buildstage container):**
- ✅ DONE + VERIFIED (rc=0): the `prefix_identity` merge — the baseline's execution-split
  `ResidentPrefixIdentity` + gpillon's packed FNV/hash impls + the FNV helpers + `Writer`/`Reader`
  (with the `<cstring>` include) compile clean against the baseline's design; the 3
  `PrefixReusePath` values added; `kv_ram_cache.cpp` added to `qwen3_6/CMakeLists.txt`.
  Branch `v2/t8-agentic` (`9f028cc4`).
- The substrate (`kv_ram_cache`) is **deeply entangled** (the finding above): it needs the paged-KV
  pool subsystem + the runtime's core types (GDN/dflash/Tensor/`RequestClass`) — a **deep re-target**
  onto the baseline's restructured paged-KV cache, not a 3-file port.
- **Substrate re-target type mapping (the foundation, 09-09):** the baseline already has
  `LinearAttentionStatePool` (GDN, 6 files), `CyclicKVCache` (dflash, 7 files), `Tensor`, and its own
  `src/core/paged_kv_cache.h` — but a **different paged-KV design** (`DeviceKVPagePool`/
  `DeviceKVPageHandle`/`DeviceKVPageLease`/`KVExecutionTablePool`, vs gpillon's `PagedKVAllocation`/
`HostKVExtentStore`. `concurrent_executor.h` is **gone**: 0 matches in the baseline tree; scheduling half → `Scheduler`,
  `PagedKVAllocation*`/`PagedKVPool*` (`RamCaptureSource.text/text_pool/backend/backend_pool`) to the
  baseline's `DeviceKVPageHandle`/`DeviceKVPagePool` page-pool API (the main work); (2) add the small
  `RequestClass` enum (0 baseline files) or re-target the `owner_class` filter; (3) map the GDN/dflash/
  Tensor usage (the baseline has these types, API re-target if needed). Bounded — days to weeks for
  the substrate re-target, not months.
- **RequestClass increment committed (09-09, `a3ee165f`):** the 3-value enum added to the baseline's
  `include/ninfer/types.h` + the substrate's `runtime::RequestClass` re-targeted to `RequestClass`
  (the `choose_boundary_capture` ref is a doc-comment, not a call). The per-TU syntax check now shows
  the substrate fails **only** on the API surface: the baseline's `PagedKVCache` lacks
  `residual_enabled`/`residual_slot_host_bytes`/`ring_valid_slot_host_bytes`/`pack|unpack_residual_slot_
  from_host`; `LinearAttentionStatePool` lacks `conv_host_image_bytes`/`recurrent_host_image_bytes`/
  `pack_slot_to_host`/`unpack_slot_from_host`; `CyclicKVCache` lacks `copy_lane_to_host`/
  `copy_lane_from_host`/`lane_host_bytes`. So the substrate re-target is a **broad API surface**
  (paged-KV + GDN + dflash + PagedKVCache methods), not just the paged-KV pool — the re-architecture
  of the substrate's host-image pack/unpack + page-pool usage onto the baseline's designs.
- **Scope correction (09-09, per-TU syntax check):** the substrate's host-image pack/unpack API
  (`pack_slot_to_host`/`copy_lane_to_host`/`residual_*`/`ring_valid_slot_host_bytes`/…) is **genuinely
  absent** in the baseline — not a name mapping, but a re-architecture onto the baseline's
  state-image/host-KV mechanism (`host_kv_extent_store`/`state_image`/`host_kv_arena`, a different
  design). So the substrate re-target is **weeks, not days** (the host-image re-architecture + the
  paged-KV page-pool re-target), and the whole V2-T8 port (substrate re-target + the 4 cluster
  picks' executor re-targeting + buildstage build + ctest + battery + A/B + shipwatch) is
  **a month+**. Committed so far: substrate + `HostPinnedArena` + `prefix_identity` merge (verified
  rc=0) + `PrefixReusePath` + `RequestClass` + CMake.
- **Baseline host-RAM mechanism READ (09-09, per advisory):** read `host_kv_extent_store.h` +
  `state_image.h` + `host_kv_arena.h` directly. The baseline **already serializes the same
  structures to host RAM** via two mechanisms: (1) the **state image** (`StateImageDevicePool` +
  `HostStatePool`) — per-slot `copy_to_host`/`copy_from_host` over a pinned buffer, serializing GDN
  (`LinearAttentionStatePool` conv + recurrent) + continuation-hidden `Tensor` + dflash-local
  `CyclicKVCache` (`dflash_local_k`/`_v`) — an **exact** counterpart of the substrate's GDN/hidden/
  dflash-local host image; (2) the **host-KV extent store** (`HostKVExtentStore` + `HostKVArena`) —
  the host-KV page-replica demotion path (`prepare`/`publish`/`device_sources`/`writable_view`/
  `release`), serializing paged-KV (text/backend) pages — the equivalent of the substrate's paged-
  KV host image. So the re-target is **not** "write new CUDA host-copy routines from gpillon's byte
  layout" (the corruption risk): the baseline already owns both serializations with its own layouts
  (`StateImageHostLayout` + `HostKVPageLayout`). The real re-target work is the **policy layer** —
  the substrate captures a *whole active lane* as one cache image for prefix restore, vs the
  baseline's per-slot state image + per-extent demotion — plus two genuine gaps: the substrate's
  `dflash_checkpoint` (the baseline state image carries only `dflash_local`) and the whole-lane
  paged-KV checkpoint policy (vs the per-extent demotion). This makes the re-target **smaller than
  "weeks of new serialization"**: a policy + gap bridge onto existing baseline mechanisms, not a
  from-scratch byte layout.
- **Both bridge halves PROVEN by compiling spikes (09-09, exit 0, md5-verified against quasar-master
  headers):** the state-image bridge (`spike_stateimage_bridge.cpp`: `StateImageDevicePool::
  copy_to_host`/`copy_from_host` + `HostStatePool::allocate`/`writable_view`/`view` for the
  GDN/hidden/dflash-local slice) and the host-KV extent store bridge (`spike_hostkv_bridge.cpp`:
  `HostKVExtentStore::prepare`/`device_sources`/`writable_view`/`publish` for the text/backend
  paged-KV slice) both compile cleanly against the real baseline API. The two-mechanism mapping is
  now proven, not just inferred: state half -> the baseline's state image (slot-granular), paged-KV
  half -> the baseline's host-KV extent store (page-granular). Remaining: wiring both into
  `kv_ram_cache.cpp` (replacing the substrate's `make_capture_header`/`pack`/`unpack`) + closing the
  two gaps (the `dflash_checkpoint` + the whole-lane checkpoint policy) + the cluster picks'
  executor re-targeting + the supervised build/ctest/battery.

**FINAL — WIP port complete, full build GREEN (2026-09-09, commit `a5ba1ee5` on `v2/t8-agentic`):**
the previously-uncommitted WIP half (the `de386ad6` + `f144f052` engine layer, never committed to
any branch) is now ported: `kv_ram_cache.{h,cpp}` fully re-targeted onto the baseline
(`create_active`/`mapped_pages`/`physical_page` page-pool API; KV-page capture/restore via the
baseline's `LogicalKVPageStore` + `DeviceKVPagePool` host-copy; state half via the state-image
bridge, kRamVersion 4); `request_plan_impl.h` `plan_ram_reuse()` (terminal admission-time prefix
match → `reuse_source=HostRam` + `ram_entry_id`, MTP-aware); `program_impl.h`
`capture_retained_lane()`/`restore_ram_entry()` wired into the lane lifecycle; stats plumbing
`kv_ram_snapshot()` → `RuntimeStats` + the `--kv-ram-mib` serve option (0 = disabled, default)
through `SequencePlanningInputs` → `SequencePlanImpl` → `ProgramImplCore`.
Verified: per-TU `g++ -fsyntax-only` clean on `kv_ram_cache.cpp`, the 27b `variant.cpp` (pulls
`program_impl.h` + engine), `serve_options.cpp`, `generation_service.cpp`; and a **full
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=ON` + `cmake --build --target
ninfer ninfer-serve` = RC=0** (buildstage container, CUDA 13.1, arch 120a).

**Test suite port (2026-09-09, commit `857f8ca7` on `v2/t8-agentic`):** the re-targeted unit
suite `tests/targets/qwen3_6/test_kv_ram_cache.cpp` (registered as
`ninfer_qwen3_6_kv_ram_cache_test`, `SKIP_RETURN_CODE 77`) covers the substrate's observable
contract: paged-KV image round trip through the shared `HostKVArena` (including captures with
non-contiguous physical page runs), the complete `StateImage` round trip (linear conv/recurrent,
continuation hidden, DFlash local cyclic K/V), honest `prefix_hash_chain` index behavior
(longest match, frontier beats checkpoint, checkpoint fallback, exclusive-claim hiding,
consume-erase, multi-claim stays matchable), two-tier FIFO eviction with
lineage-demonstrated protection, and destructor safety with in-flight copies. Compile + link
GREEN (buildstage container, `BUILD_TESTING=ON`); the no-GPU SKIP path verified (rc 77). GPU
execution is part of the supervised runtime gate below.
**V2-T8 status: ADOPTED (code-complete, build-verified on `quasar-master`).** Remaining before
ship: runtime A/B (host-RAM hit path: TTFT/decode/cache-hit) + battery 16/16 + greedy parity —
the lane gate of §6.5, executed via the supervised pipeline (§10.2).

## 7. V2 tier plan (the next tiers, in adoption order)

Order: stability → cheap agentic wins → re-adopt our own still-unique work → external perf →
the big hand-port. Every tier ships through the supervised pipeline (§10.2).
**Status as of 2026-09-09:** **V2-T1 has shipped** (branch `v2/t1-stream-kv` @ `ba21e676`, image tag `v2t1-ba21e676`); **V2-T2 SHIPPED 2026-09-09** (2nd attempt; branch `v2/t2-agentic` @ `0faef6d4`, tag `v2t2-0faef6d4`, image `408c7df8` = `:quasar`): first ship rolled back on the battery's cold decode-fresh gate (15/16), the user accepted the ~5 % decode trade-off for the host-core savings, and a supervised retag+restart window deployed it — verified live, host ~0.9 % CPU during decode (busy-wait gone); pick 17 only, §6.2). **V2-T4 REJECTED (measured 2026-09-09):** the 2 solid on-path picks (`49400365` GDN + `d3278b79` rope; branch `v2/t4-readopt` @ `aa27864c`, image `68cec959`) built clean + ctest-clean, but the battery's DECODE-FRESH + DECODE-8K gates FAILED (140.6 / 142.8 tps vs the 162.0 / 165.3 V2-T2 baseline — a ~9 % decode regression beyond V2-T2's accepted ~5 % trade-off) → ship **ROLLED BACK** (lane back on V2-T2 `408c7df8`, verified). Our old T18/t42wave perf picks regress on the evolved baseline; the remaining V2-T4 picks are off-path / hand-port / content-superseded (see the row). **V2-T3 SHIPPED 2026-09-09** (user overrode the red DECODE-8K gate — it was red only from a power-constrained GPU state, not a V2-T3 regression; the fix is decode-neutral): tool-call XML leak fix, branch `v2/t3-path-remap` @ `458376c4`, image `12b87f4d` = `:quasar` (now live). Deployed via a supervised retag-only window; the first window run rolled back on a script self-check bug (an unexpanded `~` in a double-quoted template path tripped the ERR-trap rollback, and a missing `set -e` then clobbered the marker with a false DEPLOYED) — the script was fixed + retried clean (~23s, no rollback). Verified live: serving image `12b87f4d`, `/v1/models` → `qwen3.8-27b`, template sha256 `180e7015…` byte-identical, quadlet untouched, decode smoke OK. V2-T5…V2-T9 remain pending.
**V2-T5 SHIPPED 2026-09-09 (gate override, A/B-verified):** the 3 on-path decode picks (`38f52b34` argmax winner-init kernel + `61250e89` nvfp4 SwiGLU expf + `ed150906` w8 rowsplit cache policy; branch `v2/t5-decode` @ `324a8de3`, image `f8b76e5a4dc2` = `:quasar`, retag-only deploy) — the build+ctest+battery pipeline auto-rolled back on a **DECODE-8K-only red** (132.5 vs stored baseline 153.8; 15/16 battery green incl. replay 4/4 + vision + soak 5/5), and a same-window A/B decode-differential with concurrent `nvidia-smi` power/clock sampling proved it a **stale-baseline power artifact, not a regression**: in the same throttled window, live (V2-T3) 8k = 112.1 tps vs candidate 8k = 135.5 tps (**+21 %**), fresh 156.4 vs 128.9, with the candidate at an equal-or-better power state (491 W / 2902 MHz vs 526 W / 2865-2872 MHz). A supervised retag-only window deployed it; verified live (`/v1/models` 200 qwen3.8-27b, quadlet byte-identical, chat_template sha256 `180e7015…`).
**V2-T8 ADOPTED (code-complete, build-verified 2026-09-09):** the full WIP port landed on
`v2/t8-agentic` — `71f8e0b1` (paged-KV bridge onto the baseline `HostKVArena` + `DeviceKVPagePool`)
and `a5ba1ee5` (WIP host-RAM prefix reuse: `kv_ram_cache.{h,cpp}` re-target + `plan_ram_reuse` +
`capture_retained_lane`/`restore_ram_entry` + `--kv-ram-mib` serve option + stats). Full
`ninfer` + `ninfer-serve` Release build in the buildstage container (CUDA 13.1, arch 120a)
**RC=0** (log `/tmp/v2-t8-wt/build.log`). Not yet live: the runtime gate (§6.5 — host-RAM A/B +
battery 16/16 + greedy parity) is the next supervised window.
Ported unit test suite `857f8ca7` (`ninfer_qwen3_6_kv_ram_cache_test`: KV/state-image
round trips, irregular page runs, hash-chain index behavior, tiered eviction, lifecycle,
dtor safety; compile + link + no-GPU SKIP path verified) — GPU ctest execution is part of
the same gate.

| Tier | Content | Source | Gate |
|---|---|---|---|
| **V2-T1 (P0)** | KV membership publish on the compute stream — PR #211: `logical_kv_store.h` (+2/−2: `activate()` gains `cudaStream_t stream = nullptr`, `commit_activation(std::move(reservation), stream)`), `program_impl.h` (+4/−2: both `bind_sequence_kv` call sites pass `device.stream`), `tests/test_kv_cache.cpp` (+156/−2: page-release fence test + opt-in `--dangerous` race test). PR base is **exactly `b88c0f6f`**, mergeable-state clean (verified 2026-09-09). Baseline evidence: `logical_kv_store.h:895-896` (activate signature), `:902` (`commit_activation` with no stream), `:954` (default `cudaStream_t stream = nullptr`); `program_impl.h:10756`/`:10759` (the two call sites) | Neroued/ninfer PR #211 (the whole PR — no path scoping needed) | build + free-GPU ctest (new page-release fence test; the `--dangerous` race test is opt-in on a sacrificial GPU) + agentic soak (C=4, multi-turn, ≥100k prompt with catalog re-activation) + #210 no longer reproduces. Unfixed residuals from #210's own audit: `release_page()` frees without a stream fence; staged-tail COW for non-64-aligned prefixes (the 114k prefix ≡ 41 mod 64 exercises it) |
| **V2-T2** | **pick 17 `adf494c2` (block host sync — the 100 %-CPU decode fix) only.** Re-derived 2026-09-09: picks 15 (`6a1b62c5`) & 16 (`27417ca2`) are **SUPERSEDED by the base** (warmup already `DeadlinePolicy::UnboundedStartup` + fail-fast `main.cpp`; residual = cosmetic strings) — §6.2. Adopted = `src/core/device.{cu,h}` (2-stream adaptation; CUDA 13.1.2 3-arg `cudaInitDevice`). **SHIPPED 2026-09-09** (2nd attempt): branch `v2/t2-agentic` @ `0faef6d4`, tag `v2t2-0faef6d4`, image `408c7df8` = `:quasar`. First ship rolled back on the battery's cold decode-fresh gate (15/16); user accepted the ~5 % decode trade-off for the host-core savings; a supervised retag+restart window deployed it. Verified live: host ~0.9 % CPU during decode (the 100 %-core busy-wait is gone); frees up to 4 cores at C=4. | `gpillon/gpillon/coding` | ctest (failure-set diff); battery 16/16; idle-host-CPU check during decode (the 100 %-CPU regression) |
| **V2-T3 — SHIPPED (2026-09-09, gate override)** | gpillon PATH-REMAP pick `5f014910` (tool-call XML leak: the base's `ToolCallOutputDecoder::finish()` restructured onto the restructured decoder — the tool-call protocol region is no longer replayed as visible content; the fallback test updated to the no-leak behavior). `093c1fdd` (sibling-overlap telemetry) deferred — pure measurement + needs the dropped `concurrent_executor.h` re-targeted. **MEASURED (branch `v2/t3-path-remap` @ `458376c4`, image `12b87f4d`):** built clean + ctest clean (incl. the updated `test_tool_call_parser`) + 15/16 battery (every functional check passed: vision x3, replay 4/4, think-smoke, xhigh, quality, soak 5/5, 4xx-watch). DECODE-FRESH passed (163.9 / 159.2 / 171.4 tps); DECODE-8K red (131.6–144.9 tps) **only from a power-constrained GPU state** — the live V2-T2 baseline image itself measures ~137–160 tps on the same 8k probe (GPU at 544/575 W, 95 % power limit), so V2-T3 is **decode-neutral**. **Verdict: SHIPPED 2026-09-09** — the user overrode the red DECODE-8K gate (a power-state artifact, not a V2-T3 regression). Deployed via a supervised retag-only window (image `12b87f4d` = `:quasar`, now live); the first window run rolled back on a script self-check bug (unexpanded `~` in a double-quoted template path tripped the ERR-trap rollback; a missing `set -e` then clobbered the marker with a false DEPLOYED) — the script was fixed (tilde-safe `$HOME` path + `die(){rollback; exit 1}` in the ERR trap + hard model-id/template gates) and retried clean (~23s, no rollback). Verified live: serving image `12b87f4d`, `/v1/models` → `qwen3.8-27b`, template sha256 `180e7015…` byte-identical, quadlet untouched, decode smoke OK. The 100 %-CPU busy-wait fix (V2-T2 pick 17) is preserved — V2-T3 is a strict descendant of V2-T2. | `gpillon/gpillon/coding` | tool-call parser unit tests; telemetry = no behavior change |
| **V2-T4 — REJECTED (measured 2026-09-09)** | **MEASURED:** the 2 solid on-path picks — `49400365` (T18 GDN chunked-prefill) + `d3278b79` (t42wave q/k rmsnorm decode-attention fuse) — applied (branch `v2/t4-readopt` @ `aa27864c`, image `68cec959`); ctest PASSED, but the battery's DECODE-FRESH + DECODE-8K gates FAILED (140.6 / 142.8 tps vs the 162.0 / 165.3 V2-T2 baseline; a ~9 % decode regression beyond V2-T2's accepted ~5 % trade-off) → ship **ROLLED BACK** (lane back on V2-T2 `408c7df8`, verified). **Verdict: REJECTED** — the old T18/t42wave perf picks regress on the evolved (restructured-attention) baseline. Remaining picks, not re-adopted: `52fabe3e`+`bb535075` (content-superseded by upstream `ee9d5192`), `15729d9b`+`2e99db7b` (FP16-PV — off-path, the lane is nvfp4), `fa12e8ef` (MTP stem-norm — off-path, the lane runs dflash2), `ed505ebc` (sigmoid-gate — a kernel hand-port into the restructured `softmax_attention/dense` kernel; deferred to a V2-T4b if ever re-attempted), `67bf4b78`/`0f84adaf` (softmax-fold / context-cost — re-derivation candidates, not measured) | our branches | per-pick `git merge-tree` clean; ctest; decode A/B vs the live tps baseline (the 2 measured picks FAILED this gate) |
| **V2-T5 — SHIPPED (2026-09-09, gate override, A/B-verified)** | md single-commit decode wave: `38f52b34` (argmax winner-init kernel), `61250e89` (#194 nvfp4 SwiGLU fast), `ed150906` (#201 w8 rowsplit cache policy), `1dfeed7e` (draft-head-narrow branch, 9 commits). Excluded with reasons: `0d9841d2` (bpe-flat-merge-table) — **ABSORBED** (upstreamed as `b158afe2`; content-identical diffstat `tokenizer.{cpp,h}` +82/−15); `0deee4d8` (l2-pin linear-attention state) — GDN dead path for 27B (T44 triage); `01591621` (fp8-a8-tma-staging) — is PR #167's own head, already covered by V2-T4's `52fabe3e` | `md/*` branches | **SHIPPED 2026-09-09** (branch `v2/t5-decode` @ `324a8de3`, image `f8b76e5a4dc2` = `:quasar`, retag-only deploy): the build+ctest+battery pipeline auto-rolled back on a **DECODE-8K-only red** (132.5 vs stored baseline 153.8; 15/16 green incl. replay 4/4 + vision + soak 5/5), and a same-window A/B decode-differential with concurrent `nvidia-smi` power/clock sampling proved it a **stale-baseline power artifact, not a regression** — in the same throttled window, live (V2-T3) 8k = 112.1 tps vs candidate 8k = 135.5 tps (**+21 %**), fresh 156.4 vs 128.9, with the candidate at an equal-or-better power state (491 W / 2902 MHz vs 526 W / 2865-2872 MHz). A supervised retag-only window deployed it; verified live (`/v1/models` 200 qwen3.8-27b, quadlet byte-identical, chat_template sha256 `180e7015…`). Gate §10.5 met in-window |
| **V2-T6** | cometkim: `c17ccc30` (`feat/qwen3.8-nvfp4qat`, 11 commits — QUASAR-QAT NVFP4 profile; our `f7727926` already carries the `Qwen38Nvfp4*` family → A/B against ours, adopt only if upstream merges their form or the A/B wins) + `6c3fdbf4` (`feat/kernel-perf`, 14 commits — PDL decode chain; the +77 %/+56 % claims must be re-derived on our base first: 3 force-pushes since the 09-08 audit) | `cometkim/*` | profile A/B on the live artifact; PDL claim re-measured on 5090 before any window |
| **V2-T7** | gzenz host-KV safety-net re-derivation (old T31/T34) from `62b857c1` (117 ahead, 2026-09-09): show the B2 (entitlement) / B3 (frontier) blockers are fixed in the current line, then re-derive the pick set | `gzenz/fix/checkpoint-host-demotion` | only if host-KV re-enable is approved; ctest + host-KV soak |
| **V2-T8 (large) — ADOPTED 2026-09-09 (code complete; runtime gate pending)** | gpillon RAM-KV agentic cluster hand-port: picks 1–11 + 13 (§6.2) **plus the never-committed WIP engine layer** (`de386ad6`/`f144f052`, §6.6). **LANDED on `v2/t8-agentic`:** `71f8e0b1` (paged-KV bridge onto the baseline `HostKVArena` + `DeviceKVPagePool`) + `a5ba1ee5` (WIP host-RAM prefix reuse: `kv_ram_cache.{h,cpp}` fully re-targeted onto the baseline page-pool API + state-image bridge kRamVersion 4; `plan_ram_reuse()` terminal admission-time match, MTP-aware; `capture_retained_lane()`/`restore_ram_entry()` in the lane lifecycle; `--kv-ram-mib` serve option + `RuntimeStats`). **Full `ninfer` + `ninfer-serve` Release build GREEN** (CUDA 13.1/120a, RC=0). Design (§6.5): integrate, not replace — eviction via `ResourceManager` pressure planning; shared `--host-kv-mib` pool with the `HostKVExtentStore` demotion mirror. Includes the T34 guard (pick 6) + `ac60331d` guard test | `gpillon/gpillon/coding` | **DONE: full build green.** NEXT (supervised lane window): host-RAM hit-path A/B (TTFT/decode/cache-hit at C=4, ≥100k agentic prompt) + battery 16/16 + greedy parity |
| **V2-T9 (conditional)** | adaptive MTP widths trio `c2708ec8` → `9d86436c` → `9bef0f73` + MTP items `505d1af7`, `1f155fed` (= our `fa12e8ef`) — **only if the lane returns to `--spec mtp`**; today it runs upstream dflash2 | `gpillon/gpillon/coding`, `md/*` | acceptance + decode A/B under MTP |
| **Watches** | #208 (stability, tracks V2-T1), #213/#201 (groupwise-W8 for 27B text projections), #197 `ignore_eos`, #183 `--chat-template FILE` (T38), #152/#163/#162 (serve ergonomics; T32 cluster #176–#181/#184), #61 (per-image vision budget), #173 REJECT (sub-floor KV), #107/#97/#72 (T13, 503-bad), #174/#165 (T45, MTP-conditional), #169/#168/#172 (T41 agentic inputs), #185 (idle unload), `dylan/experimental` (agentic slices only), mirko KVaRN REJECT (sub-floor) | — | — |

## 8. Fork survey (post-fetch 2026-09-09, vs `quasar-master`)

| Remote/branch | Tip | Ahead | Verdict |
|---|---|---|---|
| `cometkim/cometkim/dev` | `58b6b5b1` (09-09) | 13 | WATCH (force-rebased; was 32 ahead on 09-08 — the delta to upstream shrank) |
| `cometkim/feat/kernel-perf` | `6c3fdbf4` (09-09) | 14 | **V2-T6** (PDL decode chain; re-verify claims) |
| `cometkim/feat/qwen3.8-nvfp4qat` | `c17ccc30` (09-09) | 11 | **V2-T6** (QAT profile; compare against our `f7727926`) |
| `cometkim/feat/qwen3.8-nvfp4full` | — (09-09) | 11 | NOTHING-NEW (the nvfp4full profile family is in the baseline via `f7727926`) |
| `cometkim/feat/mtp7` | `e3bf30db` (09-09) | 2 | WATCH (draft-tokens-7 option; the lane already runs `--draft-tokens 7` on upstream dflash2) |
| `cometkim/feat/1m-context` | (09-09) | 12 | WATCH (1M ctx; the standing decision is T15's 400k) |
| `cometkim/feat/dflash2` | (09-01) | 9 | REJECT — DFLASH2 engine work; excluded by policy (§6.3) |
| `cometkim/feat/hyperquant` | (09-09) | 6 | REJECT — off-lane KV route (T34 note: skip) |
| `cometkim/feat/webui` / `feat/windows-port` | (09-09) | 6 / 5 | NOTHING-NEW (lane is Linux; webui = T38-adjacent watch) |
| `gzenz/fix/checkpoint-host-demotion` | `62b857c1` (09-09) | 117 | **V2-T7** (host-KV safety net re-derivation) |
| `gzenz/master` / `fix/materialization-root-fallback` / `fix/checkpoint-stateimage` | (09-06/09-09) | 50 / 49 / 45 | WATCH |
| `dylan/experimental` | `42c9c7d4` (09-07) | 126 | WATCH — GDN/qwen4/dflash bulk = dead path for dense 27B; agentic slices `42c9c7d4`/`a39c5c25` feed V2-T8 |
| `mirko/feat/kvarn-production` | `cf63feac` (09-07) | 54 | REJECT — sub-floor KV k4v2-g128, no E2E quality evidence (§10.4) |
| `mirko/master` / `fix/dflash-prefill-state-slot` / `integration/upstream-master-through-ce7dee50` | `8debca38` etc. (09-07) | 13 / 13 / 12 | NOTHING-NEW / WATCH (= upstream through `ce7dee50` + dflash2 ops; dead path for us) |
| `md/perf/argmax-winner-init-kernel` | `38f52b34` (09-08) | 1 | **V2-T5** |
| `md/perf/draft-head-narrow` | `1dfeed7e` (09-05) | 9 | **V2-T5** |
| `md/perf/` (13 other single-commit branches, incl. `moe-*`) | (09-06/09-09) | 1–20 each | triaged: dead path (MoE/GDN) or already covered — see V2-T5 exclusions |
| `eason` | (08-22) | 2–5 | dormant |
| `gpillon/gpillon/coding` | `a00648cb` (09-03, no movement 09-09) | 103 | **V2-T2/T3/T8/T9** (§6) |

## 9. Upstream PR/issue watch (live 2026-09-09, `api.github.com`)

**Open, lane-relevant:**
#211 (P0 → **V2-T1 — SHIPPED 2026-09-09** as `ba21e676`; PR still open upstream, head force-moved
dce5f773 → `0687a66c`, same 3-file stream-threading fix) · #213 (groupwise-W8 for 27B text
projections + W8 leading-dim fix; makes #201 relevant if 27b-W8 lands) · #202 (L2 linear-attention
pin — GDN dead path for us) · #201 (w8 rowsplit activation-cache policy) · #200/#199 (MoE — dead
path) · #197 (`ignore_eos`) · #195 (weights-format preset fallback, prefill-cost 3.1× → 1.15×) ·
#194 (nvfp4 SwiGLU expf slow-path drop) · #183 (`--chat-template FILE`, T38) · #173 (rk2v4-e8
compressed KV — REJECT, sub-floor) · **#167 (fp8-A8 GEMM TMA staging — our T23 pick `52fabe3e`) +
#160 (nvfp4 tile-contiguous scales — our T23 pick `bb535075`), still open → V2-T4** ·
#163/#162 (serve timings / metadata) · #152 (automatic shared-prefix write; T32 cluster) ·
#148 (Responses API) · #107/#97/#84 (T13 family — 503-bad evidence stands) · #61 (per-image vision
budget — we run `--vision`) · #59/#54 (known; #54 = the exception-naming fix our t42wave already
carries as a local pick).

**Merged since the old audits (in `b88c0f6f`):** #206, #205, #204, #203, #198, #193, #191, #161, #159.

**Open issues:** #210 (P0 → V2-T1, fix SHIPPED 2026-09-09 — `ba21e676`; watch upstream merge) · #208 (intermittent `cudaErrorIllegalAddress`, 5090,
NVFP4+MTP — T49 stability watch, tracks V2-T1) · #192, #207, #212 · #174 (full-vocab Q4G64 MTP
head → T45) · #165 (YaRN — hosts #174's implementation fork) · #169/#168/#172 (agentic serve →
T41 inputs) · #176–#181/#184 (T32 cluster) · #185 (idle unload — our sentinel partially covers) ·
#164/#166.

## 10. Standing policy (carried forward)

1. **Pipeline** (from `ADOPTION.md` "Pipeline"): one PR = one tagged commit; `git merge-tree`
   pre-check before any pick; one PR at a time; per-batch rebuild → boot ledger → `/v1/models` →
   thinking smoke test; ship only via the supervised pipeline (`ninfer-ship.sh` + battery).
2. **Lane safety:** a supervisor agent on a **non-lane model** must be running before any
   lane-touching job; windows are self-healing (BAK-first, ERR-trap restore, always-restart,
   watchdog dead-man, DONE flag); quiet window mandatory; battery 16/16; OMP-mounted files stay
   byte-identical (verify `chat_template.jinja` sha256
   `180e7015759b2b6b57574d6c2ca5c2d19eb2b05a4aaffa80866f71eb1a1fad1a` after every quadlet change).
3. **Model chain:** keep the QUASAR artifact chain at all costs (smaller + more precise than the
   official weights); grafts go on a COPY, never onto the base artifact.
4. **KV precision floor:** no sub-floor KV (KVaRN k4v2-g128, rk2v4-e8, ninfer-fusion) without E2E
   quality evidence (T30/T39 precedent).
5. **Speculative stack:** dflash2 comes from **upstream only** — no `dflash*` path from gpillon
   (§6.3); MTP work stays MTP-conditional (V2-T9, T45); adoption gate = acceptance AND decode
   improve AND battery green (T7 rule, generalized by T35's revert).
6. **Tier numbering:** strictly incremental — the legacy T1–T49 are immutable history; the V2
   tiers use the `V2-Tn` scheme (no collisions); any future legacy-style numbering resumes after
   the highest existing tier (T49).

---

## Appendix A — measurement harnesses used for this document (all read-only)

```bash
cd /home/gevil/Work/Personal/ninfer

# absorption (patch-id is unusable here — §4)
git rev-list --left-right --count <branch>...quasar-master
git cherry quasar-master <branch> | grep -c '^+'

# ancestor checks (absorption evidence)
git merge-base --is-ancestor <sha> quasar-master

# per-pick standalone feasibility vs the baseline (fresh temp index per pick; repo untouched)
for sha in <picks>; do
  T=$(mktemp)
  GIT_INDEX_FILE=$T git read-tree quasar-master
  git diff-tree -p --no-color ${sha}^ ${sha} > /tmp/p.diff
  err=$(GIT_INDEX_FILE=$T git apply --cached --3way /tmp/p.diff 2>&1); rc=$?
  # existing-file conflicts (unmerged entries):
  GIT_INDEX_FILE=$T git diff --name-only --diff-filter=U
  # missing substrate paths:
  echo "$err" | grep -oE 'error: [^:]+: does not exist in index'
  rm -f "$T"
done
# rc=0 → CLEAN; rc=1 with unmerged entries → CONFLICT;
# rc=1 with only "does not exist in index" → missing-substrate failure.
# NEVER reuse one index across picks: earlier picks' staging contaminates later results.

# record provenance
for b in $(git branch --format='%(refname:short)'); do
  printf '%s %s %s\n' "$b" "$(git rev-parse --quiet --verify $b:ADOPTION.md 2>/dev/null || echo MISSING)" \
    "$(git show $b:ADOPTION.md 2>/dev/null | wc -l)"
done

# fork survey
git rev-list --count quasar-master..<remote-ref>; git log -1 --format='%h %cs' <remote-ref>

# upstream PR/issue state
curl -s 'https://api.github.com/repos/Neroued/ninfer/pulls?state=open&per_page=40' \
  | jq -r '.[] | "\(.number) \(.title) \(.head.sha)"'
```

**Non-commit hex tokens (SHA-check convention):** the SHA-resolution check (§Appendix A)
passes for every *commit* SHA in the doc. Five tokens are intentionally non-commit:
four Docker image ids (`12b87f4d`, `408c7df8`, `68cec959`, `f8b76e5a4dc2` — the deployed
lane images for V2-T3/T2/T4/T5) and the 8-char sha256 prefix `180e7015` of
`chat_template.jinja`. PR #211's head `0687a66c` resolves via the fetched `refs/pr/211`.

*End of ADOPTION-V2. Previous authorities: `t42wave-quasar:ADOPTION.md` (operational, until this
file) and `master:ADOPTION.md` (historical round ledger).*
