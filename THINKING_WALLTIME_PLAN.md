# Wall-time-to-accurate-answer plan (qwen3.8-27b NVFP4 lane, OMP harness)

Supersedes the ranking in the "Thinking-tuning research & plan — reduce overthinking" section of
`ADOPTION.md` (2026-09-05). That doc optimised *thinking tokens*. The operator's actual objective is
**wall time to an accurate answer (T2A)**. Under that objective the old ranking is wrong: it attacks
the smallest of four terms. Docs-only; commit separately from any lane work.

Constraints unchanged and enforced: thinking ON, `reasoning_effort=xhigh` held, no temperature
change, no hard reasoning-token caps, prompt-layer preferred, no accuracy loss.

---

## 1. The objective, decomposed

```
T2A  =  queue  +  prefill·(1 − cache_hit)  +  thinking_tokens / decode_rate  +  P(zero-yield) · retry
        ─────     ────────────────────────     ──────────────────────────      ───────────────────────
         W0              W1 / W2                      W3 / W6                        W4
```

Measured on the live lane (`ninfer-nvfp4`, `mirko_quasar.ninfer`, 2026-09-06, all four terms):

| Term | Measurement | Source |
|---|---|---|
| Average real OMP turn | **34.5 s generation, of which 30.2 s is TTFT (87.6%)**; 718 output tokens; decode ≈170 t/s | `~/.omp/agent/agent.db:model_perf`, `ninfer/qwen3.8-27b-quasar`, n=244 turns |
| Real turn, cache **miss** | prompt 140,001 · **cache 0.0%** · prefill 3.19k tok/s → **TTFT 2m14.8s** for 175 output tokens | server log `req#424` |
| Same turn class, cache **hit** | prompt 140,205 · **cache 100.0%** · TTFT 37.0 s of which **queue 36.7 s** → true prefill ≈0.3 s | `req#427` |
| Queue | one effective slot in practice (`running 1 … waiting 2`) despite `--max-concurrency 4`; observed waits 36.7 s / 45.2 s / 50.5 s / 1m30.2s | throughput lines, `req#421/424/426` |
| Thinking, easy task | GSM8K at xhigh: **41–161 reasoning tokens**, whole turn 0.3–1.2 s | own probe, n=12 |
| Thinking, hard open-ended task | **16.8k–18.2k reasoning tokens, 84–152 s wall**, answers non-empty | own probe (issue #216 reproducer), n=3 at xhigh |
| Prefill rate decay | 13.2k tok/s @8k prompt → 3.2k tok/s @140k; within one 140k prefill it decays 5.94k → 1.84k tok/s | throughput lines |
| MTP acceptance | **40.7% on a 5.9k-token thinking turn** vs 87.1% on a short turn; decode 148–235 t/s | `req#421/424/426/427` |

**Read of the data:** on this harness the enemy is not thinking length. It is (a) a 140k-token prompt
being re-prefilled when the prefix cache misses (~44 s), (b) queueing behind any other consumer of
the lane (~37–90 s), and only then (c) thinking tokens, which are near-zero on constrained tasks and
17k+ on open-ended ones. Prompt-layer thinking-shape work (old candidates C/C'/A/B/D) addresses
term 3 only — worth ~12% of an average turn, ~100% of a hard single-shot turn.

**Methodological warning, learned the hard way:** the probes behind this table *were* the queue that
stalled `req#424`/`req#427`. Benchmarking on the interactive lane invalidates the benchmark *and*
degrades the thing being measured. Also: identical prompts come back as `cache … (100.0%, response
replay)` — repeated identical requests are **not** independent samples. Every A/B request needs a
nonce.

---

## 2. Ranked candidates (by expected wall-time saving per turn)

### W0 — lane exclusivity for interactive work *(free, do today)*
Nothing on this lane is free-threaded: a single 140k prefill occupies the slot for ~45 s and
everything else waits. Bench runs, subagent fan-out, OWUI tabs and the interactive session are
competing for one slot and one KV pool.

- Rule: interactive session gets the lane alone; batch/bench work gets a second instance on another
  port with its own `--kv-capacity`, or waits.
- OMP-side: cap concurrent lane-backed subagents at 1 (fan-out to lane-backed workers is a wall-time
  anti-pattern here; fan out to cloud models instead, keep the lane serial).
- Expected: removes the 36–90 s queue term outright. Largest single win measured.

### W1 — prefix-cache hit engineering *(free, prompt-ordering only)*
`req#427` proves the machinery works: 140k prompt, 100% hit, prefill ≈0.3 s. `req#424` proves what a
miss costs: 44 s. So the whole game is *never breaking the prefix*.

- Audit the rendered prompt head for volatile text. OMP's own system prompt currently carries
  `Current time: …` and `_(refreshed 2026-09-06T09:55:51…)` markers on the memory blocks; Hindsight
  recalls are injected as `<memories>` near the head. Any byte that changes at the head invalidates
  everything after it.
- Required order: **immutable** (harness rules, skills, tool schemas) → **slow-changing** (project
  conventions) → **volatile** (clock, recalled memories, session ids) → history → current turn.
- Kill or coarsen clocks: minute-resolution timestamps guarantee a miss on every turn; use
  day-resolution, or move the clock into the *last* user message.
- Verify continuously: `usage.prompt_tokens_details.cached_tokens / prompt_tokens` ≥ 0.95 on every
  turn after the first. Below 0.90 → treat as a defect, not a tuning opportunity.
- Expected: 44 s → 0.3 s on long turns; strictly zero model-behaviour risk.

### W2 — context budget vs KV pool *(config + harness hygiene)*
`--kv-capacity 225280` with `--host-kv-mib 32768`. A single 140k-token session occupies 62% of the
GPU pool, so a second consumer of any size evicts it — the mechanism behind `req#424`'s 0% hit.

- Decide the pool policy explicitly: one 140k interactive session pinned, or two ~90k sessions.
- Confirm the host-KV restore path actually engages on eviction (host activity showed 16–20% during
  prefill, yet `req#424` still paid full prefill — either restore was not attempted for that entry or
  it was cold). This is the one engine-side question worth answering before any template work.
- Shrink the input: 201 messages / 140k tokens for a turn that emitted 175 tokens. Compaction,
  subagent offload of file-reading, lazy skill bodies (pointers already), and MCP tool pruning
  (the HA server alone contributes ~90 tool schemas; `req#427` rendered 11 tools — measure the token
  weight of each server and gate the unused ones per project).
- Expected: keeps W1's hit rate reachable; also lifts prefill rate (3.2k → 13.2k tok/s at 8k) and
  restores MTP acceptance on the miss path.

### W3 — thinking-shape guidance *(prompt layer only)*
The hard-task measurement (17–18k reasoning tokens, 84–152 s) shows the real overthinking signature:
unbounded enumeration on open-ended asks, exactly as in QwenLM/Qwen3.8 issue #216 (traces ending in
`Could there be a "support" for "Iron"? Not specific. Omit.` repeated for the whole candidate space).

Prompt-layer levers, in order:

1. **Stop-rule guidance in the system prompt** (no caps, xhigh held):
   "Enumerate the candidate set once. Do not re-walk a set you have already walked. When a check has
   passed, do not re-verify it. Consider an alternative only where a check actually failed."
   This targets re-walking, which is what the token count is made of, not depth of reasoning.
2. TRS static skill cards (old candidate C) stay on the list but move **after** the stop-rule probe:
   same mechanism class, much more work, and their published win is measured on token count, not T2A.

- Expected: on hard open-ended turns, the plausible target is 17k → 6–9k reasoning tokens
  (40–80 s → 30–45 s). On constrained turns it should be a no-op (nothing to cut: 41–161 tokens).

### E — one-clause xhigh instruction edit *(NEW candidate; template layer, same gating as A/B)*
The live template (`~/.local/share/ninfer/templates/chat_template.jinja`, 29,063 B, Sharp v22.4.0,
line 84) injects at xhigh:
> `Reasoning effort is set to xhigh. Please think carefully through the task, validate key
> assumptions, consider plausible alternatives, and prioritize correctness, consistency, and
> clarity in the final answer.`

`consider plausible alternatives` is an unbounded instruction with no stop condition — the plausible
driver of the enumeration tail. A single-clause variant ("…consider alternatives where a check has
failed…") keeps xhigh, keeps thinking ON and adds no cap, and is far smaller in blast radius than
swapping template families (A/B). **It is still a template-layer change, not a prompt change:** it
edits the bind-mounted jinja that renders tool-call wire format, reasoning parsing and history, so it
carries A/B's process, not W3's — full ninfer template regression battery + ship gate, canary one
instance, KV-hit-rate monitoring (a system-slot wording change invalidates every cached prefix until
re-cached; W1's ≥0.95 gate will show it), `check_applied.py` dual-source check, and no ad-hoc edits
to the live file while a session is being served. Proposal only in this document.

Related, measurement-validity item (no change required): `_default_reasoning_effort = 'medium'`
(line 17) means any client that does not pass `reasoning_effort` explicitly runs *medium* with **no**
instruction injected. OMP does pass xhigh (`req#427`: `thinking xhigh`); check OWUI's per-model
params before comparing anything.

### W4 — zero-yield turn insurance *(measure before adopting)*
A turn that thinks 20k tokens and returns empty content costs its full wall time *and* a retry —
the worst T2A event that exists. Issue #216 measures 19.4% empty-answer at xhigh on llama.cpp/vLLM
(0/72 at low/medium; does not reproduce on first-party), mitigated in their data by
`frequency_penalty 0.3` (0/12) or `repetition_penalty 1.05–1.10` (0/23, p=0.036; ≥1.20 fails in the
opposite direction). My 3 hard xhigh calls on the NVFP4 lane all answered, so **we do not yet know
our rate.** Sequence: measure P(empty) at xhigh over n≥30 nonce'd hard prompts, and only if it is
non-zero adopt `frequency_penalty 0.3` (OpenAI-standard field, survives stacks that drop
`repetition_penalty`). It is a sampler change, not a temperature change — flag it to the operator as
such rather than assuming it is inside the red lines.

### W6 — MTP/draft tuning *(accuracy-neutral by construction, pure wall time)*
Speculative decode is verified, so acceptance rate moves wall time with zero accuracy risk — the only
lever on this list that cannot cost accuracy. Observed acceptance collapses on exactly the traffic we
care about: **40.7% on a 5.9k-token thinking turn** vs 87.1% on a short one; decode swings 148–235
t/s. Sweep `--draft-tokens {2,3,4}` (and the `--lm-head-draft` toggle) *against thinking-heavy
traces*, not short prompts. A 40% → 70% acceptance recovery on a 17k-token thinking turn is worth
~20–30 s — comparable to the entire W3 prompt effort, with no behavioural risk.

### Parked, with reasons
- **Racing / best-of-2 for latency:** needs a free slot; on a one-slot lane it doubles the queue. Only
  viable after W0 gives the interactive session dedicated capacity.
- **CoD / TALE / hard budgets:** unchanged verdict — brevity caps collapse on hard problems, and CoD
  made 27B thinking *longer* in our own `q38-bench` test. Do not quote that repo's exact numbers here
  without reconciling it first: `docs/REASONING.md` runs the sweep at T=0.6 while `README.md`
  presents the daily driver at T=1.0/top_p 0.95/top_k 20, and the model there is the huihui GGUF, not
  our NVFP4 lane. The directional conclusion (a brevity cap did not shorten thinking) is what carries.
- **Template family swap (froggeric v22.5 / Sharper, old A/B):** unchanged risk class (render + tool
  wire format + reasoning parsing), 3.6-era numbers, needs the full regression battery. Candidate E
  gets most of the plausible win for one string, under the same gate.
- **Full TRS retrieval (old D):** only if hard-slice thinking still dominates T2A after W0–W3.

---

## 3. Measurement protocol (rewritten for T2A)

- **Primary metric:** wall time from request start to a *verified-correct* answer, including retries.
  Report **median and p90** per condition — p90 is where the empty-answer and enumeration tails live.
- **Secondary, per call, all available from the server log line:** `queue`, `TTFT`, `prefill tok/s`,
  `decode tok/s`, `cache N (X%)`, `mtp accepted a/b`, plus `usage.completion_tokens_details.reasoning_tokens`
  and `prompt_tokens_details.cached_tokens` from the response.
- **Slices:** constrained-answer tasks (GSM8K-style, verifiable) / hard open-ended tasks (the #216
  reproducer class) / real agentic coding turns. The first slice has no thinking to cut — do not let
  it dilute the average.
- **Hygiene, mandatory:** separate lane instance for benching; nonce in every prompt (response replay
  is real); pin `reasoning_effort` explicitly per request (template default is `medium`); n≥30 per
  condition; paired on tasks; one variable at a time.
- **Instrumentation first:** a log parser over `podman logs ninfer-nvfp4` producing per-request rows
  (queue/TTFT/cache%/mtp/decode) is ~30 lines and turns every future change into a measurable one.
  `agent.db:model_perf` already gives the aggregate ground truth (samples / output_tokens / gen_ms /
  ttft_ms per model key) — keep watching TTFT share of gen_ms as the headline number.

## 4. Red lines (operator, unchanged + two new)

No hard reasoning-token caps of any kind. No temperature change. No effort downgrade. Nothing adopted
at n<30, on easy-slice-only evidence, on medium-effort numbers, or on llama.cpp-specific findings.
**New:** never benchmark on the interactive lane; never treat repeated identical prompts as
independent samples.

## 5. Sequencing

| Phase | Content | Risk | Expected T2A effect |
|---|---|---|---|
| P0 | log parser + cache-hit/TTFT dashboard; confirm OWUI effort param | none | measurement only |
| P1 | W0 lane exclusivity + W1 prefix ordering (volatile text to tail) | none (no model change) | −40 to −90 s on long turns |
| P2 | W2 context budget / KV policy + host-KV restore question | config | protects P1; lifts prefill rate |
| P3 | W6 draft-token sweep on thinking-heavy traces | none (verified decode) | −10 to −30 s on hard turns |
| P4 | W4 empty-answer rate measurement, adopt mitigation only if non-zero | sampler change | removes retry tail |
| P5 | W3 stop-rule prompt A/B (prompt layer), then candidate E one-clause template A/B (full battery + canary) | prompt / template | −30 to −50% thinking tokens on hard slice |
| P6 | TRS cards / template family swap, only if hard-slice thinking still dominates | high | unproven at xhigh |

## 6. External sources added in this pass

Provenance rule for this section: only primary sources are cited for numbers — the GitHub issue body
(read in full, including its 2026-08-27 self-correction), the arXiv papers, the vendor docs, and the
config repo. Search surfaced several secondary blog/aggregator write-ups of the same effort/latency
claims (including one paywalled benchmark post); none of their figures are used here.

| Source (primary, read) | What it contributes |
|---|---|
| QwenLM/Qwen3.8 issue #216 (~1,000 calls, corrected 2026-08-27) | xhigh empty-answer 19.4% (0/72 at low/medium), not reproducible first-party; `frequency_penalty 0.3` / `repetition_penalty 1.05–1.10` mitigations with the upper-bound caveat; **xhigh reasoning tokens scale with input length (5.5× at 6k chars → 14.7× at 48k)** while low/medium stay flat at 700–1,900 — the mechanism linking context bloat to thinking time; template default is xhigh upstream and `medium` injects no instruction |
| Raju et al., *The Limits of Long-Context Reasoning in Automated Bug Fixing* (arXiv 2602.16069) | successful agentic trajectories stay under 20–30k tokens and longer accumulated context correlates with *lower* success; single-shot 64k resolve rates collapse (7% / 0%) — supports W2 (shrink input) on accuracy grounds, not just wall time |
| QwenCloud thinking docs | `preserve_thinking` semantics (lane runs `--preserve-thinking`): omitting `reasoning_content` in multi-turn tool flows degrades accuracy — do not "save prompt tokens" by dropping it; `thinking_budget` exists as a serve-side cap and stays banned here |
| soster/qwen38-thinking-levels (config repo + verification notes) | independent confirmation that Qwen3.8 effort is an **instruction-based** soft prompt injected by the template, per-request via `chat_template_kwargs`, no reload needed — the basis for candidate E being a one-string change |
| MCP tool-schema bloat (modelcontextprotocol SEP-1576, *title/abstract only — not opened*) | direction only: tool-schema overhead is a recognised first-class prompt-size term, supporting per-project MCP gating in W2. No figure from this line is used; measure our own servers' token weight instead |
| TRS (arXiv 2604.21764) | unchanged: only published accuracy+tokens Pareto, but token-framed and off-family; correctly demoted under a T2A objective |
