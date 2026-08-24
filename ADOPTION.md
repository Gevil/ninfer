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

### Tier 2 — the xhigh track (planned)

- forward `chat_template_kwargs.reasoning_effort` into the Jinja template (the
  `reasoning_effort` channel currently 400s on the OpenAI serve path),
- upgrade the Sharp chat template to **v22.3.1** (by file swap — no C++ port needed),
- warmup fail-fast + clearer `--kv-capacity auto` bounds, log the exception that
  terminates the server.

### Tier 3 — cherry-picks from community forks (planned)

Agent-checkpoint fixes and exact-`cached_tokens` observability from contributor
branches (`dylanbrodiefafard`, `easonLiangWorldedtech`), evaluated per commit at
execution time.

## Lane build

The lane image is built **from this branch** (public repo clone → `podman build`),
then the lane is restarted and verified; build/verification logs live in the host's
`~/.local/share/ninfer/logs/`. The lane is the verifier of its own history: the model
serving developer sessions runs on it.

*Record maintained 2026-08-23 — Tier 1 merged + verified (PR #1 open); Tier 2/3 planned. Update this file at each tier boundary.*