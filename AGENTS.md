# AGENTS.md

Working agreement for agents operating in this environment. Part 1 is universal.
Part 2 applies to GPU operator/kernel work.

## Part 1 — Basic Rules

### Scope and ownership

- Touch only what the task requires. Never modify system code, vendored/third-party
  sources, other people's files, branches, or environments — ask first and wait for
  approval, even if the change looks trivially safe.
- Read a file before editing it; inspect the target before deleting or overwriting.
- Actions that leave this machine (push, PR, publish, send) need explicit approval
  each time. Approval for one action is not standing approval for the next.

### Git

- No agent attribution anywhere: no `Co-Authored-By` for the agent, no
  "Generated with …" footer in commit messages or PR bodies. The user is the author.
- Never commit on `main`/`master` — branch first. Commit or push only when asked.
- Conventional commit subjects; one logical change per commit.

### Running tasks: timeouts

Always set an explicit timeout, and set it tight. A generous timeout turns a hang
into a silent stall; a timeout that fires is a diagnosis. Never raise a timeout to
make a failure go away — find out why it got slow.

| Job                                        | Timeout                     |
| ------------------------------------------ | --------------------------- |
| Lint / format / CPU unit test              | 1–2 min                     |
| Incremental build                          | 5 min (full build: background) |
| Single-GPU correctness or one test file    | 3–5 min                     |
| Single-GPU benchmark or profiler sweep     | 5–10 min                    |
| Multi-GPU, single node                     | 10 min, prefer background   |
| Multi-node                                 | background + poll           |

- The foreground limit is 10 min; anything longer runs in the background and gets
  polled, not wrapped in a bigger timeout.
- Distributed jobs need their own watchdog — a hung rank never exits on its own. Set
  the collective/init timeout inside the job well below the outer limit so it dies
  with a rank id instead of stalling silently.

### Shared GPUs

- Check occupancy before every launch (running compute processes and used memory). If
  someone else's job is on the device, wait for it to go idle or move to a genuinely
  free one. Never squeeze in alongside it, and never kill or preempt it.
- If the wait is long, say so and ask — do not run anyway to save time.
- A measurement that shared the device is void. Check occupancy before *and* after a
  run (and periodically during a long sweep); if any foreign process appeared,
  discard the numbers, do not log or report them, and re-measure on an idle device.
- Baseline and candidate must be measured on the same device under the same idle
  conditions. A contended run compared against an idle baseline is not a comparison.
- Device-unavailable or unexpected-OOM errors on a shared box are usually contention,
  not a regression — check occupancy before debugging the code. Under
  exclusive-process mode, a context that is still releasing also counts as busy, so
  back-to-back runs of your own can collide too.

### Design and code style

- Diffs should read like the surrounding code: same naming, idiom, and comment
  density. Prefer the smallest change that solves the whole problem.
- Maintainability over cleverness. No speculative abstraction, no options nobody
  asked for, no indirection layer with a single caller, no premature generality.
- Attack the essence, not the symptom. Removing a special case beats adding one.
- Leave nothing behind: temp scripts, debug prints, and dead branches are either
  cleaned up or explicitly reported.

### Plan-driven execution

- Plan before coding. State goal, constraints, files to be touched, and how the
  result will be verified. Get agreement on any non-trivial design before writing it.
- Ask when a wrong assumption would waste real work — feasibility, interface
  ownership, acceptance criteria, performance targets. Do everything that does not
  depend on the answer first, then ask at the point it actually blocks.
- Report faithfully. Failing tests are shown with their output; skipped steps are
  named; "done" is said only when verified.

### No shortcut solutions

- Never fit the tested case. Clamping, silent fallbacks, hardcoded shapes,
  retry-until-green, and test-specific branches are not fixes.
- Root-cause first: reproduce minimally, identify the mechanism, fix it at the level
  where the mechanism lives. If only a workaround is possible now, label it as one
  and write down the root cause next to it.
- Generic correctness beats tested correctness. A guard that fails loudly for an
  unsupported configuration is far better than a silent wrong result.

### Multi-agent workflow

- Use parallel subagents when subtasks are genuinely independent (per-file, per-op,
  per-config sweeps, independent review lenses). Don't parallelize work that needs
  shared context or serial decisions.
- The main agent supervises: it owns the plan, hands each agent a scoped brief
  (goal, constraints, files it may touch, acceptance test, what to return), reviews
  every returned diff, and integrates. Never merge a diff you have not read.
- Give agents that write to the same tree isolated workspaces (separate worktrees).
- Verification is a separate agent from implementation when it matters — an author
  reviewing their own work finds less.

## Part 2 — Operator Development and Optimization

### 2.1 Adding a new op

- Map the layers before writing anything: kernel → instantiation/registration →
  runtime binding → host-language wrapper → test. The closest existing sibling op is
  the spec; mirror it end to end.
- Naming: when the build globs sources, every source filename must be globally
  unique — same-name files in different directories collide silently. Test file and
  test function names must be unique too, or the test runner's import system clashes.
- Model variants belong in runtime dispatch (a config struct or enum), not in copied
  files. Shape/dtype differences are data, not code paths.
- Declare read-only vs in-place arguments correctly in the binding schema. A wrong
  in-place annotation produces aliasing bugs that no test will name.
- For cross-device data movement: symmetric buffers, an explicit
  flag/payload protocol, and ping-pong slots so a write cannot pass an unread read.
- Verification order: format/lint changed files → build → correctness → only then
  performance.
- Golden references must model the hardware's arithmetic (accumulation order,
  intermediate precision, quantization steps). A naive high-precision golden yields a
  meaningless tolerance.
- A test that skips because the op is missing is not a passing test. Read the skip list.

### 2.2 Isolated dev worktree

**Why:** a build that registers an editable/in-place install binds the package name
to whichever tree built last. Several worktrees sharing one environment silently
break each other.

- One worktree = one branch = one cloned environment named after the worktree
  directory. Activate that environment before any build or test.
- Place worktrees as siblings of the main checkout; keep `dir name == branch slug ==
  env name` so the mapping is discoverable from another session.
- Clone the environment from the user's current one (ask which), offline, rather than
  resolving dependencies again.
- Copy vendored/submodule directories from the main checkout instead of
  re-initializing them: faster, no network, guaranteed same commit. Use a copy that
  replaces the target directory rather than nesting inside it.
- After a bulk copy, watch for mtime-based staging: a copy stamps every file with the
  same timestamp, so build steps that skip when source and destination mtimes match
  will silently no-op — re-touch overlay/patch files afterwards.
- Copy the tool permission settings into the new worktree so it does not re-prompt.
- Detect a worktree at session start: if `git rev-parse --git-dir` differs from
  `--git-common-dir`, you are in one. Then activate the matching environment and
  never push to main.
- Resume across sessions via `git worktree list`, the environment list, and the
  branch's recent log.
- Cleanup after merge: confirm a clean tree and a closed PR → remove the worktree
  (force is needed when submodules were copied in) → delete the cloned environment.

### 2.3 Kernel optimization loop

**Fix the constraints with the user first:** resource ceilings (block/CTA count,
shared-memory or page budget, register target), pipeline configuration shared with
neighbouring ops, which interfaces are frozen, whether the weight format may change,
where to focus, and the acceptance threshold (default: ≥5% median improvement on the
target configs).

**Phase 1 — Understand.** Read the kernel, the template/primitive it builds on, the
host wrapper and any weight converter, the registration/executor path, and the test.
Write down: instruction mix per unit of work, memory access pattern (layout, stride,
bank-conflict risk), pipeline structure (stages, page size, producer/consumer warp
split), tunable parameters, and resulting occupancy. Summarize to the user.

**Phase 2 — Baseline.** Confirm the device is idle first (see *Shared GPUs*) — every
number below is void otherwise. Then: a benchmark script (warmup + N iterations,
report min/P10/median/P90), a correctness check against golden over every supported
config, a profiler sweep across the full config matrix, and a resource-usage dump
(registers, spills, shared memory). Record constraints, baseline config, and baseline
numbers in an optimization log.

**Phase 3 — Locate the bottleneck before changing code.**

| Bottleneck            | Symptom                                    | Where to act                          |
| --------------------- | ------------------------------------------ | ------------------------------------- |
| Compute-bound         | Instruction count dominates, high pressure | Fewer ops, better ILP                 |
| Transfer/prefetch     | Compute idles waiting for the next page    | Less volume per page, deeper overlap  |
| Shared-mem bandwidth  | Bank conflicts on operand or reduction     | Pad strides, change access pattern    |
| Sync overhead         | Many barrier pairs per kernel              | Fewer pages, merge stages             |
| Register spills       | Non-zero stack in the resource dump        | Less unrolling, simpler inner loop    |

Estimate transfer time vs compute time per page, but treat the estimate as a
hypothesis only: the compiler interleaves loads and math, so phases are coupled, not
separable. Measurement decides.

**Phase 4 — Iterate, one change at a time.** build → correctness on all configs →
profile on the same device as the baseline → accept if it clears the threshold, else
revert immediately. Record the result either way; the rejected attempts are the most
valuable part of the log.

**Catalog, in increasing order of effort:**

1. *Compute pipeline* — contiguous instead of interleaved work-to-warp mapping;
   hoist per-group scalars out of the inner loop; use the MMA accumulator as its own
   C-input instead of a separate add; unroll the inner loop; pad any shared-memory
   stride that is an exact multiple of the bank count.
2. *Memory access* — wider vectorized loads; smaller dtypes for scales/metadata;
   layout or swizzle changes (the converter must change with them).
3. *Structural* — tile size, elements per page, warp count, reduction shape. Each
   moves occupancy and block count; re-check the ceilings before proposing one.
4. *Infrastructure* — async-copy → TMA, kernel fusion, wider (warpgroup) MMA. Last
   resort, high effort.

**Compound optimizations.** Some changes only pay off together and regress alone —
e.g. contiguous mapping makes all of a warp's tiles share one scale group, which
enables hoisting the scale out of the loop, which frees the registers that make
unrolling profitable. Apply such chains in dependency order, express the
prerequisite as a compile-time condition, and fall back safely when it does not hold.

**Transferable lessons:**

- Optimize the bottleneck you measured. Fixing bank conflicts on a transfer-bound
  kernel buys nothing.
- Apparent "barrier overhead" is usually consumers spinning on data that has not
  arrived. Reduce transfer volume, not barrier count.
- The compiler may already do what your knob does — diff generated assembly or
  resource usage before believing a tuning parameter changed anything.
- Bank conflicts hide in innocent strides: check `stride % num_banks == 0` for every
  shared array read across warps.
- Register pressure is the hidden cost of unrolling; check spills, not only time.
- Measure on the same device every time; compare `min` for stability, report median.
- Guard conditions must be derived for the general case, not the tested one. A wrong
  guard on a shared template gives silent wrong results for future instantiations,
  not a crash — have template-level changes independently reviewed before merging.
- Ops at the degenerate end of the pipeline (single page/stage) gain nothing from
  pipeline work.

**Session close:** every attempt logged (accepted and rejected); clean build; all
correctness configs passing; a final measurement on an idle device; independent
review of shared-template changes with any latent bug fixed; regression suite green.
