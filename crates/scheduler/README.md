# backplane-scheduler

The backplane's scheduler as one self-contained crate: **placement, room planning, pins and rsvp holds, model-card
concurrency profiles, ETAs and the wait decisions**. It is pure and deterministic. Everything the box does - read the
cards, start and stop a model's server, sleep - is the *host's*, through the traits in `src/host.rs`; the crate decides.

`VERSION` is in `src/lib.rs` (and `Cargo.toml`); `CHANGELOG.md` says what each version changed. nextcuda depends on this
crate (no fork); nextsycl copies the directory and implements the host.

- Dependencies: **std and `serde_json`** only. No other nextcuda crate, no process spawning, no HTTP, no GPU API.
- Size: ~2,600 lines with its tests (33 unit tests, 22 scenarios on a fake host and a fake clock).

## What is in it

| module | what |
| --- | --- |
| `profile` | a model card's `concurrency` / `coexistence`: schema, defaults, validation (`Profile::from_card`), `may_share`, and the bench's decisions (`recommend`, `scaling_of`, `verdict`) |
| `rsvp` | `rsvp_s` keep-alives: `granted` (clamp to the cap), `Hold`/`extend`/`left`/`kept_from`, the `NS_MAX_RSVP_S` cap, `max_willing_to_wait_s` (`wait_limit`), the `wait_too_long` error tag, the per-call thread-local `scope` |
| `room` | `Snapshot` (cards, processes, servers, models, pins/holds) and the pure planning: `room_plan`, `plan_room`, `pick_card`, `plan_with_blockers` (exclusive and `shared_with`), `no_room` message |
| `place` | `place` (one try at a server for a model), `ensure` (the retry loop), `free_plan`, `wait_room`, the second-copy logic, `asked_mb`/`ask_mb` |
| `shape` | request shapes per method (`Shapes`: a registry of rules; `standard()` has image, video, speech, song, transcribe, chat, 3D) |
| `history` | the rolling history, last 100 per key (shape and model), `Storage` trait with `FileStorage` (JSON lines, writer thread, rotation) and `MemStorage`; `History::eta` |
| `eta` | work ahead from the jobs (`ahead_s`), `check_wait`, `wait_expired`, `would_accept`, `late`/`withdrawn` (queued jobs), `running_left` |
| `host` | the traits the host implements |
| `fake` | `FakeHost`: cards with memory, servers, a fake clock; used by `tests/scenarios.rs` and available to a port's own tests |

## What to copy

The whole directory `crates/scheduler/` (src, tests, Cargo.toml, README.md, CHANGELOG.md). In the other workspace:

1. add `"crates/scheduler"` to the workspace `members`, and `backplane-scheduler = { path = "crates/scheduler" }` to
   `[workspace.dependencies]`; depend on it from `glue/serve` (and from `glue/models` if the model cards are loaded there);
2. do **not** keep a `[workspace]` table in the crate's `Cargo.toml` (the copy-out check adds one, see below).

What stays in the host's glue (it is not in the crate, on purpose): the rpcgen method table and the request-field
declarations, the HTTP wiring, the process spawning, the card sampler, the registry file, the job table, the UI. Their
shapes are in "JSON shapes" below.

## The traits to implement (`host::Host`, `room::Idleness`)

On nextcuda they are implemented by `Sched` in `glue/serve/src/home/life.rs` (`impl Host for Sched<'_>`) over `Home`. On
nextsycl the same lineage has the same `Home` functions; map them like this (names are nextcuda's; nextsycl's are the
same or close - check by grep):

| trait method | what it needs | nextcuda's function behind it |
| --- | --- | --- |
| `now`, `sleep` | the clock; a test fake advances on `sleep` | `home::now()`, `std::thread::sleep` |
| `snapshot` | cards (`total/used`, processes with kind, model, port, MiB), servers by kind (`slot`, `port`, `model`, `loading`, `busy`, `inflight`), models (id, kind, enabled, cards, `Profile`), pinned, kept (pinned + holds), hold seconds left | `Home::snapshot` (from `named_cards`, `servers`, `models`, `pinned_ids`, `kept_ids`) |
| `server(kind, port)` | one server read fresh (HTTP `/api/info`, `/api/progress`) | `Home::server_at` |
| `start(kind, id, slot, port, gpu)` | start the model on the card, **block until it answers**; a refusal for memory must contain `needs about X GiB ...; Y GiB are free` | `Home::start_at` (runs `<kind> start ID --gpu N --port P --slot S`, then `wait_up`) |
| `stop(kind, slot, port, label)` | stop the server in the slot | `Home::exec([kind, stop, --slot, N])` |
| `free_video`, `free_chat` | let the video daemon's idle worker on a card go; unload the idle chat model | `/engine/unload` on the studio; `Home::unload` |
| `video_idle`, `chat_idle` | the studio has nothing to make; the chat model, when it may be preempted (not answering, not kept) | `Home::video_idle`, `Home::chat_idle` |
| `try_lock_kind` | the per-kind lock, `None` if taken (the guard is dropped to release it) | `Home::kind_lock[kind].try_lock()` |
| `wait_loaded(port, id)` | wait until a loading server answers as `id` | `Home::wait_up` |
| `note(job, stage, why)`, `cancelled(job)` | a job's stage / waiting-for words; the cancel flag | `Home::note`, `Home::job_cancelled` |
| `learn_ask(id, err, gpu, want)` | remember what a server refused for; true if more than planned | `Home::learn_ask` |

Also the host's own, outside the traits: the **history storage** (`history::FileStorage::new(dir)`, or your own
`Storage`), the **jobs** it hands to `eta::ahead_s` as `JobView`s, and the **registry** it turns into `ModelInfo`
(`Home::model_info`, `profile_of`).

## Config keys and environment

| name | where | meaning |
| --- | --- | --- |
| `NS_MAX_RSVP_S` | settings/env (host reads it, passes the string to `rsvp::cap_of`) | the longest keep-alive granted, seconds; unset or 0: 86400 |
| `NS_IDLE_MINUTES` | settings/env (host) | a server's idle time; the host passes `--idle-exit` unless the model is pinned (`idle_exit_args` in `life.rs`) |
| `NS_VIDEO_IDLE` | settings/env (video daemon) | seconds a video worker idles before unloading; the host asks the daemon to keep workers while a video model is pinned/held (`POST /engine/keep`) |
| `NS_BENCH_LOCKED`, `~/.nextcuda-gpu.lock` | the bench tool | the flock every GPU run takes |

Constants in `room.rs`: `MARGIN_MB` (1536: kept free beyond a model's memory), `ROOM_WAIT_S` (90), `JOB_ROOM_WAIT_S`
(1800). In `rsvp.rs`: `DEFAULT_MAX_RSVP_S`. In `profile.rs`: `WORTH_GAIN` (0.25), `COLLAPSE_SHARE` (0.5).

## JSON shapes

Requests (added to every method that runs a model; the host declares them, the crate reads them with `rsvp::scope(body)`):

```json
{"rsvp_s": 300, "max_willing_to_wait_s": 120}
```

Answers:

- an object result gets `"rsvp_granted_s": 300` (clamped to the cap) when `rsvp_s` was sent;
- `no_room` (HTTP 200 envelope, `error.status` 503, `retryable: true`): the message names the holder - `pinned: X (unpin to
  free the room)`, `held by rsvp_s: X (N s more)`, `exclusive: X (runs alone on its card)`;
- `wait_too_long` (`error.status` 503, `retryable: true`): `error: {code: "wait_too_long", eta, max_willing_to_wait_s, waited_s?}`.
  The crate returns it as a `Fail` whose message is `rsvp::wait_tag(fields, words)`; the host's error builder calls
  `rsvp::wait_untag` and merges the fields into the body;
- `system.eta {method, params, model?, max_willing_to_wait_s?}` ->
  `{method, eta: {wait_s, run_s, total_s, samples, basis: "shape"|"model"|"none", spread, shape}, max_willing_to_wait_s, would_accept}`;
- a model row (`system.models`): `concurrency {max_copies_per_card, recommended_copies, scaling, measured, status}`,
  `coexistence {mode, shared_with, exclusive_reason, peak_extra_gib}`, `copies_now`, `recommended_copies`,
  `max_copies_per_card`, `pinned`, `held_s`; `system.busy.engines` is one `{id, kind, copies_now, recommended_copies,
  max_copies_per_card, mode, status}` per enabled model.

## Model-card fields

```yaml
concurrency:
  max_copies_per_card: 2          # >= 1, default 1
  recommended_copies: 1           # 1..max, default 1: extra copies start only when requests wait
  scaling: {"1": 1.0, "2": 1.18}  # total throughput of n copies relative to one (omit: unmeasured)
  measured: {date: 2026-10-31, card: RTX 5070, method: bench, notes: ""}
coexistence:
  mode: shared                    # shared | exclusive
  shared_with: ["kind:audio", "model:whisper-large-v3-turbo"]   # empty: anything that fits
  exclusive_reason: ""            # required when exclusive
  peak_extra_gib: 1.5
```

No block = *unmeasured*: one copy, shared with anything that fits. `Profile::from_card` rejects bad values with a
message naming the field; call it when the cards load.

## Migration order (port to another box)

1. Copy the directory; `cargo test -p backplane-scheduler` (the scenario suite must pass untouched: it needs no host).
2. Implement `Host` + `Idleness` for the box's `Home` (a struct holding `&Home`), mapping each method as in the table.
   Build `Home::snapshot()` first and test it against a real `named_cards` output.
3. Replace the box's room planning / `place` / `free` / `wait_room` / `no_room` with calls into `place::place`,
   `place::free_plan`, `place::wait_room`, `room::room_plan`, `room::no_room`; keep the kind lock and the lease in the host.
4. Move the pins and holds: `rsvp::extend`/`kept_from`; `Lease::drop` calls `hold_after`; registry `pinned` flag.
5. Add the history (`FileStorage` beside the registry) and record each finished request (`History::record_work`); use
   `eta::*` for `system.eta`, job views and the wait decisions.
6. Declare `rsvp_s` and `max_willing_to_wait_s` on the methods that run a model (rpcgen), and `system.pin`, `system.unpin`,
   `system.eta`; add `wait_too_long` to the error codes.
7. Model cards: load with `Profile::from_card`; expose the profile on the model rows.
8. Run the checklist.

## After copying: the checklist

- [ ] `cargo test -p backplane-scheduler` passes untouched (33 unit tests, 22 scenarios).
- [ ] The box's own tests still pass (the placement tests around `place`/`ensure`; nextcuda's are `home::fixture_tests`).
- [ ] A cold request starts its model; a second request for it does not start another.
- [ ] A pinned model survives a request for another model that needs its room (answer `no_room`, naming the pin); after
      unpin it is evicted by the next such request.
- [ ] An idle unpinned model is stopped for room; a busy one is waited for (watch the job's `waiting_for`).
- [ ] `rsvp_s: 300` on a request: `rsvp_granted_s` in the answer; the model is not evicted for 300 s after the request ends.
- [ ] `max_willing_to_wait_s: 1` against a busy box with history: `wait_too_long` at once, nothing started.
- [ ] An unmeasured card shows `status: "unmeasured"`; a card with `recommended_copies: 2` starts a second copy only while
      the first has work.
- [ ] An exclusive card's model is alone on its card; a request for another model gets `no_room` naming it.
- [ ] `history/*.jsonl` appears beside the registry after a few requests, and `system.eta` answers from it.

## Checking it stands alone

The crate builds and tests outside the workspace. Copy it, add an empty `[workspace]` table (so Cargo does not look for a
parent), and test:

```sh
cp -r crates/scheduler /tmp/sched-copy
printf '\n[workspace]\n' >> /tmp/sched-copy/Cargo.toml
cd /tmp/sched-copy && cargo test --release     # 33 unit + 22 scenario tests
```

(done for v0.1.0 in the debian-13-dev distrobox, with `serde_json` from the local registry cache).
