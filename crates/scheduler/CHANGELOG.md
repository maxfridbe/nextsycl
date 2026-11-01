# Changelog

Bump `VERSION` in `src/lib.rs` and `version` in `Cargo.toml` with each entry.

## 0.1.0+nextsycl.1 - 2026-10-31

nextsycl's copy of 0.1.0 (nextcuda f6f57a5). To take a later nextcuda version, copy it over and re-apply these; `tests/scenarios.rs` is
unchanged, `tests/hooks.rs` is nextsycl's. Every hook on `Host` has a default that does what nextcuda does, so they could go upstream as they are.

- `Cargo.toml`: no `preserve_order` on serde_json (feature unification would reorder JSON across nextsycl's whole workspace).
- `ensure` is the only driver: it takes the kind's lock around each try (`Host::wait_lock_kind`), counts the request before the lock goes
  (`Host::placed`), so a host needs no wait loop of its own (nextcuda's `ensure_for` keeps one).
- `Host::barred` / `Outcome::Barred`: nothing may load (NS_GPU_LOCK): a hot model still answers, a request is refused, a job waits and its room
  wait starts over. `Host::not_ready`: files moving, waited for like room, 409 at the limit. `Host::give_up`: a caller's limit tracked by the
  host, checked after each try and inside `drain` (the crate's `drain` could outlast `max_willing_to_wait_s`).
- `Host::ram_plan` / `wait_ram`: a host-memory rule beside the VRAM one (first load and second copy); `room::idle_on` and `room::kept_words`
  split out for it.
- `Host::reserve` + `room::may_start`: a last check that also takes the room, closing the exclusive-while-loading race and two kinds planning
  for one card. `Host::free_mb`: the wait after holders went without a whole snapshot each second.
- `Host::first_listed_card` / `room::pick_first`: the first listed card that can hold the model, stopping idle work there before going to a later card (the user's choice for hardac1, B70 first).
- `History::whole_run`: a method whose records include its load adds no spin-up to its wait.

## 0.1.0 - 2026-10-31

First extraction from nextcuda's backplane (`glue/serve/src/home`): nextcuda depends on it, behaviour unchanged.

- `profile`: model-card `concurrency` / `coexistence` schema, defaults, validation; the bench's decisions.
- `rsvp`: `rsvp_s` keep-alives and holds, the server cap, `max_willing_to_wait_s`, the `wait_too_long` error tag.
- `room`, `place`: snapshots and pure room planning; `place` / `ensure` over the `Host` trait (idle eviction, busy-holder
  drain, second copies up to the recommended count, exclusive and `shared_with` residents, pins and holds honoured).
- `shape`, `history`, `eta`: request shapes per kind (a registry), the rolling history (last 100, storage behind a trait),
  ETAs, `would_accept`, withdrawal of late queued jobs.
- `fake`: a fake host on a fake clock; `tests/scenarios.rs`: 22 scenarios.
