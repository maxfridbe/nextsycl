//! `nextsycl drives` and `nextsycl models move`: the drives the models are on, and a model's files moved to another
//! one - nextsycl_serve::drives, the front door's own code. With the front door up the move is its job (its page shows
//! it, and it knows what is loaded); without it, it runs here.
//!
//! ```text
//!   nextsycl drives [--json]                     each drive: device, link, mount, size, free, the models, read speed
//!   nextsycl drives measure <drive>              its read speed (4 GiB of model files past the page cache)
//!   nextsycl models move <id> --to <drive | mount> [--yes]    its files there, checked, the registry repointed
//!   nextsycl models move --cancel                the move under way stopped (what it copied removed)
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use nextsycl_models::{config::Config, registry as models};
use nextsycl_serve::drives::{self, Drive};
use nextsycl_serve::http::{self, Target};
use serde_json::{json, Value};

const GIB: f64 = (1u64 << 30) as f64;

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// The front door, when it answers (its port: NS_SERVE_PORT)
fn front_door(cfg: &Config) -> Option<Target> {
    let t = Target::Tcp(format!("127.0.0.1:{}", cfg.or("NS_SERVE_PORT", "8000")));
    http::call_for(&t, "GET", "/health", None, 3).ok().map(|_| t)
}

/// One of the front door's page actions (no token: its own)
fn action(t: &Target, body: &Value) -> Result<Value, String> {
    match http::call_status(t, "POST", "/api/action", Some(body), 60) {
        Ok((200, v)) => Ok(v),
        Ok((_, v)) => Err(v["error"].as_str().unwrap_or("refused").to_string()),
        Err(e) => Err(format!("the front door: {e}")),
    }
}

fn state(t: &Target) -> Result<Value, String> {
    http::call_for(t, "GET", "/api/state", None, 30).map(|v| v["drives"].clone())
}

/// serve.json beside the registry: the front door's state (the measured speeds under "drives")
fn state_file(cfg: &Config) -> PathBuf {
    models::registry(cfg).with_file_name("serve.json")
}

fn measured(cfg: &Config) -> serde_json::Map<String, Value> {
    std::fs::read(state_file(cfg)).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["drives"].as_object().cloned()).unwrap_or_default()
}

fn folders(cfg: &Config) -> Vec<PathBuf> {
    drives::folders(cfg.get("NS_MODELS"), cfg.get("NS_MODEL_DIRS"))
}

fn date(at: f64) -> String {
    // yyyy-mm-dd from unix seconds (civil from days)
    let z = (at as i64).div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{}-{m:02}-{d:02}", yoe + era * 400 + i64::from(m <= 2))
}

/// `nextsycl drives ...`
pub fn cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let all = models::all(cfg)?;
    let list = drives::drives(&all, &folders(cfg));
    match args.first().map(String::as_str) {
        Some("measure") => {
            let name = args.get(1).ok_or("nextsycl drives measure <drive>   (nextsycl drives lists them)")?;
            let d = drives::find(&list, name).ok_or_else(|| format!("no drive {name} (nextsycl drives)"))?.clone();
            if let Some(t) = front_door(cfg) {
                action(&t, &json!({"do": "measure", "drive": d.id}))?;
                eprintln!("measuring {} (the front door) ...", d.id);
                loop {
                    std::thread::sleep(Duration::from_secs(2));
                    let st = state(&t)?;
                    if st["measuring"].is_null() {
                        let m = st["drives"].as_array().into_iter().flatten().find(|x| x["id"] == d.id.as_str()).map(|x| x["measured"].clone()).unwrap_or(Value::Null);
                        println!("{}: {} MB/s", d.id, m["mb_s"]);
                        return Ok(());
                    }
                }
            }
            let ms = drives::mounts();
            let files = drives::largest_on(&all, &d, &ms);
            let (bytes, secs) = drives::measure(&files, drives::MEASURE_BYTES, &AtomicBool::new(false))?;
            let mb_s = (bytes as f64 / secs / 1e6).round();
            println!("{}: {mb_s} MB/s ({:.1} GiB in {secs:.2} s, past the page cache)", d.id, bytes as f64 / GIB);
            // kept where the front door keeps it (it is not running: it reads the file when it starts)
            let f = state_file(cfg);
            let mut st: Value = std::fs::read(&f).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
            st["drives"][d.id.as_str()] = json!({"mb_s": mb_s, "bytes": bytes, "seconds": (secs * 100.0).round() / 100.0, "at": drives::now().round(),
                                                 "file": files.first(), "device": d.model, "link": d.link.as_ref().map(|l| &l.0)});
            if st.get("token").is_some() {
                std::fs::write(&f, serde_json::to_string_pretty(&st).unwrap_or_default()).map_err(|e| format!("{}: {e}", f.display()))?;
            }
            Ok(())
        }
        None | Some("list") | Some("--json") => {
            let meas = measured(cfg);
            if args.iter().any(|a| a == "--json") {
                let v: Vec<Value> = list.iter().map(|d| {
                    let mut v = d.json();
                    v["measured"] = meas.get(&d.id).cloned().unwrap_or(Value::Null);
                    v
                }).collect();
                println!("{}", serde_json::to_string_pretty(&json!({"drives": v})).unwrap_or_default());
                return Ok(());
            }
            println!("{:<11} {:<26} {:<13} {:<18} {:>8} {:>8} {:>9}  READ", "DRIVE", "DEVICE", "LINK", "MOUNT", "SIZE", "FREE", "MODELS");
            for d in &list {
                let read = meas.get(&d.id).map_or("- (nextsycl drives measure)".to_string(), |m| {
                    format!("{:.1} GB/s ({})", m["mb_s"].as_f64().unwrap_or(0.0) / 1000.0, date(m["at"].as_f64().unwrap_or(0.0)))
                });
                println!("{:<11} {:<26} {:<13} {:<18} {:>7.0}G {:>7.0}G {:>8.0}G  {read}", d.id, d.model.chars().take(26).collect::<String>(),
                         d.link.as_ref().map_or("-", |l| l.0.as_str()), d.mount.display(), d.size as f64 / GIB, d.free as f64 / GIB, d.bytes as f64 / GIB);
                println!("{:<11} {} models: {}", "", d.models.len(), d.models.join(" "));
            }
            Ok(())
        }
        Some(other) => Err(format!("nextsycl drives {other}: list | measure <drive>")),
    }
}

/// Ctrl-C while a move runs here: it stops at the next chunk and removes what it copied
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_int(_: i32) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

fn progress(v: &Value) -> String {
    format!("{} {}/{} {:.1}% {} MB/s{}  {}", v["state"].as_str().unwrap_or(""), v["file_index"], v["files"], v["pct"].as_f64().unwrap_or(0.0), v["rate_mb_s"],
            v["eta_s"].as_f64().map_or(String::new(), |s| format!(" eta {}m{:02}s", (s as u64) / 60, (s as u64) % 60)),
            v["file"].as_str().and_then(|f| f.rsplit('/').next()).unwrap_or(""))
}

fn report(v: &Value) {
    println!("moved {} to {}: {} files, {:.1} GiB, {} s", v["model"].as_str().unwrap_or(""), v["to"].as_str().unwrap_or(""), v["files"],
             v["bytes"].as_f64().unwrap_or(0.0) / GIB, v["seconds"]);
    for (a, b) in v["renamed"].as_object().into_iter().flatten() {
        println!("  {a}\n    -> {}", b.as_str().unwrap_or(""));
    }
    for k in v["kept"].as_array().into_iter().flatten() {
        println!("  kept {} (named by {})", k["file"].as_str().unwrap_or(""), k["models"]);
    }
}

/// `nextsycl models move <id> --to <drive | mount> [--yes]` | `--cancel`
pub fn move_cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let door = front_door(cfg);
    if args.iter().any(|a| a == "--cancel") {
        if let Some(t) = &door {
            let v = action(t, &json!({"do": "move_cancel"}))?;
            println!("cancelling the move of {}", v["model"].as_str().unwrap_or("?"));
            return Ok(());
        }
        return match drives::moving(&models::registry(cfg)) {
            Some(v) => Err(format!("{} moves in process {} (Ctrl-C there stops it)", v["model"].as_str().unwrap_or("?"), v["pid"])),
            None => Err("no move is under way".into()),
        };
    }
    let id = args.iter().skip(1).find(|a| !a.starts_with("--") && Some(a.as_str()) != opt(args, "--to")).cloned()
        .ok_or("nextsycl models move <id> --to <drive | mount> [--yes]   (nextsycl drives lists them)")?;
    let to_name = opt(args, "--to").ok_or("--to <drive | mount> (nextsycl drives)")?;
    let all = models::all(cfg)?;
    let set = folders(cfg);
    let list = drives::drives(&all, &set);
    let to: Drive = drives::find(&list, to_name).cloned().ok_or_else(|| format!("no drive {to_name} (nextsycl drives: {})",
        list.iter().map(|d| format!("{} {}", d.id, d.mount.display())).collect::<Vec<_>>().join(", ")))?;
    let m = all.iter().find(|m| m["id"] == id.as_str()).ok_or_else(|| format!("no model {id} (nextsycl models list)"))?;
    let ms = drives::mounts();
    let plan = drives::plan(&all, &id, &to, &list, &drives::roots(&set, &list), &ms)?;
    // a model that reads its weights while it serves, to a slower drive: asked
    let meas = measured(cfg);
    let mb = |d: &Drive| meas.get(&d.id).and_then(|v| v["mb_s"].as_f64());
    let cards = nextsycl_models::catalog::Catalog::load(cfg).map(|c| c.models).unwrap_or_default();
    if drives::streams(m, &cards) {
        let from = plan.from.iter().filter_map(|f| list.iter().find(|d| d.id == *f)).filter_map(|d| drives::rate(d, mb(d))).fold(f64::INFINITY, f64::min);
        let here = drives::rate(&to, mb(&to)).unwrap_or(0.0);
        if here < from && !args.iter().any(|a| a == "--yes") {
            return Err(format!("{id} reads its weights from the disk while it serves: on {} ({here:.1} GB/s, from {from:.1}) it serves slower - --yes to move it anyway", to.id));
        }
    }
    println!("{id}: {} files, {:.1} GiB ({:.1} GiB to write) from {} to {} ({})", plan.steps.len(), plan.bytes as f64 / GIB, plan.need as f64 / GIB,
             plan.from.join(", "), to.id, to.folder.display());
    if let Some(t) = door {
        let v = action(&t, &json!({"do": "move", "model": id, "drive": to.id}))?;
        if let Some(w) = v["move"]["warning"].as_str() {
            eprintln!("note: {w}");
        }
        eprintln!("the front door moves it (its page shows it; nextsycl models move --cancel stops it)");
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let mv = state(&t)?["move"].clone();
            if mv["model"] != id.as_str() {
                return Err("the front door no longer shows the move".into());
            }
            if mv["running"] != true {
                eprintln!();
                return match mv["state"].as_str() {
                    Some("done") => {
                        report(&mv["result"]);
                        Ok(())
                    }
                    _ => Err(mv["error"].as_str().unwrap_or("failed").to_string()),
                };
            }
            eprint!("\r{:<100}", progress(&mv));
        }
    }
    // here: nothing else may have its files open (no front door to say what is loaded)
    let open = drives::open_by(&plan.steps.iter().map(|s| s.from.clone()).collect::<Vec<_>>());
    if !open.is_empty() {
        return Err(format!("{id}'s files are open in {}: stop what has the model loaded first", open.join(", ")));
    }
    let (size, free) = drives::space(&to.mount).unwrap_or((to.size, to.free));
    drives::room(plan.need, free, size).map_err(|e| format!("{}: {e}", to.id))?;
    let _lock = drives::lock(&models::registry(cfg), &id, &to.id)?;
    // SAFETY: SIGINT (2) to a handler that only sets an atomic.
    unsafe { crate::signal(2, on_int as *const () as usize) };
    let mv = Arc::new(drives::Move::new(plan, None));
    let job = mv.clone();
    let worker = std::thread::spawn(move || job.run(&crate::host::Store(Config::load())));
    while !worker.is_finished() {
        std::thread::sleep(Duration::from_millis(500));
        if INTERRUPTED.load(Ordering::Relaxed) {
            mv.cancel.store(true, Ordering::Relaxed);
        }
        eprint!("\r{:<100}", progress(&mv.view()));
    }
    eprintln!();
    let r = worker.join().map_err(|_| "the move panicked".to_string())??;
    report(&r);
    Ok(())
}

/// A model whose files are moving does not start (`nextsycl <kind> start <id>`; the front door waits for it)
pub fn not_moving(cfg: &Config, id: &str) -> Result<(), String> {
    match drives::moving_why(&models::registry(cfg), id) {
        Some(why) => Err(why),
        None => Ok(()),
    }
}
