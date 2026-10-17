//! A kind's server as a container (`image` and `audio`): `serve` in the foreground, `start` in the background (its
//! port on a label), `ps` (the model and the request running), `logs`, and `stop` - which waits for the requests in
//! progress, so an engine is never ended mid-kernel. `--slot N` (1..) names another of the kind's servers
//! (`nextsycl-image-1` ...): the same model or another, on another card and port, beside the first.

use std::time::{Duration, Instant};

use nextsycl_models::Config;
use nextsycl_serve::http::{call_for, Target};

use crate::container::Ce;

/// How long a request's progress may stand still before `stop` takes it for hung (seconds)
const STUCK_S: u64 = 120;

pub struct Served {
    /// the container's name (slot 0's; slot N's has `-N` after it)
    pub name: &'static str,
    /// the command's (`nextsycl <kind> ...`)
    pub kind: &'static str,
    /// the default port
    pub port: &'static str,
}

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// `--idle-exit SECONDS`: the server ends after that long without work (none: never)
pub fn idle_exit(args: &[String]) -> Result<Option<&'static nextsycl_serve::idle::Idle>, String> {
    match opt(args, "--idle-exit") {
        None => Ok(None),
        Some(v) => {
            let s: u64 = v.parse().map_err(|_| format!("--idle-exit {v}: seconds"))?;
            Ok((s > 0).then(|| &*Box::leak(Box::new(nextsycl_serve::idle::Idle::new(Some(std::time::Duration::from_secs(s)))))))
        }
    }
}

fn me() -> String {
    std::env::args().next().unwrap_or_else(|| "nextsycl".into())
}

/// The arguments without `--slot N` (the server inside has no use for it)
pub fn without_slot(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--slot" {
            it.next();
            continue;
        }
        out.push(a.clone());
    }
    out
}

impl Served {
    /// The container's name for `--slot N` in `args` (none or 0: the first's)
    pub fn name_for(&self, args: &[String]) -> Result<String, String> {
        match opt(args, "--slot") {
            None => Ok(self.name.into()),
            Some(v) => match v.parse::<u32>() {
                Ok(0) => Ok(self.name.into()),
                Ok(n) => Ok(format!("{}-{n}", self.name)),
                Err(_) => Err(format!("--slot {v}: a number")),
            },
        }
    }

    /// `podman run`'s head: removed when it ends in the foreground, kept (for `logs`) in the background
    pub fn run_args(&self, detach: bool, args: &[String]) -> Result<Vec<String>, String> {
        // --init: the server is not PID 1, so a stop's SIGTERM ends it (PID 1 ignores signals it does not handle)
        Ok(vec![
            "run".into(), if detach { "-d".into() } else { "--rm".into() }, "--init".into(), "--name".into(), self.name_for(args)?,
            "--network".into(), "host".into(), "--stop-timeout".into(), "120".into(),
            "--label".into(), format!("nextsycl.port={}", opt(args, "--port").unwrap_or(self.port)),
        ])
    }

    pub fn launch(&self, ce: &Ce, a: &[String], detach: bool) -> Result<(), String> {
        let mut c = ce.cmd();
        c.args(a);
        if detach {
            c.stdout(std::process::Stdio::null());
        }
        let st = c.status().map_err(|e| e.to_string())?;
        match (st.success(), detach) {
            (true, _) => Ok(()),
            (false, true) => Err(format!("{} run failed ({st})", ce.bin)),
            (false, false) => Err(format!("the {} server ended ({st})", self.kind)),
        }
    }

    /// `start`: `serve` in the background, ready when it answers /health
    pub fn start(&self, cfg: &Config, args: &[String], run: impl FnOnce(bool) -> Result<(), String>) -> Result<(), String> {
        let name = self.name_for(args)?;
        run(true)?;
        let port = opt(args, "--port").unwrap_or(self.port);
        let t = Target::Tcp(format!("127.0.0.1:{port}"));
        let ce = Ce::new(cfg)?;
        let t0 = Instant::now();
        loop {
            if call_for(&t, "GET", "/health", None, 4).is_ok() {
                println!("{name} is up on port {port} ({:.0} s): {} {} ps | logs | stop", t0.elapsed().as_secs_f64(), me(), self.kind);
                return Ok(());
            }
            if !ce.running(&name) {
                return Err(format!("{name} ended while loading: {} {} logs", me(), self.kind));
            }
            if t0.elapsed() > Duration::from_secs(600) {
                return Err(format!("{name} did not answer in 10 minutes ({} logs)", self.kind));
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    /// The running server's address (its container's label)
    pub fn addr(&self, cfg: &Config, args: &[String]) -> Result<Target, String> {
        let name = self.name_for(args)?;
        let ce = Ce::new(cfg)?;
        if !ce.running(&name) {
            return Err(format!("{name} is not running ({} start)", self.kind));
        }
        let o = ce.cmd().args(["container", "inspect", "-f", "{{index .Config.Labels \"nextsycl.port\"}}", &name]).output().map_err(|e| e.to_string())?;
        let port = String::from_utf8_lossy(&o.stdout).trim().to_string();
        Ok(Target::Tcp(format!("127.0.0.1:{}", if port.is_empty() || port.contains("no value") { self.port } else { &port })))
    }

    /// `ps`: the model, what is running and how many wait
    pub fn ps(&self, cfg: &Config, args: &[String]) -> Result<(), String> {
        let t = self.addr(cfg, args)?;
        let info = call_for(&t, "GET", "/api/info", None, 10)?;
        let p = call_for(&t, "GET", "/api/progress", None, 10)?;
        let loras: Vec<String> = info["loras"].as_array().into_iter().flatten().filter(|l| !l["loaded"].is_null())
            .map(|l| format!("{}:{}", l["id"].as_str().unwrap_or("?"), l["loaded"])).collect();
        println!("{} ({}) on {t}{}{}", info["model"].as_str().unwrap_or("?"), info["arch"].as_str().unwrap_or("loading"),
                 if loras.is_empty() { String::new() } else { format!(", LoRAs {}", loras.join(" ")) },
                 if info["loading"] == true { ", loading" } else { "" });
        println!("{}", serde_json::to_string_pretty(&p).unwrap_or_default());
        Ok(())
    }

    /// `stop`: the request in progress (and those waiting) finish first - unless its progress stands still for
    /// STUCK_S (a hung engine never finishes): then it is stopped as it is
    pub fn stop(&self, cfg: &Config, args: &[String]) -> Result<(), String> {
        let name = self.name_for(args)?;
        let ce = Ce::new(cfg)?;
        if !ce.running(&name) {
            ce.remove(&name);
            return Err(format!("{name} is not running"));
        }
        if let Ok(t) = self.addr(cfg, args) {
            let t0 = Instant::now();
            let mut said = false;
            let (mut last, mut moved) = (serde_json::Value::Null, Instant::now());
            while let Ok(p) = call_for(&t, "GET", "/api/progress", None, 10) {
                if p["busy"] != true && p["waiting"].as_u64().unwrap_or(0) == 0 {
                    break;
                }
                if p != last {
                    (last, moved) = (p.clone(), Instant::now());
                } else if moved.elapsed() > Duration::from_secs(STUCK_S) {
                    println!("no progress for {STUCK_S} s (stuck at {}): stopping it as it is", p["phase"].as_str().unwrap_or("?"));
                    break;
                }
                if !said {
                    println!("waiting for the requests in progress");
                    said = true;
                }
                if t0.elapsed() > Duration::from_secs(1800) {
                    return Err("still busy after 30 minutes; not stopped".into());
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
        ce.stop(&name, 120);
        ce.remove(&name);
        println!("{name} stopped");
        Ok(())
    }

    pub fn logs(&self, cfg: &Config, rest: &[String]) -> Result<(), String> {
        let ce = Ce::new(cfg)?;
        let mut a = vec!["logs".to_string()];
        a.extend(without_slot(rest));
        a.push(self.name_for(rest)?);
        ce.cmd().args(&a).status().map_err(|e| e.to_string()).map(|_| ())
    }
}
