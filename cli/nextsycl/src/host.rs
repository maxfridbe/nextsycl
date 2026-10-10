//! The host services that sit beside the model servers: `nextsycl switch` (the chat front end's model switcher) and
//! `nextsycl gpustat` (the GPU telemetry file pages read).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use nextsycl_models::{config::Config, registry as models};
use nextsycl_serve::http::Target;
use nextsycl_serve::switch::{Model, Switch};

fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn secs(args: &[String], k: &str, d: f64) -> Result<Duration, String> {
    let v = opt(args, k).map(|s| s.parse::<f64>().map_err(|_| format!("{k} SECONDS"))).transpose()?.unwrap_or(d);
    Ok(Duration::from_secs_f64(v.max(0.0)))
}

/// `nextsycl switch [--host 0.0.0.0] [--port 8001] [--upstream 127.0.0.1:8085] [--studio URL] [--wait 420]
/// [--in-use 90] [--alias OLD=NEW]... [--images 127.0.0.1:8086]`
pub fn switch(args: &[String]) -> Result<(), String> {
    // the same settings, the closure's own (the registry is read on every request)
    let cfg = Config::load();
    let list: nextsycl_serve::switch::Models = Box::new(move || {
        let all = models::all(&cfg).unwrap_or_else(|e| {
            eprintln!("the registry: {e}");
            Vec::new()
        });
        // the chat models (entries of other kinds - image, video, audio, lora - are not chat models)
        all.iter().filter(|m| m["enabled"] != false && models::kind_of(m) == "llm").filter_map(|m| {
            let id = m["id"].as_str()?.to_string();
            Some(Model { title: m["title"].as_str().unwrap_or(&id).to_string(), tools: m["tools"] != false, tasks: m["tasks"] != false, id })
        }).collect()
    });
    let mut aliases = HashMap::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--alias" {
            let (o, n) = args.get(i + 1).and_then(|x| x.split_once('=')).ok_or("--alias OLD=NEW")?;
            aliases.insert(o.to_string(), n.to_string());
        }
    }
    let upstream = Target::Tcp(opt(args, "--upstream").unwrap_or("127.0.0.1:8085").to_string());
    let studio = opt(args, "--studio").unwrap_or("http://127.0.0.1:8090/rpc/llm.mode");
    let mut sw = Switch::new(list, aliases, upstream, studio, secs(args, "--wait", 420.0)?, secs(args, "--in-use", 90.0)?)?;
    sw.images = opt(args, "--images").map(|a| Target::Tcp(a.to_string()));
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("0.0.0.0"), opt(args, "--port").unwrap_or("8001"));
    std::sync::Arc::new(sw).run(&addr)
}

/// `nextsycl gpustat [--out /run/gpustat.json] [--interval 3] [--power-window 15] [--pci ADDR] [--vram-mb N] [--once]`
pub fn gpustat(args: &[String]) -> Result<(), String> {
    nextsycl_serve::gpustat::run(nextsycl_serve::gpustat::Options {
        out: PathBuf::from(opt(args, "--out").unwrap_or("/run/gpustat.json")),
        interval: secs(args, "--interval", 3.0)?,
        power_window: secs(args, "--power-window", 15.0)?,
        pci: opt(args, "--pci").map(str::to_string),
        vram_mb: opt(args, "--vram-mb").map(|v| v.parse().map_err(|_| "--vram-mb N")).transpose()?,
        once: args.iter().any(|a| a == "--once"),
    })
}
