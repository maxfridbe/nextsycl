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

/// `nextsycl serve [--host 0.0.0.0] [--port 8000] [--serve-host 0.0.0.0]`: the GPUs and the services, with links and
/// controls (nextsycl_serve::home). Ports from the settings: NS_PORT, NS_SWITCH_PORT, NS_IMAGE_PORT, NS_AUDIO_PORT,
/// NS_VIDEO_PORT, NS_CHAT_UI_PORT.
pub fn home(cfg: &Config, args: &[String]) -> Result<(), String> {
    use nextsycl_serve::home::{Home, Options, Ports};
    let port = |k: &str, d: u16| cfg.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let wfe = cfg.dist.join("wfe");
    if !wfe.join("home/index.html").exists() {
        return Err(format!("{}: the page is not built yet (./build.sh wfe)", wfe.display()));
    }
    let c2 = Config::load();
    let o = Options {
        wfe,
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        ports: Ports {
            llm: port("NS_PORT", 8085), switch: port("NS_SWITCH_PORT", 8001), image: port("NS_IMAGE_PORT", 8086), audio: port("NS_AUDIO_PORT", 8087),
            video: port("NS_VIDEO_PORT", 8090), chat_ui: port("NS_CHAT_UI_PORT", 8080),
        },
        serve_host: opt(args, "--serve-host").unwrap_or("0.0.0.0").to_string(),
        models: Box::new(move || nextsycl_serve::home::with_kinds(models::all(&c2).unwrap_or_default(), |m| models::kind_of(m).to_string())),
    };
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("0.0.0.0"), opt(args, "--port").or(cfg.get("NS_SERVE_PORT").as_deref()).unwrap_or("8000"));
    Home::new(o).run(&addr)
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
