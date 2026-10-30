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
/// [--in-use 90] [--alias OLD=NEW]... [--images 127.0.0.1:8086] [--idle-minutes 10]`
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
    // the front door's maintenance notice, beside the registry
    sw.maintenance = Some(nextsycl_serve::maintenance::file_beside(&models::registry(&Config::load())));
    // the GPU run lock (NS_GPU_LOCK): no chat model swapped in while a run holds it
    sw.gpu_lock = nextsycl_serve::gpulock::path(Config::load().get("NS_GPU_LOCK"));
    // the same idle time as the front door's (NS_IDLE_MINUTES), unless given
    let idle = opt(args, "--idle-minutes").map(str::to_string).or_else(|| Config::load().get("NS_IDLE_MINUTES"));
    sw.idle = idle.map(|v| v.parse::<f64>().map_err(|_| "--idle-minutes N")).transpose()?.filter(|m| *m > 0.0)
        .map(|m| Duration::from_secs_f64(m * 60.0));
    let addr = format!("{}:{}", opt(args, "--host").unwrap_or("0.0.0.0"), opt(args, "--port").unwrap_or("8001"));
    std::sync::Arc::new(sw).run(&addr)
}

/// `nextsycl serve [--host 0.0.0.0] [--port 8000] [--serve-host 0.0.0.0] [--idle-minutes 10]`: the front door - the GPUs
/// and the models, enabling and loading them, one API for all (nextsycl_serve::home). Ports from the settings:
/// NS_PORT, NS_SWITCH_PORT, NS_IMAGE_PORT(2), NS_AUDIO_PORT(2), NS_VIDEO_PORT, NS_CHAT_UI_PORT; the API's token NS_API_TOKEN
/// (else one made and kept beside the registry); NS_IDLE_MINUTES.
/// The registry and the settings for the front door (nextsycl-models', which the glue may not depend on)
pub(crate) struct Store(pub(crate) Config);

impl nextsycl_serve::home::registry::Store for Store {
    fn models(&self) -> Result<Vec<serde_json::Value>, String> {
        models::all(&self.0)
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut serde_json::Value)) -> Result<serde_json::Value, String> {
        models::update(&self.0, id, f)
    }
    fn registry_file(&self) -> PathBuf {
        models::registry(&self.0)
    }
    fn get(&self, name: &str) -> Option<String> {
        self.0.get(name)
    }
    fn in_env(&self, name: &str) -> bool {
        self.0.from_env(name)
    }
    fn file(&self) -> PathBuf {
        self.0.file()
    }
    fn guide(&self, entry: &serde_json::Value) -> Option<String> {
        nextsycl_models::catalog::Catalog::load(&self.0).ok()?.guide(entry)
    }
    fn cards(&self) -> Vec<serde_json::Value> {
        nextsycl_models::catalog::Catalog::load(&self.0).map(|c| c.models).unwrap_or_default()
    }
    fn card_files(&self, id: &str) -> Vec<(String, u64)> {
        nextsycl_models::catalog::Catalog::load(&self.0).ok().and_then(|c| c.model(id).map(|m| c.file_list(m))).unwrap_or_default()
    }
}

pub fn home(cfg: &Config, args: &[String]) -> Result<(), String> {
    use nextsycl_serve::home::{Home, Options, Ports};
    let port = |k: &str, d: u16| cfg.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let wfe = cfg.dist.join("wfe");
    if !nextsycl_serve::wfe::exists(&wfe, "home/index.html") {
        return Err(format!("{}: the page is not built yet (./build.sh wfe)", wfe.display()));
    }
    let idle = opt(args, "--idle-minutes").map(str::to_string).or_else(|| cfg.get("NS_IDLE_MINUTES")).map_or(Ok(10), |v| v.parse::<u64>().map_err(|_| "--idle-minutes N"))?;
    let host = opt(args, "--host").unwrap_or("0.0.0.0");
    let port_s = opt(args, "--port").map(str::to_string).or_else(|| cfg.get("NS_SERVE_PORT")).unwrap_or("8000".into());
    let o = Options {
        wfe,
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        ports: Ports {
            llm: port("NS_PORT", 8085), switch: port("NS_SWITCH_PORT", 8001), image: port("NS_IMAGE_PORT", 8086), audio: port("NS_AUDIO_PORT", 8087), model3d: port("NS_MODEL3D_PORT", 8088),
            image2: port("NS_IMAGE_PORT2", 8096), audio2: port("NS_AUDIO_PORT2", 8097), model3d2: port("NS_MODEL3D_PORT2", 8098),
            video: port("NS_VIDEO_PORT", 8090), chat_ui: port("NS_CHAT_UI_PORT", 8080),
        },
        serve_host: opt(args, "--serve-host").unwrap_or("0.0.0.0").to_string(),
        door: format!("{}:{port_s}", if host == "0.0.0.0" || host == "::" { "127.0.0.1" } else { host }),
        cfg: Box::new(Store(Config::load())),
        idle_minutes: idle,
    };
    Home::new(o)?.run(&format!("{host}:{port_s}"))
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

/// `nextsycl maintenance [on|off|status] [--message TEXT] [--eta MINUTES] [--unload] [--port 8000]`: the front door's
/// maintenance mode (nextsycl_serve::maintenance) - callers told, new work refused, the queues waiting; on answers the
/// bypass key (header X-Maintenance-Bypass) for testing through the front door meanwhile
pub fn maintenance(cfg: &Config, args: &[String]) -> Result<(), String> {
    let port = opt(args, "--port").map(str::to_string).or_else(|| cfg.get("NS_SERVE_PORT")).unwrap_or("8000".into());
    let mut body = serde_json::json!({"do": "maintenance"});
    match args.first().map(String::as_str) {
        Some("on") => {
            body["on"] = true.into();
            body["message"] = opt(args, "--message").unwrap_or("").into();
            if let Some(m) = opt(args, "--eta") {
                body["eta_s"] = (m.parse::<f64>().map_err(|_| "--eta MINUTES")? * 60.0).into();
            }
            body["unload"] = args.iter().any(|a| a == "--unload").into();
        }
        Some("off") => body["on"] = false.into(),
        None | Some("status") => {}
        Some(x) => return Err(format!("maintenance {x}: on, off or status")),
    }
    let t = Target::Tcp(format!("127.0.0.1:{port}"));
    let v = nextsycl_serve::http::call_for(&t, "POST", "/api/action", Some(&body), 60).map_err(|e| format!("the front door (:{port}): {e}"))?;
    if let Some(e) = v["error"].as_str() {
        return Err(e.to_string());
    }
    match v["maintenance"].as_object() {
        Some(n) => {
            println!("maintenance ON: {}", n.get("notice").and_then(|x| x.as_str()).unwrap_or(""));
            if let Some(k) = v["bypass"].as_str() {
                println!("bypass: X-Maintenance-Bypass: {k}");
            }
            for r in v["draining"].as_array().into_iter().flatten() {
                println!("still running: {}", r.as_str().unwrap_or(""));
            }
            for m in v["unloading"].as_array().into_iter().flatten() {
                println!("unloading {}", m.as_str().unwrap_or(""));
            }
        }
        None => println!("maintenance off"),
    }
    Ok(())
}
