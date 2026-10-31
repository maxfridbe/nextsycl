//! Two clips at once, the real daemon and the real studio with no GPU and no model: the workers are a shell script (the
//! program a daemon starts workers with is the path it was started by, so the daemon is started under the script's name). The
//! script holds a job until the test lets it go, and answers a cancel. A door and a chat switcher are threads.
//!
//! What only the cards can show is not here: the memory a real worker takes, and the kernels of two at once.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const WORKER: &str = r#"#!/bin/bash
ctrl="$(dirname "$0")"
case "$2" in
gpus) echo '[{"index":0,"name":"fake B70","mem_gib":32,"pci":""},{"index":1,"name":"fake B65","mem_gib":32,"pci":""}]'; exit 0;;
worker) ;;
*) exit 2;;
esac
gpu=$4; model=$6
echo "$gpu $model" >> "$ctrl/workers"
say() { printf '%s\n' "$1"; }
say '{"event":"engine","state":"loading"}'
say "{\"event\":\"ready\",\"info\":{\"gpu\":$gpu,\"model\":\"$model\"}}"
job=""; cancelled=""
while :; do
  if IFS= read -r -t 0.05 line; then
    case "$line" in
      *'"exit"'*) exit 0;;
      '{"cancel":'*) c=${line#*:}; cancelled=${c%\}};;
      '{"job":'*) j=${line#*:}; job=${j%%,*}
        say '{"event":"log","line":"prompt : x"}'
        say '{"event":"log","line":"encoded: 1 s in 1.0 s"}'
        say '{"event":"log","line":"tokens : 100 in the denoiser"}'
        say "{\"event\":\"progress\",\"job\":$job,\"done\":40,\"total\":100}";;
    esac
  else
    [ $? -gt 128 ] || exit 0
  fi
  if [ -n "$job" ]; then
    if [ "$cancelled" = "$job" ]; then say "{\"event\":\"result\",\"job\":$job,\"ok\":false,\"error\":\"cancelled\"}"; job=""
    elif [ -e "$ctrl/fail-$job" ]; then say "{\"event\":\"result\",\"job\":$job,\"ok\":false,\"error\":\"it broke\"}"; job=""
    elif [ -e "$ctrl/release-$job" ]; then say "{\"event\":\"result\",\"job\":$job,\"ok\":true,\"value\":{}}"; job=""
    fi
  fi
done
"#;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A small HTTP server: `answer(path, body)`
fn serve(answer: impl Fn(&str, &str) -> String + Send + 'static) -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for mut c in l.incoming().flatten() {
            let mut b = vec![0u8; 8192];
            let mut n = 0;
            let head_end = |x: &[u8]| x.windows(4).position(|w| w == b"\r\n\r\n");
            while head_end(&b[..n]).is_none() {
                let got = c.read(&mut b[n..]).unwrap_or(0);
                if got == 0 {
                    break;
                }
                n += got;
            }
            let text = |x: &[u8]| String::from_utf8_lossy(x).to_string();
            if let Some(h) = head_end(&b[..n]) {
                let want = text(&b[..h]).lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(0);
                while n < h + 4 + want {
                    let got = c.read(&mut b[n..]).unwrap_or(0);
                    if got == 0 {
                        break;
                    }
                    n += got;
                }
                let head = text(&b[..h]);
                let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                let a = answer(&path, &text(&b[h + 4..n]));
                let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{a}", a.len());
            }
        }
    });
    addr
}

fn post(addr: &str, path: &str, body: &Value) -> Value {
    let mut c = TcpStream::connect(addr).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let b = body.to_string();
    write!(c, "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()).unwrap();
    let mut s = String::new();
    c.read_to_string(&mut s).unwrap();
    serde_json::from_str(s.split("\r\n\r\n").nth(1).unwrap_or("null")).unwrap_or(Value::Null)
}

struct Stack {
    dir: PathBuf,
    studio: Option<Child>,
    daemon: Child,
    addr: String,
    env: Vec<(String, String)>,
    door_addr: String,
    /// what the door says
    door: Arc<Mutex<Value>>,
    /// the modes the chat switcher was told
    told: Arc<Mutex<Vec<String>>>,
}

impl Drop for Stack {
    fn drop(&mut self) {
        if let Some(s) = self.studio.as_mut() {
            let _ = s.kill();
            let _ = s.wait();
        }
        let _ = self.daemon.kill();
        let _ = self.daemon.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn start_studio(dir: &Path, addr: &str, door: &str, env: &[(String, String)]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_nextsycl"))
        .args(["video", "studio", "--listen", addr, "--engine-name", "INT8=fake"])
        .arg("--socket").arg(dir.join("d.sock")).arg("--ui").arg(dir.join("ui")).arg("--out").arg(dir.join("out")).arg("--dir").arg(dir.join("state"))
        .arg("--gpustat").arg(dir.join("gpustat.json")).arg("--door").arg(door)
        .env("NS_VIDEO_MEMINFO", dir.join("meminfo")).env("NS_WFE_FROM_DISK", "1").envs(env.iter().cloned())
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap()
}

fn start(tag: &str) -> Stack {
    let dir = std::env::temp_dir().join(format!("ns-two-clips-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for d in ["out", "state", "ui/video", "home"] {
        std::fs::create_dir_all(dir.join(d)).unwrap();
    }
    std::fs::write(dir.join("ui/video/index.html"), "<html>__CSS__</html>").unwrap();
    std::fs::write(dir.join("denoiser.gguf"), "x").unwrap();
    std::fs::write(dir.join("meminfo"), format!("MemAvailable: {} kB\n", 48 * 1024 * 1024)).unwrap();
    std::fs::write(dir.join("gpustat.json"), json!({"energy_j": 1000.0, "cards": [{"energy_j": 1000.0}, {"energy_j": 5000.0}]}).to_string()).unwrap();
    let worker = dir.join("worker.sh");
    std::fs::write(&worker, WORKER).unwrap();
    std::fs::set_permissions(&worker, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let registry = dir.join("models.json");
    let entry = |id: &str| json!({"id": id, "kind": "video", "arch": "h3", "file": dir.join("denoiser.gguf"), "files": {"denoiser": dir.join("denoiser.gguf")}});
    std::fs::write(&registry, json!({"models": [entry("fake"), entry("fake-ref2va")]}).to_string()).unwrap();
    let env: Vec<(String, String)> = vec![("HOME".into(), dir.join("home").display().to_string()), ("NS_REGISTRY".into(), registry.display().to_string())];

    let door = Arc::new(Mutex::new(json!({"blocked": [], "held": false})));
    let told = Arc::new(Mutex::new(Vec::new()));
    let (d2, t2) = (door.clone(), told.clone());
    let door_addr = serve(move |path, _| {
        let d = d2.lock().unwrap();
        if path == "/chat/held" { json!({"held": d["held"]}).to_string() } else { json!({"blocked": d["blocked"], "host_free_mb": d["host_free_mb"]}).to_string() }
    });
    let switcher = serve(move |_, body| {
        if let Some(m) = serde_json::from_str::<Value>(body).ok().and_then(|v| v["mode"].as_str().map(str::to_string)) {
            t2.lock().unwrap().push(m);
        }
        json!({"result": {"mode": "chat-a"}}).to_string()
    });
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nextsycl"));
    cmd.arg0(&worker).args(["video", "daemon", "--model", "fake", "--engine", "fake-ref2va", "--gpu", "0", "--gpu", "1", "--idle", "0"]).arg("--socket").arg(dir.join("d.sock"))
        .envs(env.iter().cloned()).stdout(Stdio::null()).stderr(Stdio::null());
    // as on the box: GPU 1 shared with chat, so a clip goes to GPU 0 first
    cmd.args(["--shared-gpu", "1", "--llm-switcher", &format!("http://{switcher}/rpc/llm.mode"), "--door", &door_addr]);
    let daemon = cmd.spawn().unwrap();
    let t0 = Instant::now();
    while !dir.join("d.sock").exists() {
        assert!(t0.elapsed() < Duration::from_secs(20), "the daemon did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
    let addr = format!("127.0.0.1:{}", free_port());
    let studio = Some(start_studio(&dir, &addr, &door_addr, &env));
    let t0 = Instant::now();
    while TcpStream::connect(&addr).is_err() {
        assert!(t0.elapsed() < Duration::from_secs(20), "the studio did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
    Stack { dir, studio, daemon, addr, env, door_addr, door, told }
}

impl Stack {
    fn rpc(&self, method: &str, body: Value) -> Value {
        post(&self.addr, &format!("/rpc/{method}"), &body)
    }
    fn gen(&self, label: &str, more: Value) -> Value {
        let mut b = json!({"prompt": format!("{label} prompt"), "label": label, "steps": 2, "seconds": 1});
        for (k, v) in more.as_object().into_iter().flatten() {
            b[k] = v.clone();
        }
        self.rpc("generate", b)
    }
    fn status(&self) -> Value {
        self.rpc("status", json!({}))["result"].clone()
    }
    fn clip(&self, label: &str) -> Value {
        self.status()["running_clips"].as_array().into_iter().flatten().find(|c| c["label"] == label).cloned().unwrap_or(Value::Null)
    }
    fn until(&self, what: &str, f: impl Fn(&Stack) -> bool) {
        let t0 = Instant::now();
        while !f(self) {
            assert!(t0.elapsed() < Duration::from_secs(30), "{what}: {}", self.status());
            std::thread::sleep(Duration::from_millis(150));
        }
    }
    /// A clip's engine job id
    fn job_of(&self, label: &str) -> String {
        self.clip(label)["job"]["id"].as_str().unwrap_or_else(|| panic!("{label} is not running: {}", self.status())).to_string()
    }
    fn release(&self, label: &str) {
        std::fs::write(self.dir.join(format!("release-{}", self.job_of(label))), "").unwrap();
    }
    fn fail(&self, label: &str) {
        std::fs::write(self.dir.join(format!("fail-{}", self.job_of(label))), "").unwrap();
    }
    fn running_on(&self, label: &str, gpu: u64) {
        self.until(&format!("{label} running on GPU {gpu}"), |s| s.clip(label)["gpu"] == gpu && s.clip(label)["stage"].as_str().is_some_and(|x| x.starts_with("sampling")));
    }
}

#[test]
fn two_clips_run_at_once_each_with_its_own_status_and_the_third_waits() {
    let s = start("two");
    assert_eq!(s.status()["slots_free"], 2);
    // the first takes the unshared card, the second the shared one
    let a = s.gen("A", json!({}));
    assert_eq!((a["ok"].clone(), a["result"]["queued"].clone(), a["result"]["reentered"].clone()), (json!(true), json!(false), json!(false)), "{a}");
    s.running_on("A", 0);
    assert!(s.told.lock().unwrap().is_empty(), "a clip on the unshared card stops no chat");
    assert_eq!(s.status()["accepts"], true);
    let b = s.gen("B", json!({}));
    assert_eq!(b["result"]["queued"], false, "{b}");
    s.running_on("B", 1);
    assert_eq!(*s.told.lock().unwrap(), vec!["none".to_string()], "the shared card: its idle chat model goes");

    // the top level describes the first, the list both, and a clip named its own
    let st = s.status();
    assert_eq!((st["idle"].clone(), st["job"]["label"].clone(), st["gpu"].clone(), st["slots_free"].clone(), st["accepts"].clone()), (json!(false), json!("A"), json!(0), json!(0), json!(false)), "{st}");
    assert_eq!(st["running_clips"].as_array().unwrap().len(), 2);
    let own = s.rpc("status", json!({"label": "B"}))["result"].clone();
    assert_eq!((own["job"]["label"].clone(), own["gpu"].clone()), (json!("B"), json!(1)), "{own}");

    // a third, with no card: busy only if it says queue: false; else queued, with or without queue: true
    let c = s.gen("C", json!({"queue": false}));
    assert_eq!((c["ok"].clone(), c["error"]["code"].clone()), (json!(false), json!("busy")), "{c}");
    let c = s.gen("C", json!({}));
    assert_eq!((c["result"]["queued"].clone(), c["result"]["position"].clone()), (json!(true), json!(1)), "{c}");
    assert_eq!(s.rpc("status", json!({"label": "C"}))["result"]["stage"], "queued");
    // the same call again is that clip, not one more
    let again = s.gen("C", json!({}));
    assert_eq!((again["result"]["reentered"].clone(), again["result"]["position"].clone()), (json!(true), json!(1)), "{again}");
    let d = s.gen("D", json!({"queue": true}));
    assert_eq!(d["result"]["position"], 2, "{d}");
    s.rpc("queue.remove", json!({"label": "D"}));

    // energy and preview are each clip's own: only the second card's counter rises, only B has a picture
    std::fs::write(s.dir.join("gpustat.json"), json!({"energy_j": 1000.0, "cards": [{"energy_j": 1000.0}, {"energy_j": 8600.0}]}).to_string()).unwrap();
    let bname = s.clip("B")["name"].as_str().unwrap().to_string();
    std::fs::write(s.dir.join("out").join(format!("{bname}.preview.png")), "png").unwrap();
    s.until("B's energy", |s| s.clip("B")["energy_wh"].as_f64() == Some(1.0));
    assert_eq!(s.clip("A")["energy_wh"].as_f64(), Some(0.0), "{}", s.clip("A"));
    assert!(s.clip("A")["preview"].is_null() && s.clip("B")["preview"].as_str().unwrap().contains(&bname));

    // naming no clip with two running is refused, naming one stops that one; the queued one takes its card
    let e = s.rpc("cancel", json!({}));
    assert_eq!(e["error"]["code"], "ambiguous", "{e}");
    assert_eq!(s.rpc("cancel", json!({"label": "B"}))["result"]["stopped"], true);
    s.until("C on B's card", |s| s.clip("C")["gpu"] == 1);
    assert!(s.clip("A")["gpu"] == 0, "A runs on");
    let gone = s.rpc("status", json!({"label": "B"}))["result"].clone();
    assert_eq!((gone["idle"].clone(), gone["failure"]["code"].clone()), (json!(true), json!("cancelled")), "{gone}");
    assert_eq!(s.status()["ended_clips"][0]["label"], "B");

    // A's record is A's
    let aname = s.clip("A")["name"].as_str().unwrap().to_string();
    s.release("A");
    s.until("A's record", |s| s.dir.join("out").join(format!("{aname}.json")).exists());
    let rec = s.rpc("jobs.get", json!({"name": aname}))["result"].clone();
    assert_eq!((rec["job"]["label"].clone(), rec["energy_wh"].as_f64()), (json!("A"), Some(0.0)), "{rec}");
    let done = s.rpc("status", json!({"label": "A"}))["result"].clone();
    assert_eq!((done["done"].clone(), done["file"].clone()), (json!(true), json!(format!("/out/{aname}.mp4"))), "{done}");
    assert_eq!(s.status()["job"]["label"], "C", "the first running is C now");
}

#[test]
fn a_restarted_studio_follows_both_and_a_re_submit_is_that_clip() {
    let mut s = start("re");
    s.gen("A", json!({"reentrant_id": "run#1", "seed": 7}));
    s.running_on("A", 0);
    s.gen("B", json!({"seed": 8}));
    s.until("both", |s| s.status()["running_clips"].as_array().unwrap().len() == 2);
    let child = s.studio.take().unwrap();
    { let mut c = child; let _ = c.kill(); let _ = c.wait(); }
    s.addr = format!("127.0.0.1:{}", free_port());
    s.studio = Some(start_studio(&s.dir, &s.addr, &s.door_addr, &s.env));
    s.until("the studio back", |s| TcpStream::connect(&s.addr).is_ok() && s.status()["running_clips"].as_array().is_some_and(|a| a.len() == 2));

    // by reentrant_id
    let r = s.gen("other label", json!({"reentrant_id": "run#1"}));
    assert_eq!((r["result"]["reentered"].clone(), r["result"]["state"].clone(), r["result"]["job"]["label"].clone()), (json!(true), json!("running"), json!("A")), "{r}");
    // by a running label, the same request
    let r = s.gen("B", json!({"seed": 8}));
    assert_eq!((r["result"]["reentered"].clone(), r["result"]["job"]["label"].clone()), (json!(true), json!("B")), "{r}");
    // the running label with another prompt, another seed or another id is not that clip
    for more in [json!({"prompt": "another", "seed": 8, "queue": true}), json!({"seed": 9, "queue": true})] {
        let r = s.gen("B", more.clone());
        assert_eq!(r["result"]["reentered"], false, "{more}: {r}");
    }
    let r = s.gen("A", json!({"seed": 7, "reentrant_id": "other", "queue": true}));
    assert_eq!(r["result"]["reentered"], false, "A has another id: {r}");
    assert_eq!(s.rpc("admission", json!({"prompt": "?", "reentrant_id": "run#1"}))["result"]["reentered"], true);
    assert_eq!(s.rpc("admission", json!({"reentrant_id": "run#1"}))["result"]["reentered"], true, "with the shape only (system.eta)");
    assert_eq!(s.rpc("admission", json!({"label": "B", "prompt": "B prompt", "seed": 8}))["result"]["reentered"], true);
    assert_eq!(s.rpc("admission", json!({"label": "new", "prompt": "new"}))["result"]["reentered"], false);
    s.rpc("queue.clear", json!({}));

    // once made: by reentrant_id it is the record; its label says nothing, a new clip may reuse it
    s.release("A");
    s.release("B");
    s.until("both made", |s| s.status()["idle"] == true);
    let r = s.gen("A", json!({"reentrant_id": "run#1"}));
    assert_eq!((r["result"]["reentered"].clone(), r["result"]["state"].clone()), (json!(true), json!("done")), "{r}");
    let r = s.gen("A", json!({"seed": 7}));
    assert_eq!((r["result"]["reentered"].clone(), r["result"]["queued"].clone()), (json!(false), json!(false)), "the old label on a new clip is new: {r}");
    assert_eq!(s.rpc("admission", json!({"label": "B", "prompt": "B prompt", "seed": 8}))["result"]["reentered"], false);
}

#[test]
fn chains_run_in_order_beside_each_other_and_prev_is_their_own() {
    let s = start("chain");
    s.gen("One 01/03: a", json!({"project": "one", "seed": 1}));
    s.running_on("One 01/03: a", 0);
    let name_a = s.clip("One 01/03: a")["name"].as_str().unwrap().to_string();

    // its next clip does not start beside it, though a card is free
    let chained = json!({"project": "one", "chain": {"from": "prev", "context_frames": 22}});
    let e = s.gen("One 02/03: b", { let mut m = chained.clone(); m["queue"] = json!(false); m });
    assert_eq!(e["error"]["code"], "busy", "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("One 01/03"), "{e}");
    assert_eq!(s.rpc("admission", json!({"prompt": "x", "project": "one", "label": "One 02/03: b", "first_frame": "prev"}))["result"]["gate"], "series");
    let q = s.gen("One 02/03: b", chained.clone());
    assert_eq!(q["result"]["queued"], true, "{q}");
    std::thread::sleep(Duration::from_millis(2500));
    assert!(s.clip("One 02/03: b").is_null(), "the next clip of a chain stays queued: {}", s.status());

    // the other chain goes past it, on the other card
    s.gen("Two 01/03: x", json!({"project": "two"}));
    s.running_on("Two 01/03: x", 1);

    // the other chain's clip is the newest made when One's first ends: One's next still carries on from One's
    s.release("Two 01/03: x");
    s.until("Two made", |s| s.clip("Two 01/03: x").is_null());
    s.release("One 01/03: a");
    s.until("One's next starts", |s| !s.clip("One 02/03: b").is_null());
    let rec = s.clip("One 02/03: b")["job"].clone();
    assert_eq!(rec["guide_clip"].as_str().unwrap(), name_a, "{rec}");
    assert_eq!(rec["first_audio"].as_str().unwrap(), name_a, "{rec}");
}

#[test]
fn a_chat_model_kept_on_the_shared_card_stays_and_that_card_is_no_slot() {
    let s = start("held");
    *s.door.lock().unwrap() = json!({"blocked": [1], "held": true});
    std::thread::sleep(Duration::from_millis(3500)); // the daemon's look at the door
    s.gen("A", json!({}));
    s.running_on("A", 0);
    let st = s.status();
    assert_eq!((st["slots_free"].clone(), st["accepts"].clone()), (json!(0), json!(false)), "{st}");
    let e = s.gen("B", json!({"queue": false}));
    assert_eq!(e["error"]["code"], "busy", "{e}");
    assert!(s.told.lock().unwrap().is_empty(), "nobody told the chat model anything");
    // the keep ends: the card is a slot again
    *s.door.lock().unwrap() = json!({"blocked": [], "held": false});
    s.until("the card back", |s| s.status()["slots_free"] == 1);
    assert_eq!(s.gen("B", json!({}))["result"]["queued"], false);
    s.running_on("B", 1);
    assert_eq!(*s.told.lock().unwrap(), vec!["none".to_string()]);
}

#[test]
fn host_memory_keeps_a_second_worker_out() {
    let s = start("ram");
    s.gen("A", json!({}));
    s.running_on("A", 0);
    assert_eq!(s.status()["slots_free"], 1);
    std::fs::write(s.dir.join("meminfo"), format!("MemAvailable: {} kB\n", 20 * 1024 * 1024)).unwrap();
    s.until("no room", |s| s.status()["slots_free"] == 0);
    assert!(s.status()["slots_why"].as_str().unwrap().contains("host memory"));
    let e = s.gen("B", json!({"queue": false}));
    assert_eq!(e["error"]["code"], "busy", "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("host memory"), "{e}");
    // queued, it waits for the memory or the card, it does not start
    s.gen("B", json!({}));
    std::thread::sleep(Duration::from_millis(2500));
    assert!(s.clip("B").is_null());
    std::fs::write(s.dir.join("meminfo"), format!("MemAvailable: {} kB\n", 43 * 1024 * 1024)).unwrap();
    s.until("B starts", |s| s.clip("B")["gpu"] == 1);
}

#[test]
fn holds_priority_and_a_failure_are_per_clip() {
    let s = start("hold");
    s.gen("A", json!({}));
    s.running_on("A", 0);
    // a paused queue starts nothing though a card is free; a direct clip still goes
    s.rpc("queue.pause", json!({"project": "held"}));
    s.gen("H", json!({"project": "held", "queue": true, "reentrant_id": "h"}));
    std::thread::sleep(Duration::from_millis(1500));
    assert!(s.clip("H").is_null(), "held");
    assert_eq!(s.status()["accepts"], true, "an all-held queue takes no card");
    let d = s.gen("D", json!({}));
    assert_eq!(d["result"]["queued"], false, "{d}");
    s.running_on("D", 1);
    // both cards busy, two queued with priorities: the higher first
    s.gen("low", json!({"queue": true, "priority": 0}));
    s.gen("high", json!({"queue": true, "priority": 5}));
    s.fail("D");
    // D failed: it goes round once more ahead of the queue
    s.until("D again", |s| !s.clip("D").is_null() && s.clip("D")["job"]["retry"] == true);
    s.fail("D");
    // failed twice in a row: the queue holds; A runs on, nothing else starts
    s.until("the hold", |s| s.status()["paused"] == true);
    std::thread::sleep(Duration::from_millis(1500));
    let st = s.status();
    assert_eq!((st["running_clips"].as_array().unwrap().len(), st["job"]["label"].clone()), (1, json!("A")), "{st}");
    s.rpc("queue.resume", json!({}));
    s.until("high first", |s| !s.clip("high").is_null());
    assert!(s.clip("low").is_null(), "the higher priority took the one card");
}

#[test]
fn a_clip_named_by_file_waits_and_a_lost_predecessor_is_an_answer_not_an_older_clip() {
    let s = start("dep");
    s.gen("P 01/03: a", json!({"project": "p", "reentrant_id": "p#1"}));
    s.running_on("P 01/03: a", 0);
    let name = s.clip("P 01/03: a")["name"].as_str().unwrap().to_string();
    s.release("P 01/03: a");
    s.until("the first made", |s| s.clip("P 01/03: a").is_null());

    // the second runs; one naming its file waits with a card free, the third by anchor too, an unrelated clip goes past
    s.gen("P 02/03: b", json!({"project": "p", "first_frame": "prev", "reentrant_id": "p#2"}));
    s.running_on("P 02/03: b", 0);
    assert_eq!(s.clip("P 02/03: b")["job"]["first_frame"].as_str().unwrap(), format!("{name}.mp4"));
    let second = s.clip("P 02/03: b")["name"].as_str().unwrap().to_string();
    let by_file = s.gen("byfile", json!({"guide_clip": format!("{second}:22:0"), "project": "elsewhere"}));
    assert_eq!(by_file["result"]["queued"], true, "{by_file}");
    // a file that does not exist yet: its clip is being made, so it waits rather than "not found"
    let by_frame = s.gen("byframe", json!({"first_frame": format!("{second}.mp4"), "project": "elsewhere2"}));
    assert_eq!(by_frame["result"]["queued"], true, "{by_frame}");
    assert_eq!(s.gen("nofile", json!({"first_frame": "h3_20200101_000000.mp4"}))["error"]["code"], "not_found");
    let third = s.gen("P 03/03: c", json!({"project": "p", "first_frame": "prev", "reentrant_id": "p#3"}));
    assert_eq!(third["result"]["queued"], true, "{third}");
    s.gen("free", json!({"project": "q"}));
    s.running_on("free", 1);
    assert!(s.clip("byfile").is_null() && s.clip("P 03/03: c").is_null());

    // the predecessor cancelled: its dependents are told, and do not start from the older clip
    assert_eq!(s.rpc("cancel", json!({"reentrant_id": "p#2"}))["result"]["stopped"], true);
    s.until("the dependents told", |s| s.rpc("status", json!({"reentrant_id": "p#3"}))["result"]["failure"]["code"] == "dependency_failed");
    assert_eq!(s.rpc("status", json!({"label": "byfile"}))["result"]["failure"]["code"], "dependency_failed");
    assert_eq!(s.rpc("status", json!({"label": "byframe"}))["result"]["failure"]["code"], "dependency_failed");
    assert!(s.status()["queue_items"].as_array().unwrap().is_empty());
    assert!(s.status()["running_clips"].as_array().unwrap().iter().all(|c| c["label"] == "free"), "{}", s.status());
    let e = s.gen("P 04/04: d", json!({"project": "p", "first_frame": "prev"}));
    assert_eq!(e["error"]["code"], "dependency_failed", "{e}");

    // made again, the chain carries on, from the new one, on whichever card is free
    let r = s.gen("P 02/03: b", json!({"project": "p", "first_frame": "prev", "reentrant_id": "p#2"}));
    assert_eq!(r["result"]["queued"], false, "its own failure is not in its way: {r}");
    s.until("the retake", |s| !s.clip("P 02/03: b").is_null());
    let retake = s.clip("P 02/03: b")["name"].as_str().unwrap().to_string();
    s.release("P 02/03: b");
    s.until("the retake made", |s| s.clip("P 02/03: b").is_null());
    s.release("free");
    let r = s.gen("P 03/03: c", json!({"project": "p", "first_frame": "prev", "reentrant_id": "p#3"}));
    assert_eq!(r["ok"], true, "{r}");
    s.until("the third", |s| !s.clip("P 03/03: c").is_null());
    assert_eq!(s.clip("P 03/03: c")["job"]["first_frame"].as_str().unwrap(), format!("{retake}.mp4"));
}

#[test]
fn a_studio_restarted_mid_clip_reattaches_also_from_the_old_builds_file() {
    let mut s = start("old");
    s.gen("A", json!({"reentrant_id": "a"}));
    s.running_on("A", 0);
    let job = s.job_of("A");
    { let mut c = s.studio.take().unwrap(); let _ = c.kill(); let _ = c.wait(); }
    // what the production studio leaves: job.json only, its record without per-card counters
    std::fs::remove_file(s.dir.join("state/jobs.json")).unwrap();
    let mut rec: Value = serde_json::from_slice(&std::fs::read(s.dir.join("state/job.json")).unwrap()).unwrap();
    rec.as_object_mut().unwrap().remove("energy0_cards");
    std::fs::write(s.dir.join("state/job.json"), rec.to_string()).unwrap();
    s.studio = Some(start_studio(&s.dir, &s.addr, &s.door_addr, &s.env));
    s.until("A followed", |s| TcpStream::connect(&s.addr).is_ok() && s.clip("A")["gpu"] == 0);
    assert_eq!(s.job_of("A"), job, "the same engine job");
    assert_eq!(std::fs::read_to_string(s.dir.join("workers")).unwrap().lines().count(), 1, "no second worker");
    let name = s.clip("A")["name"].as_str().unwrap().to_string();
    s.release("A");
    s.until("its record", |s| s.dir.join("out").join(format!("{name}.json")).exists());
}
