//! The server as a container (`nextsycl`, runs this program's `serve`, needs the GPUs) - sycl-h3's scheme. Podman by
//! default (rootless, `crun` for the GPU's render group), docker if that is what there is.

use std::path::Path;
use std::process::{Command, Stdio};

use nextsycl_models::Config;

pub const SERVER: &str = "nextsycl";
/// Where the socket directory appears inside the container.
pub const SOCKET_DIR_IN: &str = "/run/nextsycl";

pub struct Ce {
    pub bin: String,
    pub image: String,
}

fn found(cmd: &str) -> bool {
    Command::new(cmd).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

fn group_id(name: &str) -> Option<String> {
    std::fs::read_to_string("/etc/group")
        .ok()?
        .lines()
        .find(|l| l.split(':').next() == Some(name))
        .and_then(|l| l.split(':').nth(2))
        .map(str::to_string)
}

impl Ce {
    pub fn new(cfg: &Config) -> Result<Ce, String> {
        let bin = match cfg.get("NS_CONTAINER_ENGINE") {
            Some(b) => b,
            None if found("podman") => "podman".into(),
            None if found("docker") => "docker".into(),
            None => return Err("need podman (or docker) on PATH".into()),
        };
        Ok(Ce { bin, image: cfg.or("NS_IMAGE", "localhost/nextsycl-build") })
    }

    pub fn cmd(&self) -> Command {
        Command::new(&self.bin)
    }

    pub fn running(&self, name: &str) -> bool {
        self.cmd()
            .args(["container", "inspect", "-f", "{{.State.Running}}", name])
            .stderr(Stdio::null())
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
    }

    pub fn need_image(&self) -> Result<(), String> {
        let ok = self.cmd().args(["image", "inspect", &self.image]).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success());
        if ok {
            Ok(())
        } else {
            Err(format!("the image {} is not there (./build.sh image makes it; NS_IMAGE names another with the oneAPI runtime)", self.image))
        }
    }

    /// Files the container writes must belong to the caller: rootless podman maps its root to the caller, docker
    /// needs the ids.
    pub fn user_args(&self) -> Vec<String> {
        if self.bin.ends_with("podman") {
            vec!["--security-opt".into(), "label=disable".into()]
        } else {
            let id = |f: &str| Command::new("id").arg(f).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
            vec!["--user".into(), format!("{}:{}", id("-u"), id("-g")), "-e".into(), "HOME=/tmp".into()]
        }
    }

    /// The GPUs: the device nodes, and the groups that may open them.
    pub fn gpu_args(&self) -> Vec<String> {
        if self.bin.ends_with("podman") {
            // only crun can hand a rootless container the caller's groups (keep-groups)
            if !found("crun") {
                eprintln!("note: crun is not installed; rootless podman cannot reach /dev/dri without it");
            }
            vec!["--runtime".into(), "crun".into(), "--device".into(), "/dev/dri".into(), "--group-add".into(), "keep-groups".into()]
        } else {
            let mut v = vec!["--device".to_string(), "/dev/dri".to_string()];
            for g in ["render", "video"] {
                if let Some(id) = group_id(g) {
                    v.extend(["--group-add".to_string(), id]);
                }
            }
            v
        }
    }

    pub fn remove(&self, name: &str) {
        let _ = self.cmd().args(["rm", "-f", name]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }

    /// `podman stop -t <secs>`: SIGTERM, and the kill only after the grace time.
    pub fn stop(&self, name: &str, secs: u32) {
        let _ = self.cmd().args(["stop", "-t", &secs.to_string(), name]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
}

pub fn mount(host: &Path, inside: &str, ro: bool) -> Vec<String> {
    vec!["-v".into(), format!("{}:{inside}{}", host.display(), if ro { ":ro" } else { "" })]
}

/// Whether a command runs the kernels in this process (the GPU runtime: oneAPI's libraries)
pub fn needs_runtime(args: &[String]) -> bool {
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    let cmd = if a.first() == Some(&"llm") { &a[1..] } else { &a[..] };
    a.contains(&"--here")
        || matches!(cmd.first(), Some(&("gpus" | "generate" | "spec-check" | "batch-check" | "kernels" | "check" | "selftest")))
        || (cmd.first() == Some(&"serve") && cmd.get(1).is_some_and(|m| Path::new(m).exists()))
        || (matches!(a.first(), Some(&("image" | "video" | "audio" | "model3d" | "3d"))) && matches!(a.get(1), Some(&("selftest" | "check"))))
}

/// Whether the kernel library loads here: every library it links found (oneAPI's are inside the image, not on a
/// plain host)
fn runtime_here(cfg: &Config) -> bool {
    let lib = cfg.dist.join("libnextsycl-llm.so");
    Command::new("ldd").arg(&lib).output().is_ok_and(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).contains("not found"))
}

/// A command that needs the GPU runtime, on a host without it: this program again, the same arguments, inside the
/// image (NS_IMAGE) with the GPUs - its directory, the settings file, the working directory and every path the
/// arguments name mounted at the same paths (a file read-only, where a new one goes read-write), the settings in the
/// environment passed on. None: run here.
pub fn rerun_in_image(cfg: &Config, args: &[String]) -> Option<Result<i32, String>> {
    if std::env::var_os("NS_IN_IMAGE").is_some() || !needs_runtime(args) || runtime_here(cfg) {
        return None;
    }
    Some((|| {
        let ce = Ce::new(cfg)?;
        ce.need_image()?;
        let exe = std::env::current_exe().and_then(|p| p.canonicalize()).map_err(|e| e.to_string())?;
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
        let mut mounts: Vec<(std::path::PathBuf, bool)> = vec![(cfg.dist.canonicalize().unwrap_or(cfg.dist.clone()), true), (cwd.clone(), false)];
        if cfg.file().exists() {
            mounts.push((cfg.file().canonicalize().unwrap_or(cfg.file()), true));
        }
        if let Some(m) = cfg.get("NS_MODELS").map(std::path::PathBuf::from).filter(|p| p.is_dir()) {
            mounts.push((m, true));
        }
        // the registry (a job names its model by id) and the files of its models, read-only
        let reg = nextsycl_models::registry::registry(cfg);
        if let Some(d) = reg.parent().filter(|d| d.is_dir()) {
            mounts.push((d.to_path_buf(), true));
        }
        for m in nextsycl_models::registry::all(cfg).unwrap_or_default() {
            for p in nextsycl_models::registry::paths(&m) {
                if let Some(d) = if p.is_dir() { Some(p.clone()) } else { p.parent().map(Path::to_path_buf) } {
                    if d.is_dir() && args.iter().any(|a| m["id"] == a.as_str()) {
                        mounts.push((d, true));
                    }
                }
            }
        }
        for a in args.iter().filter(|a| !a.starts_with("--")) {
            let p = cwd.join(a);
            if let Ok(c) = p.canonicalize() {
                if let Some(d) = if c.is_dir() { Some(c.clone()) } else { c.parent().map(Path::to_path_buf) } {
                    mounts.push((d, true));
                }
            } else if a.contains('/') {
                // a file to be written: its directory, writable
                if let Some(d) = p.parent().and_then(|d| d.canonicalize().ok()) {
                    mounts.push((d, false));
                }
            }
        }
        let mut c = ce.cmd();
        c.args(["run", "--rm", "-i", "--network", "host"]);
        if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            c.arg("-t");
        }
        c.args(ce.user_args()).args(ce.gpu_args());
        // one mount a path, read-only unless any use of it writes
        let mut by_path: std::collections::BTreeMap<std::path::PathBuf, bool> = std::collections::BTreeMap::new();
        for (p, ro) in mounts {
            let e = by_path.entry(p).or_insert(true);
            *e &= ro;
        }
        for (p, ro) in &by_path {
            c.args(mount(p, &p.to_string_lossy(), *ro));
        }
        c.args(["-w", &cwd.to_string_lossy(), "-e", "NS_IN_IMAGE=1"]);
        // the same home (the default registry and settings under ~/.config)
        if let Ok(h) = std::env::var("HOME") {
            c.args(["-e", &format!("HOME={h}")]);
        }
        if std::env::var_os("ONEAPI_DEVICE_SELECTOR").is_none() {
            c.args(["-e", "ONEAPI_DEVICE_SELECTOR=level_zero:*"]);
        }
        for (k, _) in std::env::vars() {
            if ["NS_", "NSD_", "H3_", "STRATA_", "ONEAPI_", "SYCL_", "ZE_"].iter().any(|p| k.starts_with(p)) {
                c.args(["-e", &k]);
            }
        }
        c.arg(&ce.image).args(["bash", "-c", "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec \"$0\" \"$@\""]).arg(&exe).args(args);
        let st = c.status().map_err(|e| format!("{}: {e}", ce.bin))?;
        Ok(st.code().unwrap_or(1))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn every_kinds_self_check_runs_in_the_image() {
        for kind in ["llm", "image", "audio", "video", "model3d", "3d"] {
            assert!(needs_runtime(&a(&format!("{kind} selftest"))), "{kind} selftest runs the kernels");
        }
        for kind in ["image", "audio", "video", "model3d", "3d"] {
            assert!(needs_runtime(&a(&format!("{kind} check x"))), "{kind} check runs the kernels");
        }
        assert!(needs_runtime(&a("gpus")));
    }

    #[test]
    fn commands_that_only_manage_do_not_need_the_runtime() {
        for c in ["models list", "maintenance on", "stop", "image engines", "model3d engines", "llm engines"] {
            assert!(!needs_runtime(&a(c)), "{c} is bookkeeping");
        }
    }
}
