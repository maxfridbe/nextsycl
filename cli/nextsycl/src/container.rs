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
        Ok(Ce { bin, image: cfg.or("NS_IMAGE", "localhost/h3-build") })
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
            Err(format!("the image {} is not there (NS_IMAGE: an image with the oneAPI runtime, as ./build.sh uses)", self.image))
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
