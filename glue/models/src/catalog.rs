//! The catalog of supported models: the model cards (`model-cards/<kind>/<id>.yml`, read when it is needed -
//! `NS_MODEL_CARDS` names another directory; model-cards/README.md has the format). A card is one model: what it is,
//! every file it needs as a direct link pinned to a revision with its size and SHA-256, the settings its registry
//! entry gets, its prompt guide. `nextsycl models search` lists them; `nextsycl models pull <id>` downloads what is
//! missing, checks it, and registers the model.
//!
//! Files are keyed once and shared by key: a card defines a file (`key`, `url`, `size`, `sha256`, `path`) or names
//! one another card defines (two models naming the same file download it once); a file with the same content
//! already on the disk under another name (the Coder's shard 2 is the original's) is hard-linked, not fetched
//! again. Some parts are made locally rather than downloaded (Qwen3.8-Flash-Next's MTP draft layer is built by
//! Strata's tools): a model lists them under "local", and `pull` uses one it finds (the setting in nextsycl.conf,
//! or another registry entry's) or registers the model without it.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;

/// The kinds of model, in the order `search` lists them
pub const KINDS: &[&str] = &["llm", "image", "video", "audio", "model3d", "lora"];

/// The catalog: its files by key, its models (each card as JSON, its files as role -> key)
pub struct Catalog {
    pub files: Map<String, Value>,
    pub models: Vec<Value>,
    /// the cards' directory (a card's guide is relative to it)
    pub dir: PathBuf,
}

/// The model cards' directory: `NS_MODEL_CARDS`, else model-cards/ beside the program (a release), else in the
/// repository it was built in (dist/../model-cards)
pub fn cards_dir(cfg: &Config) -> PathBuf {
    if let Some(d) = cfg.get("NS_MODEL_CARDS") {
        return PathBuf::from(d);
    }
    let here = cfg.dist.join("model-cards");
    if here.is_dir() {
        return here;
    }
    cfg.dist.join("../model-cards")
}

impl Catalog {
    /// The cards of `cards_dir`
    pub fn load(cfg: &Config) -> Result<Catalog, String> {
        Catalog::from_dir(&cards_dir(cfg))
    }

    /// Every `<kind>/<id>.yml` under `dir`: checked (the id is the file's name, the kind its folder, each file
    /// defined once and every key named defined), in kind order, then by id
    pub fn from_dir(dir: &Path) -> Result<Catalog, String> {
        let mut cards = Vec::new();
        for kind in KINDS {
            let Ok(rd) = std::fs::read_dir(dir.join(kind)) else { continue };
            let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "yml" || x == "yaml")).collect();
            paths.sort();
            for p in paths {
                let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                let v: Value = serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
                let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                if v["id"].as_str() != Some(stem.as_str()) {
                    return Err(format!("{}: its id is not its file's name ({stem})", p.display()));
                }
                if v["kind"].as_str() != Some(kind) {
                    return Err(format!("{}: its kind is not its folder ({kind})", p.display()));
                }
                // its concurrency and coexistence (crates/scheduler `profile`; absent: one copy, shared): bad values refused here, by file
                backplane_scheduler::profile::Profile::from_card(&v).map_err(|e| format!("{}: {e}", p.display()))?;
                // a picture-making model's request rules (crate::controls; schema 1): bad values refused here, by file
                if let Some(c) = v.get("controls") {
                    crate::controls::check(c).map_err(|e| format!("{}: {e}", p.display()))?;
                }
                span_check(&v).map_err(|e| format!("{}: {e}", p.display()))?;
                cards.push((p, v));
            }
        }
        if cards.is_empty() {
            return Err(format!("{}: no model cards (<kind>/<id>.yml; NS_MODEL_CARDS names another directory)", dir.display()));
        }
        let mut files = Map::new();
        let mut models = Vec::with_capacity(cards.len());
        let mut named = Vec::new();
        for (p, mut v) in cards {
            if let Some(fs) = v.get_mut("files").and_then(Value::as_object_mut) {
                for (role, f) in fs.iter_mut() {
                    if let Some(def) = f.as_object() {
                        let key = def.get("key").and_then(Value::as_str).ok_or_else(|| format!("{}: files.{role} has no key", p.display()))?.to_string();
                        let mut body = def.clone();
                        body.remove("key");
                        if files.get(&key).is_some_and(|x| x != &Value::Object(body.clone())) {
                            return Err(format!("{}: files.{role}: {key} defined otherwise in another card", p.display()));
                        }
                        files.insert(key.clone(), Value::Object(body));
                        *f = Value::String(key);
                    } else if let Some(key) = f.as_str() {
                        named.push((p.clone(), role.clone(), key.to_string()));
                    } else {
                        return Err(format!("{}: files.{role}: a file (key, url, size, sha256, path) or another card's file key", p.display()));
                    }
                }
            }
            models.push(v);
        }
        for (p, role, key) in named {
            if !files.contains_key(&key) {
                return Err(format!("{}: files.{role} names {key}, which no card defines", p.display()));
            }
        }
        let mut ids = BTreeSet::new();
        for m in &models {
            if !ids.insert(s(m, "id")) {
                return Err(format!("{}: two cards", s(m, "id")));
            }
        }
        Ok(Catalog { files, models, dir: dir.to_path_buf() })
    }

    /// The prompt guide (Markdown) for a registry entry: its card's (the one it was pulled from, or of its id), else
    /// a card's of the same kind and architecture, else of the same kind
    pub fn guide(&self, entry: &Value) -> Option<String> {
        let kind = entry.get("kind").and_then(Value::as_str).unwrap_or("llm");
        let has = |m: &&Value| m.get("guide").and_then(Value::as_str).is_some();
        let card = [s(entry, "catalog"), s(entry, "id")].iter().find_map(|id| self.models.iter().filter(has).find(|m| !id.is_empty() && s(m, "id") == *id))
            .or_else(|| self.models.iter().filter(has).find(|m| s(m, "kind") == kind && s(m, "arch") == s(entry, "arch")))
            .or_else(|| self.models.iter().filter(has).find(|m| s(m, "kind") == kind))?;
        std::fs::read_to_string(self.dir.join(s(card, "guide"))).ok()
    }

    pub fn model(&self, id: &str) -> Option<&Value> {
        self.models.iter().find(|m| m["id"] == id)
    }

    /// A model's files: (role, key, the file's entry)
    fn files_of<'a>(&'a self, m: &'a Value) -> Vec<(&'a str, &'a str, &'a Value)> {
        m["files"].as_object().map_or_else(Vec::new, |o| {
            o.iter().filter_map(|(role, key)| {
                let k = key.as_str()?;
                Some((role.as_str(), k, self.files.get(k)?))
            }).collect()
        })
    }

    /// Bytes to download for a model (each distinct file once)
    pub fn bytes(&self, m: &Value) -> u64 {
        let keys: BTreeSet<&str> = self.files_of(m).iter().map(|f| f.1).collect();
        keys.iter().filter_map(|k| self.files.get(*k)?["size"].as_u64()).sum()
    }

    /// A model's files as `pull` lays them out: (path under the models folder, bytes), each distinct file once - what
    /// the front door watches to show a download's progress
    pub fn file_list(&self, m: &Value) -> Vec<(String, u64)> {
        let mut seen = BTreeSet::new();
        self.files_of(m).into_iter().filter(|(_, k, _)| seen.insert(*k)).map(|(_, _, f)| (s(f, "path"), f["size"].as_u64().unwrap_or(0))).collect()
    }
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// `nextsycl models search [TEXT] [--kind K]`
pub fn search(cfg: &Config, installed: &[Value], text: Option<&str>, kind: Option<&str>) -> Result<(), String> {
    let cat = Catalog::load(cfg)?;
    let want = text.map(str::to_lowercase);
    println!("{:<44} {:<6} {:>8}  {:<6} {:<10} {:<18} TITLE", "ID", "KIND", "SIZE", "GPUS", "INSTALLED", "SILO");
    for m in &cat.models {
        let id = s(m, "id");
        if kind.is_some_and(|k| s(m, "kind") != k) {
            continue;
        }
        if let Some(w) = &want {
            let hay = format!("{id} {} {}", s(m, "title"), s(m, "about")).to_lowercase();
            if !hay.contains(w.as_str()) {
                continue;
            }
        }
        let have = if installed.iter().any(|x| x["id"] == id.as_str()) { "yes" } else { "-" };
        println!("{:<44} {:<6} {:>7.1}G  {:<6} {:<10} {:<18} {}", id, s(m, "kind"), gib(cat.bytes(m)), s(m, "gpus"), have, silo_of(cfg, m), s(m, "title"));
    }
    Ok(())
}

/// The file a path names, through a symbolic link (one to another filesystem, from `pull`): a container sees the
/// directories it is given, not where a link inside them points
fn real(p: &Path) -> PathBuf {
    if std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_symlink()) {
        return p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    }
    p.to_path_buf()
}

/// The silo a card's model runs on (its tuned kernels: dist/silo/libnextsycl-<name>.so): its card's `silo`, else its
/// architecture's when one is built; `-` the shared kernels; "(not built)" a silo the card names that is not there
pub fn silo_of(cfg: &Config, m: &Value) -> String {
    if m["silo"] == "none" {
        return "-".into();
    }
    let named = m["silo"].as_str().or_else(|| m["silo"]["name"].as_str()).map(str::to_string);
    let name = named.clone().unwrap_or_else(|| s(m, "arch"));
    let built = cfg.dist.join("silo").join(format!("libnextsycl-{name}.so")).exists();
    match (built, named.is_some(), m["silo"]["required"] == true) {
        (true, _, true) => format!("{name} (required)"),
        (true, _, false) => name,
        (false, true, _) => format!("{name} (not built)"),
        (false, false, _) => "-".into(),
    }
}

/// A file's SHA-256, hex, reading it once
/// The Hugging Face token: HF_TOKEN, else huggingface_hub's file ($HF_TOKEN_PATH, $HF_HOME/token,
/// ~/.cache/huggingface/token)
fn hf_token() -> Option<String> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    if let Some(t) = env("HF_TOKEN") {
        return Some(t.trim().to_string());
    }
    let file = env("HF_TOKEN_PATH").map(PathBuf::from)
        .or_else(|| env("HF_HOME").map(|h| Path::new(&h).join("token")))
        .or_else(|| env("HOME").map(|h| Path::new(&h).join(".cache/huggingface/token")))?;
    std::fs::read_to_string(file).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty() && !t.contains(char::is_whitespace))
}

fn sha256_of(p: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", p.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `name` of `size` bytes somewhere under `dir` (a copy the machine already has)
fn find_copy(dir: &Path, name: &str, size: u64, depth: usize) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        let Ok(md) = std::fs::metadata(&p) else { continue };
        if md.is_dir() {
            subdirs.push(p);
        } else if p.file_name().is_some_and(|n| n == name) && md.len() == size {
            return Some(p);
        }
    }
    if depth == 0 {
        return None;
    }
    subdirs.iter().find_map(|d| find_copy(d, name, size, depth - 1))
}

/// One file in place at `target`: there already (the right size; `verify` hashes it too), linked from a file with the
/// same content (another catalog path under `root`, or a copy found under `from`), or downloaded and checked
fn fetch(cat: &Catalog, root: &Path, key: &str, f: &Value, from: &[PathBuf], verify: bool) -> Result<PathBuf, String> {
    let target = root.join(s(f, "path"));
    let size = f["size"].as_u64().unwrap_or(0);
    let sha = s(f, "sha256");
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if std::fs::metadata(&target).is_ok_and(|m| m.len() == size) {
        if verify && sha256_of(&target)? != sha {
            return Err(format!("{}: not the file the catalog names (SHA-256 differs)", target.display()));
        }
        println!("  {key}: have it ({})", target.display());
        return Ok(real(&target));
    }
    if let Some(d) = target.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    // the same content under another catalog path, or a copy elsewhere on the machine (by name and size)
    let same = cat.files.iter().filter(|(k, o)| *k != key && s(o, "sha256") == sha).map(|(_, o)| root.join(s(o, "path")))
        .find(|p| std::fs::metadata(p).is_ok_and(|m| m.len() == size));
    let copy = same.or_else(|| from.iter().find_map(|d| find_copy(d, &name, size, 4)));
    // a small copy found by its name and size is hashed whatever --verify says, and passed over when it differs:
    // waifu2x's three scale2.0x_model.param files share a name and a size, and the wrong one linked runs the wrong model
    let copy = match copy {
        Some(src) if !verify && size < (256 << 20) && sha256_of(&src)? != sha => {
            println!("  {key}: {} has the name and size but not the content: not used", src.display());
            None
        }
        c => c,
    };
    if let Some(src) = copy {
        if verify && sha256_of(&src)? != sha {
            return Err(format!("{}: not the file the catalog names (SHA-256 differs)", src.display()));
        }
        let _ = std::fs::remove_file(&target);
        if std::fs::hard_link(&src, &target).is_err() {
            // another filesystem: a symbolic link, and the registry names the file itself
            std::os::unix::fs::symlink(&src, &target).map_err(|e| format!("{}: {e}", target.display()))?;
        }
        println!("  {key}: linked from {}", src.display());
        return Ok(real(&target));
    }
    let part = target.with_file_name(format!("{name}.part"));
    let url = s(f, "url");
    if let Some(pieces) = f["parts"].as_array() {
        // published as byte ranges of the file (name.part01-of-40 ...): each downloaded and checked, then joined
        println!("  {key}: downloading {:.2} GiB in {} parts", gib(size), pieces.len());
        join_parts(pieces, &target, &part)?;
    } else if let Some(patterns) = tensor_patterns(&f["tensors"]) {
        // a part of a larger safetensors file: its header, then only the tensors the patterns name, by byte range
        println!("  {key}: downloading {:.2} GiB\n    {url}", gib(size));
        fetch_tensors(&url, &patterns, &part)?;
    } else if let Some(member) = f["member"].as_str() {
        // one file of a zip archive (a release's bundle: realesrgan-ncnn-vulkan's models/): the archive downloaded once
        // under <root>/.archives (resumed like any file), the member taken out of it
        fetch_member(&url, member, root, &part)?;
    } else if let Some(page) = f["gated"].as_str() {
        // a gated Hugging Face file: the account's token (HF_TOKEN, else huggingface_hub's token file), sent to curl
        // on its standard input (never on a command line), and a clear way out when there is none or it is refused
        let token = hf_token().ok_or_else(|| format!("{key}: {url} is gated - accept its terms at {page} with a Hugging Face account, then log in \
                                                      (`hf auth login`, or HF_TOKEN set to a token of that account) and pull again"))?;
        println!("  {key}: downloading {:.2} GiB (gated, with the Hugging Face token)\n    {url}", gib(size));
        let mut ch = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-K", "-", "-o"]).arg(&part).arg(&url)
            .stdin(std::process::Stdio::piped()).spawn().map_err(|e| format!("curl: {e}"))?;
        {
            use std::io::Write;
            let mut si = ch.stdin.take().ok_or("curl: no stdin")?;
            si.write_all(format!("header = \"Authorization: Bearer {token}\"\n").as_bytes()).map_err(|e| format!("curl: {e}"))?;
        }
        let st = ch.wait().map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!("{url}: the download failed - a refusal (HTTP 401 / 403) means the token's account has not been granted {page} \
                                yet; otherwise run pull again: it resumes"));
        }
    } else {
        println!("  {key}: downloading {:.2} GiB\n    {url}", gib(size));
        let st = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o"]).arg(&part).arg(&url).status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!("{url}: the download failed (run pull again: it resumes)"));
        }
    }
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if got != size {
        return Err(format!("{}: {got} bytes, the catalog says {size}", part.display()));
    }
    print!("    checking its SHA-256 ... ");
    let h = sha256_of(&part)?;
    if h != sha {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{url}: SHA-256 {h}, the catalog says {sha} (the partial file is removed: pull again)"));
    }
    println!("ok");
    std::fs::rename(&part, &target).map_err(|e| format!("{}: {e}", target.display()))?;
    Ok(target)
}

/// A file published in byte ranges (`parts`: each a url, size and sha256, in order) into `out`: each part downloaded
/// beside `target` (resumed when run again) and checked by its own SHA-256, then appended; the parts removed once
/// joined. The whole file's size and SHA-256 are checked after, as for any file.
fn join_parts(pieces: &[Value], target: &Path, out: &Path) -> Result<(), String> {
    use std::io::Write;
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut paths = Vec::with_capacity(pieces.len());
    for (i, pc) in pieces.iter().enumerate() {
        let (url, size, sha) = (s(pc, "url"), pc["size"].as_u64().unwrap_or(0), s(pc, "sha256"));
        let p = target.with_file_name(format!("{name}.piece{:02}", i + 1));
        let done = target.with_file_name(format!("{name}.piece{:02}.ok", i + 1));
        if !(done.exists() && std::fs::metadata(&p).is_ok_and(|m| m.len() == size)) {
            println!("    part {} of {}: {url}", i + 1, pieces.len());
            let st = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o"]).arg(&p).arg(&url).status()
                .map_err(|e| format!("curl: {e}"))?;
            let got = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            if !st.success() || got != size {
                return Err(format!("{url}: the download failed ({got} of {size} bytes; run pull again: it resumes)"));
            }
            let h = sha256_of(&p)?;
            if h != sha {
                let _ = std::fs::remove_file(&p);
                return Err(format!("{url}: SHA-256 {h}, the catalog says {sha} (the part is removed: pull again)"));
            }
            std::fs::write(&done, b"").map_err(|e| format!("{}: {e}", done.display()))?;
        }
        paths.push((p, done));
    }
    println!("    joining the parts");
    let mut f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for (p, _) in &paths {
        let mut r = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        std::io::copy(&mut r, &mut f).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    f.flush().map_err(|e| e.to_string())?;
    for (p, done) in &paths {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(done);
    }
    Ok(())
}

/// `member` of the zip archive at `url` into `out` (`unzip -p`), the archive kept under `<root>/.archives` for the
/// archive's other members
fn fetch_member(url: &str, member: &str, root: &Path, out: &Path) -> Result<(), String> {
    let dir = root.join(".archives");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let name = url.rsplit('/').next().filter(|n| !n.is_empty()).unwrap_or("archive.zip");
    let zip = dir.join(name);
    if !zip.is_file() {
        let part = dir.join(format!("{name}.part"));
        let st = Command::new("curl").args(["-fL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o"]).arg(&part).arg(url).status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!("{url}: the download failed (run pull again: it resumes)"));
        }
        std::fs::rename(&part, &zip).map_err(|e| format!("{}: {e}", zip.display()))?;
    }
    let o = Command::new("unzip").arg("-p").arg(&zip).arg(member).output().map_err(|e| format!("unzip: {e} (the archive's files need unzip)"))?;
    if !o.status.success() || o.stdout.is_empty() {
        return Err(format!("{}: no {member} in it", zip.display()));
    }
    std::fs::write(out, &o.stdout).map_err(|e| format!("{}: {e}", out.display()))
}

/// Bytes [a, b) of a URL
fn range(url: &str, a: u64, b: u64) -> Result<Vec<u8>, String> {
    let o = Command::new("curl").args(["-fsL", "--retry", "5", "--retry-delay", "5", "-r", &format!("{a}-{}", b - 1)]).arg(url).output()
        .map_err(|e| format!("curl: {e}"))?;
    if !o.status.success() || o.stdout.len() as u64 != b - a {
        return Err(format!("{url}: bytes {a}-{b}: the download failed ({} bytes)", o.stdout.len()));
    }
    Ok(o.stdout)
}

/// A card file's `tensors`: one prefix (`visual.`), or a list of them, where `*` stands for any run of characters
/// (`blocks.*.adaln_proj.`); a name is kept when it starts with what one of them matches
fn tensor_patterns(v: &Value) -> Option<Vec<String>> {
    match v {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => Some(a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).filter(|p: &Vec<String>| !p.is_empty()),
        _ => None,
    }
}

/// Whether `name` starts with what `pattern` matches (`*`: any run of characters)
fn tensor_match(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => name.starts_with(pattern),
        Some((head, rest)) => name.strip_prefix(head).is_some_and(|tail| (0..=tail.len()).filter(|i| tail.is_char_boundary(*i)).any(|i| tensor_match(rest, &tail[i..]))),
    }
}

/// The tensors of a remote safetensors file that the patterns name (`tensor_match`), as a safetensors file of their own
/// (the header rewritten - compact, keys sorted - the data in the source's order, read in 64 MiB ranges); the same
/// bytes every time, so its SHA-256 can be in the catalog
fn fetch_tensors(url: &str, patterns: &[String], out: &Path) -> Result<(), String> {
    use std::io::Write;
    let n = u64::from_le_bytes(range(url, 0, 8)?.try_into().map_err(|_| "a short header")?);
    let head: Value = serde_json::from_slice(&range(url, 8, 8 + n)?).map_err(|e| format!("{url}: its header: {e}"))?;
    let mut keep: Vec<(String, Value)> = head.as_object().into_iter().flatten()
        .filter(|(k, _)| k.as_str() != "__metadata__" && patterns.iter().any(|p| tensor_match(p, k))).map(|(k, v)| (k.clone(), v.clone())).collect();
    if keep.is_empty() {
        return Err(format!("{url}: no tensor matches {patterns:?}"));
    }
    let off = |v: &Value, i: usize| v["data_offsets"][i].as_u64().unwrap_or(0);
    keep.sort_by_key(|(_, v)| off(v, 0));
    let mut new = Map::new();
    let mut at = 0u64;
    for (k, v) in &keep {
        let len = off(v, 1) - off(v, 0);
        new.insert(k.clone(), serde_json::json!({"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [at, at + len]}));
        at += len;
    }
    if let Some(m) = head.get("__metadata__") {
        new.insert("__metadata__".into(), m.clone());
    }
    // the header compact with its keys sorted at every level, padded to 8 bytes
    let mut h = serde_json::to_string(&sorted(Value::Object(new))).map_err(|e| e.to_string())?;
    while h.len() % 8 != 0 {
        h.push(' ');
    }
    let mut f = std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    f.write_all(&(h.len() as u64).to_le_bytes()).and_then(|_| f.write_all(h.as_bytes())).map_err(|e| e.to_string())?;
    let base = 8 + n;
    for (_, v) in &keep {
        let (mut a, b) = (off(v, 0), off(v, 1));
        while a < b {
            let e = b.min(a + (64 << 20));
            f.write_all(&range(url, base + a, base + e)?).map_err(|e| e.to_string())?;
            a = e;
        }
    }
    Ok(())
}

/// The registry entry a pulled model gets. `local`: the made-locally parts found (role -> path)
/// A JSON value with its keys sorted at every level. The workspace's serde_json keeps insertion order
/// (preserve_order), so its Map does not sort them itself; a safetensors header is written sorted
fn sorted(v: Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.into_iter().collect::<BTreeMap<_, _>>().into_iter().map(|(k, v)| (k, sorted(v))).collect()),
        Value::Array(a) => Value::Array(a.into_iter().map(sorted).collect()),
        v => v,
    }
}

fn entry(m: &Value, paths: &BTreeMap<String, PathBuf>, local: &BTreeMap<String, String>) -> Value {
    let sub = |v: &str| -> Option<String> {
        let mut out = v.to_string();
        while let Some(a) = out.find('{') {
            let b = out[a..].find('}')? + a;
            let role = &out[a + 1..b];
            let p = paths.get(role).map(|p| p.to_string_lossy().into_owned()).or_else(|| local.get(role).cloned())?;
            out.replace_range(a..=b, &p);
        }
        Some(out)
    };
    let mut env = Map::new();
    for (k, v) in m["env"].as_object().into_iter().flatten() {
        // a setting naming a part that is not here (a local one not found) is left out
        if let Some(x) = v.as_str().and_then(sub) {
            env.insert(k.clone(), Value::String(x));
        }
    }
    // the card's silo as the settings the kernel loader reads: `none` the shared kernels only, another name that silo,
    // `required` no load without it
    if m["silo"] == "none" {
        env.insert("NS_SILO".into(), json!("0"));
    } else {
        if let Some(n) = m["silo"].as_str().or_else(|| m["silo"]["name"].as_str()).filter(|n| *n != s(m, "arch")) {
            env.insert("NS_SILO_NAME".into(), json!(n));
        }
        if m["silo"]["required"] == true {
            env.insert("NS_SILO_REQUIRED".into(), json!("1"));
        }
    }
    let main = ["model", "transformer", "denoiser", "lora"].iter().find_map(|r| paths.get(*r)).or_else(|| paths.values().next());
    let files: Map<String, Value> = paths.iter().map(|(r, p)| (r.clone(), Value::String(p.to_string_lossy().into_owned()))).collect();
    let mut e = json!({
        "id": m["id"], "kind": m["kind"], "arch": m["arch"], "title": m["title"],
        "file": main.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
        "files": files, "gpus": m.get("gpus").cloned().unwrap_or(json!("0")), "env": env, "enabled": true,
        "catalog": m["id"],
    });
    for k in ["ctx", "tools", "tasks", "base", "defaults", "speech", "voices", "languages", "does", "silo", "spans", "vram_gib", "vram_gib_per_card", "headroom_gib", "ram_gib", "ram_cache_gib", "streams", "sidecars"] {
        if let Some(v) = m.get(k) {
            e[k] = v.clone();
        }
    }
    e
}

/// A card's `spans` (an image, audio or 3D model whose one copy spans its cards) and `vram_gib_per_card` (its memory on
/// each, in its cards' order): spans only for those kinds and over two cards or more ('0,1', not a choice '0|1' or
/// 'all'), the per-card list only with spans and one figure a card
fn span_check(v: &Value) -> Result<(), String> {
    let spans = match v.get("spans") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("spans: true or false".into()),
    };
    let gpus = v["gpus"].as_str().unwrap_or("");
    let n = gpus.split(',').filter(|g| g.trim().parse::<usize>().is_ok()).count();
    if spans && (!matches!(s(v, "kind").as_str(), "image" | "audio" | "model3d") || gpus.contains('|') || n < 2) {
        return Err(format!("spans: an image, audio or 3D model over two cards or more (gpus '0,1'); this one: kind {}, gpus '{gpus}'", s(v, "kind")));
    }
    if let Some(per) = v.get("vram_gib_per_card") {
        let ok = per.as_array().is_some_and(|a| a.iter().all(|x| x.as_f64().is_some_and(|g| g > 0.0)));
        if !spans || !ok || per.as_array().map_or(0, Vec::len) != n {
            return Err(format!("vram_gib_per_card: with spans: true, one figure (GiB) for each of its {n} cards"));
        }
    }
    Ok(())
}

/// A made-locally part's path: `NS_<ROLE>` style settings in nextsycl.conf are the engine's own (the MTP layer:
/// NS_QW_MTP), else what another registry entry uses for the same setting
fn find_local(cfg: &Config, installed: &[Value], m: &Value, role: &str) -> Option<String> {
    let setting = m["env"].as_object()?.iter().find(|(_, v)| v.as_str() == Some(&format!("{{{role}}}")))?.0.clone();
    if let Some(v) = cfg.get(&setting).filter(|v| Path::new(v).exists()) {
        return Some(v);
    }
    installed.iter().filter_map(|x| x["env"][&setting].as_str()).find(|p| Path::new(p).exists()).map(str::to_string)
}

/// `nextsycl models pull <id>... [--dir DIR] [--from DIR]... [--verify]`: each model's missing files downloaded (or
/// linked), checked, and the model registered. Returns the entries to add.
pub fn pull(cfg: &Config, installed: &[Value], ids: &[String], raw: &[String]) -> Result<Vec<Value>, String> {
    let cat = Catalog::load(cfg)?;
    let opt = |k: &str| raw.iter().position(|a| a == k).and_then(|i| raw.get(i + 1)).cloned();
    let root = PathBuf::from(opt("--dir").or_else(|| cfg.get("NS_MODELS")).ok_or("set NS_MODELS (where models go) or pass --dir DIR")?);
    let mut from: Vec<PathBuf> = raw.iter().enumerate().filter(|(_, a)| *a == "--from").filter_map(|(i, _)| raw.get(i + 1)).map(PathBuf::from).collect();
    // and where the registered models' files already are (a model pulled again in another form - its files are
    // linked from there, not downloaded or copied again)
    for e in installed {
        for p in e["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()).chain(e["file"].as_str()) {
            if let Some(d) = Path::new(p).parent().filter(|d| d.is_dir() && !from.iter().any(|f| f == d)) {
                from.push(d.to_path_buf());
            }
        }
    }
    let verify = raw.iter().any(|a| a == "--verify");
    let mut out = Vec::new();
    for id in ids {
        let m = cat.model(id).ok_or_else(|| format!("{id}: not in the catalog (nextsycl models search)"))?;
        if installed.iter().any(|x| x["id"] == id.as_str()) && !raw.iter().any(|a| a == "--again") {
            println!("{id}: installed already (nextsycl models list; --again pulls it anew)");
            continue;
        }
        println!("{id}: {} ({:.1} GiB) into {}", s(m, "title"), gib(cat.bytes(m)), root.display());
        let mut paths = BTreeMap::new();
        for (role, key, f) in cat.files_of(m) {
            paths.insert(role.to_string(), fetch(&cat, &root, key, f, &from, verify)?);
        }
        let mut local = BTreeMap::new();
        for (role, about) in m["local"].as_object().into_iter().flatten() {
            match find_local(cfg, installed, m, role) {
                Some(p) => {
                    println!("  {role}: {p}");
                    local.insert(role.clone(), p);
                }
                None => println!("  {role}: not found - {}", about.as_str().unwrap_or("")),
            }
        }
        out.push(entry(m, &paths, &local));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_cards() -> Catalog {
        Catalog::from_dir(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../model-cards")).unwrap()
    }

    #[test]
    fn the_model_cards_are_whole() {
        let cat = repo_cards();
        for (k, f) in &cat.files {
            // a file published in byte ranges: each part a direct link with its size and SHA-256, the sizes the file's
            if let Some(ps) = f["parts"].as_array() {
                for p in ps {
                    assert!(s(p, "url").starts_with("https://") && s(p, "sha256").len() == 64 && p["size"].as_u64().unwrap_or(0) > 0, "{k}: a part");
                }
                assert_eq!(ps.iter().filter_map(|p| p["size"].as_u64()).sum::<u64>(), f["size"].as_u64().unwrap_or(0), "{k}: the parts' sizes");
            }
            assert!(s(f, "url").starts_with("https://") || f["parts"].is_array(), "{k}: a direct link");
            // a gated file names the page whose terms its account must accept
            assert!(f.get("gated").is_none_or(|g| g.as_str().is_some_and(|p| p.starts_with("https://huggingface.co/"))), "{k}: gated: its Hugging Face page");
            assert_eq!(s(f, "sha256").len(), 64, "{k}: a SHA-256");
            assert!(f["size"].as_u64().unwrap_or(0) > 0, "{k}: a size");
            assert!(!s(f, "path").is_empty() && !s(f, "path").starts_with('/'), "{k}: a relative path");
        }
        for m in &cat.models {
            let id = s(m, "id");
            assert!(!s(m, "title").is_empty() && !s(m, "arch").is_empty(), "{id}: a title and an architecture");
            // a model's guide is a file beside the cards (loras have none: they ride on a model)
            if s(m, "kind") != "lora" {
                assert!(cat.dir.join(s(m, "guide")).is_file(), "{id}: its guide {}", s(m, "guide"));
            }
            // a silo the card names is a kernels/<kind>/<name>/silo of the repository
            if let Some(n) = m["silo"].as_str().or_else(|| m["silo"]["name"].as_str()).filter(|n| *n != "none") {
                let kind = if s(m, "kind") == "lora" { "image".to_string() } else { s(m, "kind") };
                assert!(cat.dir.join(format!("../kernels/{kind}/{n}/silo")).is_dir(), "{id}: no silo kernels/{kind}/{n}/silo");
            }
            // every {role} in a setting is a file or a local part of the model
            for v in m["env"].as_object().into_iter().flatten().filter_map(|(_, v)| v.as_str()) {
                if let (Some(a), Some(b)) = (v.find('{'), v.find('}')) {
                    let role = &v[a + 1..b];
                    assert!(m["files"].get(role).is_some() || m["local"].get(role).is_some(), "{id}: {{{role}}}");
                }
            }
        }
    }

    /// Every picture-making image card states its request rules (controls, schema 1: what the platform's one image
    /// control may send it), whole and ordered; no other card has them (nextcuda's test of the same name)
    #[test]
    fn every_picture_making_card_states_its_request_rules() {
        let cat = repo_cards();
        let mut makers = 0;
        for m in &cat.models {
            let id = s(m, "id");
            if crate::controls::makes_pictures(m) {
                makers += 1;
                let c = m.get("controls").unwrap_or_else(|| panic!("{id}: a picture-making card states its controls (model-cards/README.md, Image controls)"));
                crate::controls::check(c).unwrap_or_else(|e| panic!("{id}: {e}"));
                assert_eq!(c["version"], 1, "{id}");
                // the card's own defaults are within its rules
                let d = &m["defaults"];
                let st = d["steps"].as_u64().unwrap();
                assert_eq!(c["steps"]["default"].as_u64(), Some(st), "{id}: controls.steps.default is the card's defaults.steps");
                if let Some(sz) = d["size"].as_str() {
                    assert_eq!(c["size"]["default"].as_str(), Some(sz), "{id}: controls.size.default is the card's defaults.size");
                }
                // no guidance: the card's cfg is 1; guidance: its default within the range
                match (&c["cfg"], d["cfg"].as_f64()) {
                    (Value::Null, Some(x)) => assert_eq!(x, 1.0, "{id}: cfg null, defaults.cfg 1"),
                    (r, Some(x)) if !r.is_null() => assert_eq!(r["default"].as_f64(), Some(x), "{id}: controls.cfg.default is the card's defaults.cfg"),
                    _ => {}
                }
                if let Some(smp) = d["sampler"].as_str() {
                    if !c["sampler"].is_null() {
                        assert_eq!(c["sampler"]["default"].as_str(), Some(smp), "{id}: controls.sampler.default is the card's");
                    }
                }
            } else {
                assert!(m.get("controls").is_none(), "{id}: makes no pictures, states no controls");
            }
        }
        assert!(makers >= 20, "the picture-making cards were not found ({makers})");
    }

    #[test]
    fn a_files_tensors_are_a_prefix_or_patterns() {
        assert_eq!(tensor_patterns(&json!("visual.")), Some(vec!["visual.".to_string()]));
        let p = tensor_patterns(&json!(["time_embedder.", "blocks.*.adaln_proj.", "final_layer.adaln_proj."])).unwrap();
        let keep = |n: &str| p.iter().any(|x| tensor_match(x, n));
        assert!(keep("blocks.12.adaln_proj.linear.weight") && keep("time_embedder.proj_in.bias") && keep("final_layer.adaln_proj.linear.bias"));
        assert!(!keep("blocks.12.attn.qkv_proj.weight") && !keep("final_layer.norm.weight") && !keep("token_refiner.blocks.0.adaln_proj.x"));
        assert!(tensor_patterns(&json!([])).is_none() && tensor_patterns(&json!(null)).is_none());
    }

    #[test]
    fn a_file_in_parts_is_joined_after_each_is_checked() {
        let dir = std::env::temp_dir().join(format!("ns-parts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut pieces = Vec::new();
        for (i, body) in [b"hello ".as_slice(), b"world".as_slice()].iter().enumerate() {
            let p = dir.join(format!("src{i}"));
            std::fs::write(&p, body).unwrap();
            pieces.push(json!({"url": format!("file://{}", p.display()), "size": body.len(), "sha256": sha256_of(&p).unwrap()}));
        }
        let (target, out) = (dir.join("whole.bin"), dir.join("whole.bin.part"));
        join_parts(&pieces, &target, &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"hello world");
        assert!(!dir.join("whole.bin.piece01").exists(), "the parts removed once joined");
        pieces[1]["sha256"] = json!("0".repeat(64));
        assert!(join_parts(&pieces, &target, &out).is_err(), "a part that does not match its SHA-256");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_header_is_sorted_at_every_level() {
        let v = serde_json::json!({"z": 1, "b": {"y": [ {"q": 1, "a": 2} ], "c": 0}, "a": null});
        assert_eq!(serde_json::to_string(&sorted(v)).unwrap(), r#"{"a":null,"b":{"c":0,"y":[{"a":2,"q":1}]},"z":1}"#);
    }

    #[test]
    fn a_models_file_list_is_what_pull_lays_out() {
        let cat = repo_cards();
        for m in &cat.models {
            let list = cat.file_list(m);
            // each distinct file once, its bytes the model's download, its path the one fetch writes under the folder
            assert_eq!(list.iter().map(|f| f.1).sum::<u64>(), cat.bytes(m), "{}", s(m, "id"));
            for (_, key, f) in cat.files_of(m) {
                assert!(list.iter().any(|(p, n)| *p == s(f, "path") && Some(*n) == f["size"].as_u64()), "{}: {key}", s(m, "id"));
            }
        }
    }

    #[test]
    fn a_guide_falls_back_to_its_architecture_then_its_kind() {
        let cat = repo_cards();
        let tts = cat.guide(&json!({"id": "mine", "kind": "audio", "arch": "qwen3-tts"})).unwrap();
        assert_eq!(tts, std::fs::read_to_string(cat.dir.join("guides/qwen3-tts.md")).unwrap());
        let chat = cat.guide(&json!({"id": "mine", "kind": "llm", "arch": "new"})).unwrap();
        assert_eq!(chat, std::fs::read_to_string(cat.dir.join("guides/chat.md")).unwrap());
    }

    #[test]
    fn a_cards_silo_becomes_the_loaders_settings() {
        let e = |silo: Value| entry(&json!({"id": "x", "kind": "llm", "arch": "a", "title": "X", "silo": silo}), &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(e(json!({"name": "a", "required": true}))["env"], json!({"NS_SILO_REQUIRED": "1"}));
        assert_eq!(e(json!("b"))["env"], json!({"NS_SILO_NAME": "b"}));
        assert_eq!(e(json!("none"))["env"], json!({"NS_SILO": "0"}));
        assert_eq!(e(Value::Null)["env"], json!({}));
    }

    #[test]
    fn a_spanning_card_is_checked_and_its_entry_carries_its_shares() {
        let card = json!({"id": "x", "kind": "audio", "arch": "a", "title": "X", "gpus": "0,1", "spans": true, "vram_gib": 23.0, "vram_gib_per_card": [23.0, 5.5]});
        assert!(span_check(&card).is_ok());
        let e = entry(&card, &BTreeMap::new(), &BTreeMap::new());
        assert_eq!((e["spans"].clone(), e["vram_gib_per_card"].clone(), e["gpus"].clone()), (json!(true), json!([23.0, 5.5]), json!("0,1")));
        let with = |k: &str, v: Value| { let mut c = card.clone(); c[k] = v; span_check(&c) };
        assert!(with("gpus", json!("0")).is_err(), "one card is no span");
        assert!(with("gpus", json!("0|1")).is_err(), "a choice is no span");
        assert!(with("kind", json!("llm")).is_err(), "a chat model spreads by its gpus alone");
        assert!(with("vram_gib_per_card", json!([23.0])).is_err(), "a figure a card");
        assert!(with("spans", json!(false)).is_err(), "shares without spans");
        assert!(with("spans", json!("yes")).is_err());
        let mut plain = card.clone();
        plain.as_object_mut().unwrap().retain(|k, _| k != "spans" && k != "vram_gib_per_card");
        assert!(span_check(&plain).is_ok());
    }

    #[test]
    fn an_entry_takes_the_files_and_drops_missing_local_parts() {
        let m = json!({"id": "x", "kind": "llm", "arch": "a", "title": "X", "gpus": "0", "ctx": "4096",
                       "env": {"A": "{model}", "B": "{mtp}", "C": "plain"}});
        let paths = BTreeMap::from([("model".to_string(), PathBuf::from("/m/x.gguf"))]);
        let e = entry(&m, &paths, &BTreeMap::new());
        assert_eq!(e["file"], "/m/x.gguf");
        assert_eq!(e["env"], json!({"A": "/m/x.gguf", "C": "plain"}));
        let e = entry(&m, &paths, &BTreeMap::from([("mtp".to_string(), "/d/rt".to_string())]));
        assert_eq!(e["env"]["B"], "/d/rt");
    }

    #[test]
    fn a_cards_profile_is_read_absent_is_fine_and_bad_values_are_refused_by_file() {
        let d = std::env::temp_dir().join(format!("nextsycl-cards-profile-{}", std::process::id()));
        std::fs::create_dir_all(d.join("image")).unwrap();
        std::fs::write(d.join("image/plain.yml"), "id: plain\nkind: image\n").unwrap();
        std::fs::write(d.join("image/full.yml"), "id: full\nkind: image\nconcurrency:\n  max_copies_per_card: 2\n  recommended_copies: 1\n  scaling: {\"1\": 1.0, \"2\": 1.18}\ncoexistence:\n  mode: shared\n  shared_with: [\"kind:audio\"]\n").unwrap();
        assert_eq!(Catalog::from_dir(&d).unwrap().models.len(), 2);
        for bad in ["concurrency:\n  max_copies_per_card: 0\n", "coexistence:\n  mode: alone\n", "coexistence:\n  mode: exclusive\n"] {
            std::fs::write(d.join("image/bad.yml"), format!("id: bad\nkind: image\n{bad}")).unwrap();
            let e = Catalog::from_dir(&d).err().unwrap();
            assert!(e.contains("bad.yml"), "{e}");
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

}
