//! A request's shape: `method|model|` the parameters that drive its size, bucketed. One rule per method in a registry
//! ([`Shapes`]); a method with no rule, or a request without the parameters, is `method|model` alone. A host adds its own
//! methods with [`Shapes::with`].

use std::collections::BTreeMap;

use serde_json::Value;

/// A rule: the request's body -> (name, bucketed value) pairs, in order
pub type ShapeFn = fn(&Value) -> Vec<(&'static str, i64)>;

/// `n` rounded to a multiple of `step` (at least `step`)
fn near(n: f64, step: f64) -> i64 {
    ((n / step).round().max(1.0) * step) as i64
}

/// A size in powers of two: 0, then the smallest power of two at or above n
fn pow2(n: f64) -> i64 {
    if n <= 0.0 {
        0
    } else {
        (n.ceil() as u64).next_power_of_two() as i64
    }
}

fn num(p: &Value, k: &str) -> Option<f64> {
    p[k].as_f64().or_else(|| p[k].as_str().and_then(|s| s.parse().ok()))
}

fn text_len(p: &Value, k: &str) -> f64 {
    p[k].as_str().map_or(0.0, |s| s.chars().count() as f64)
}

fn image(p: &Value) -> Vec<(&'static str, i64)> {
    let mut v = Vec::new();
    v.extend(num(p, "width").map(|x| ("w", near(x, 64.0))));
    v.extend(num(p, "height").map(|x| ("h", near(x, 64.0))));
    v.extend(num(p, "steps").map(|x| ("steps", near(x, 4.0))));
    v
}

fn upscale(p: &Value) -> Vec<(&'static str, i64)> {
    num(p, "scale").map(|x| ("scale", x as i64)).into_iter().collect()
}

fn video(p: &Value) -> Vec<(&'static str, i64)> {
    let mut v = Vec::new();
    v.extend(num(p, "width").map(|x| ("w", near(x, 64.0))));
    v.extend(num(p, "height").map(|x| ("h", near(x, 64.0))));
    v.extend(num(p, "frames").or_else(|| num(p, "seconds").map(|s| s * 24.0)).map(|x| ("frames", near(x, 8.0))));
    v.extend(num(p, "steps").map(|x| ("steps", near(x, 2.0))));
    v
}

fn speech(p: &Value) -> Vec<(&'static str, i64)> {
    vec![("text", pow2(text_len(p, "text").max(text_len(p, "input"))))]
}

fn song(p: &Value) -> Vec<(&'static str, i64)> {
    let mut v: Vec<(&'static str, i64)> = num(p, "duration").or_else(|| num(p, "seconds")).map(|x| ("secs", near(x, 10.0))).into_iter().collect();
    v.push(("lyrics", pow2(text_len(p, "lyrics"))));
    v
}

fn transcribe(p: &Value) -> Vec<(&'static str, i64)> {
    num(p, "seconds").map(|x| ("secs", near(x, 30.0))).into_iter().collect()
}

fn chat(p: &Value) -> Vec<(&'static str, i64)> {
    let chars: usize = p["messages"].as_array().into_iter().flatten().map(|m| m["content"].as_str().map_or(0, |c| c.chars().count())).sum();
    let mut v = vec![("prompt", pow2(chars as f64 / 4.0))];
    v.extend(num(p, "max_tokens").map(|x| ("max", pow2(x))));
    v
}

fn model3d(p: &Value) -> Vec<(&'static str, i64)> {
    let mut v = Vec::new();
    v.extend(num(p, "steps").map(|x| ("steps", near(x, 4.0))));
    v.extend(num(p, "resolution").map(|x| ("res", x as i64)));
    v.extend(num(p, "faces").map(|x| ("faces", near(x, 10000.0))));
    v
}

/// The rules by method
#[derive(Clone)]
pub struct Shapes {
    rules: BTreeMap<String, ShapeFn>,
}

impl Shapes {
    /// The rules nextcuda has: pictures, clips, speech, songs, recognition, chat, 3D
    pub fn standard() -> Shapes {
        let mut s = Shapes { rules: BTreeMap::new() };
        for m in ["image.generate", "image.edit"] {
            s = s.with(m, image);
        }
        for m in ["image.upscale", "video.upscale"] {
            s = s.with(m, upscale);
        }
        s.with("video.generate", video).with("audio.speak", speech).with("audio.generate", song).with("audio.transcribe", transcribe)
            .with("llm.chat", chat).with("model3d.generate", model3d).with("model3d.rig", model3d)
    }

    /// A rule for `method` (replacing one)
    pub fn with(mut self, method: &str, f: ShapeFn) -> Shapes {
        self.rules.insert(method.to_string(), f);
        self
    }

    /// The shape of a request: `method|model|k=v,...` (or `method|model` with no rule or no parameters). `video.generate`
    /// adds the `engine` it names.
    pub fn shape(&self, method: &str, model: &str, p: &Value) -> String {
        let mut k: Vec<String> = self.rules.get(method).map(|f| f(p)).unwrap_or_default().into_iter().map(|(n, v)| format!("{n}={v}")).collect();
        if method == "video.generate" {
            if let Some(e) = p["engine"].as_str() {
                k.push(format!("engine={e}"));
            }
        }
        if k.is_empty() { format!("{method}|{model}") } else { format!("{method}|{model}|{}", k.join(",")) }
    }
}

/// The shape under the standard rules
pub fn shape_of(method: &str, model: &str, p: &Value) -> String {
    Shapes::standard().shape(method, model, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_shape_buckets_the_size_driving_parameters() {
        let a = shape_of("image.generate", "q", &json!({"width": 1020, "height": 1024, "steps": 31, "prompt": "a fox"}));
        let b = shape_of("image.generate", "q", &json!({"width": 1024, "height": 1000, "steps": 30, "prompt": "another"}));
        assert_eq!(a, "image.generate|q|w=1024,h=1024,steps=32");
        assert_eq!(a, b, "the same bucket whatever the prompt");
        assert_eq!(shape_of("video.generate", "ltx2", &json!({"width": 768, "height": 512, "seconds": 5, "steps": 8, "engine": "ltx2"})),
                   "video.generate|ltx2|w=768,h=512,frames=120,steps=8,engine=ltx2");
        assert_eq!(shape_of("audio.speak", "t", &json!({"text": "x".repeat(300), "voice": "a"})), shape_of("audio.speak", "t", &json!({"text": "y".repeat(400), "voice": "b"})));
        assert_ne!(shape_of("audio.speak", "t", &json!({"text": "x".repeat(300)})), shape_of("audio.speak", "t", &json!({"text": "x".repeat(900)})));
        assert_eq!(shape_of("llm.chat", "c", &json!({"messages": [{"role": "user", "content": "x".repeat(4000)}], "max_tokens": 700})), "llm.chat|c|prompt=1024,max=1024");
        assert_eq!(shape_of("model3d.generate", "t", &json!({"steps": 30, "resolution": 512})), "model3d.generate|t|steps=32,res=512");
        assert_eq!(shape_of("audio.generate", "m", &json!({"duration": 42, "lyrics": "la ".repeat(50)})), "audio.generate|m|secs=40,lyrics=256");
        assert_eq!(shape_of("audio.transcribe", "w", &json!({"seconds": 100})), "audio.transcribe|w|secs=90");
    }

    #[test]
    fn unknown_parameters_and_methods_fall_back_to_method_and_model() {
        assert_eq!(shape_of("image.generate", "q", &json!({"prompt": "x"})), "image.generate|q");
        assert_eq!(shape_of("audio.voices.design", "v", &json!({"width": 5})), "audio.voices.design|v");
    }

    #[test]
    fn a_host_adds_its_own_method() {
        fn rule(p: &Value) -> Vec<(&'static str, i64)> {
            vec![("n", p["n"].as_i64().unwrap_or(0))]
        }
        let s = Shapes::standard().with("x.run", rule);
        assert_eq!(s.shape("x.run", "m", &json!({"n": 3})), "x.run|m|n=3");
    }
}
