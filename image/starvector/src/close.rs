//! Finishing an SVG the length limit stopped short. The model writes the SVG left to right; at max_length it is cut
//! wherever the token ran out - inside a path's data, an attribute's value, a tag's name, a nest of groups. `finish`
//! makes a complete document of it: the last whole piece of what was written is kept, the partial token or attribute
//! after it dropped (path data and `points` cut back to their last whole command or point, a `style` back to its last
//! whole declaration), every element still open closed in order, `</svg>` last - and the result is checked to parse as
//! XML (roxmltree: names, nesting, quotes, entities, namespace prefixes). A text that does not parse even so (the
//! model wrote a stray `</g>`, an undeclared prefix) is repaired once more (the usual `xlink` declaration) and
//! failing that cut back to its root as an empty drawing, which parses. A text that already parses and ends with
//! `</svg>` is returned untouched.

/// What `finish` made
#[derive(Clone, Debug, PartialEq)]
pub struct Finished {
    pub svg: String,
    /// the text was changed (cut, closed or replaced)
    pub repaired: bool,
    /// the text parses as XML (always true unless the fallback itself failed)
    pub valid: bool,
}

const XMLNS: &str = "http://www.w3.org/2000/svg";
const XLINK: &str = "http://www.w3.org/1999/xlink";

/// `svg` as the model wrote it (from "<svg", cut anywhere) as a complete, parsing SVG
pub fn finish(svg: &str) -> Finished {
    if svg.trim_end().ends_with("</svg>") && parses(svg) {
        return Finished { svg: svg.to_string(), repaired: false, valid: true };
    }
    let mut out = close(svg);
    if !parses(&out) && out.contains("xlink:") && !out.contains("xmlns:xlink") {
        out = out.replacen("<svg", &format!("<svg xmlns:xlink=\"{XLINK}\""), 1);
    }
    if !parses(&out) {
        out = skeleton(&out);
    }
    let valid = parses(&out);
    Finished { repaired: out != svg, svg: out, valid }
}

/// Whether `s` is well-formed XML with an svg root
pub fn parses(s: &str) -> bool {
    roxmltree::Document::parse(s).is_ok_and(|d| d.root_element().tag_name().name() == "svg")
}

/// An empty drawing with the root's own safe attributes (size, view box) - the last resort
fn skeleton(s: &str) -> String {
    let mut out = format!("<svg xmlns=\"{XMLNS}\"");
    for (n, v) in scan(s).root_attrs {
        if matches!(n.as_str(), "width" | "height" | "viewBox") && !v.contains(['"', '<', '&']) {
            out.push_str(&format!(" {n}=\"{v}\""));
        }
    }
    out.push_str("></svg>");
    out
}

/// What the scan of a (possibly cut) text found
struct Scan {
    /// the text kept: everything up to the last whole piece (a tag, a comment, text between them)
    keep: usize,
    /// the elements open at `keep`, outermost first
    stack: Vec<String>,
    /// a start tag cut inside: its start, name and the attributes whole before the cut, and the one cut
    partial: Option<Partial>,
    /// the root's whole attributes (name, value)
    root_attrs: Vec<(String, String)>,
    /// the text went wrong (a stray end tag): it is cut where that is
    broken: bool,
}

struct Partial {
    start: usize,
    name: String,
    /// the text of the tag up to the end of its last whole attribute
    whole_to: usize,
    cut: Option<(String, String)>,
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')
}

fn scan(s: &str) -> Scan {
    let b = s.as_bytes();
    let mut sc = Scan { keep: 0, stack: Vec::new(), partial: None, root_attrs: Vec::new(), broken: false };
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'<' {
            // text up to the next tag (trailing text with no tag after it is a partial token: not kept)
            match s[i..].find('<') {
                Some(n) => i += n,
                None => break,
            }
            continue;
        }
        let rest = &s[i..];
        // comments, CDATA, processing instructions, declarations: whole or dropped
        let skip = |end: &str, from: usize| s[i + from..].find(end).map(|n| i + from + n + end.len());
        let special = if rest.starts_with("<!--") {
            Some(skip("-->", 4))
        } else if rest.starts_with("<![CDATA[") {
            Some(skip("]]>", 9))
        } else if rest.starts_with("<?") {
            Some(skip("?>", 2))
        } else if rest.starts_with("<!") {
            Some(skip(">", 2))
        } else {
            None
        };
        if let Some(end) = special {
            match end {
                Some(e) => {
                    i = e;
                    sc.keep = i;
                    continue;
                }
                None => break,
            }
        }
        if rest.starts_with("</") {
            let Some(n) = rest.find('>') else { break };
            let name = rest[2..n].trim();
            if sc.stack.last().map(String::as_str) != Some(name) {
                sc.broken = true;
                break;
            }
            sc.stack.pop();
            i += n + 1;
            sc.keep = i;
            if sc.stack.is_empty() {
                break; // the root is closed: whatever follows is not the document's
            }
            continue;
        }
        // a start tag
        let start = i;
        let name_end = rest[1..].find(|c: char| !is_name_char(c)).map(|n| n + 1).unwrap_or(rest.len());
        let name = rest[1..name_end].to_string();
        if name_end >= rest.len() || name.is_empty() {
            // cut inside the name (or "<" alone, or a "<" that starts nothing)
            if name_end < rest.len() {
                sc.broken = true;
            }
            break;
        }
        let mut p = name_end;
        let mut whole_to = i + p;
        let mut attrs: Vec<(String, String)> = Vec::new();
        let mut cut: Option<(String, String)> = None;
        let mut done: Option<(usize, bool)> = None; // (end of the tag, self-closing)
        'tag: loop {
            let skip_ws = |p: usize| p + rest[p..].find(|c: char| !c.is_whitespace()).unwrap_or(rest.len() - p);
            p = skip_ws(p);
            if p >= rest.len() {
                break;
            }
            if rest[p..].starts_with('>') {
                done = Some((p + 1, false));
                break;
            }
            if rest[p..].starts_with("/>") {
                done = Some((p + 2, true));
                break;
            }
            if rest[p..].starts_with('/') && p + 1 >= rest.len() {
                break; // "/" then the cut
            }
            let a_end = rest[p..].find(|c: char| !is_name_char(c)).map(|n| n + p).unwrap_or(rest.len());
            if a_end == p {
                sc.broken = true;
                break;
            }
            let aname = rest[p..a_end].to_string();
            let q = skip_ws(a_end);
            if q >= rest.len() || !rest[q..].starts_with('=') {
                if q < rest.len() {
                    sc.broken = true;
                }
                break; // a name alone, or cut before its "="
            }
            let v = skip_ws(q + 1);
            let Some(quote) = rest[v..].chars().next().filter(|c| *c == '"' || *c == '\'') else {
                if v < rest.len() {
                    sc.broken = true;
                }
                break;
            };
            match rest[v + 1..].find(quote) {
                Some(n) => {
                    attrs.push((aname, rest[v + 1..v + 1 + n].to_string()));
                    p = v + 1 + n + 1;
                    whole_to = i + p;
                }
                None => {
                    cut = Some((aname, rest[v + 1..].to_string()));
                    break 'tag;
                }
            }
        }
        match done {
            Some((end, selfclosing)) => {
                if sc.stack.is_empty() && sc.root_attrs.is_empty() {
                    sc.root_attrs = attrs.clone();
                }
                if !selfclosing {
                    sc.stack.push(name);
                }
                i += end;
                sc.keep = i;
                if sc.stack.is_empty() {
                    break; // a self-closing root
                }
            }
            None => {
                if sc.stack.is_empty() && sc.root_attrs.is_empty() {
                    sc.root_attrs = attrs.clone();
                }
                if !sc.broken {
                    sc.partial = Some(Partial { start, name, whole_to, cut });
                }
                break;
            }
        }
    }
    sc
}

/// path data (or points) cut back to what is whole: a trailing number that may be unfinished dropped, then the last
/// command's arguments to a whole number of its sets
fn trim_path_data(d: &str, arity_of: fn(char) -> usize) -> String {
    // commands and their numbers, in order
    let mut cmds: Vec<(char, Vec<String>)> = Vec::new();
    let mut num = String::new();
    let flush = |cmds: &mut Vec<(char, Vec<String>)>, num: &mut String| {
        if !num.is_empty() {
            if let Some(last) = cmds.last_mut() {
                last.1.push(std::mem::take(num));
            } else {
                num.clear();
            }
        }
    };
    let mut prev = ' ';
    for c in d.chars() {
        if c.is_ascii_alphabetic() && !matches!(c, 'e' | 'E') {
            flush(&mut cmds, &mut num);
            cmds.push((c, Vec::new()));
        } else if c.is_ascii_digit() || (c == '.' && !num.contains('.')) || (matches!(c, 'e' | 'E') && !num.is_empty())
            || ((c == '-' || c == '+') && matches!(prev, 'e' | 'E')) {
            num.push(c);
        } else if c == '-' || c == '+' || c == '.' {
            // a sign (or a second point) starts the next number
            flush(&mut cmds, &mut num);
            num.push(c);
        } else {
            flush(&mut cmds, &mut num);
        }
        prev = c;
    }
    // the number the text ends in may be unfinished: not kept
    num.clear();
    if let Some((c, args)) = cmds.last_mut() {
        let a = arity_of(*c);
        if let Some(sets) = args.len().checked_div(a) {
            let whole = sets * a;
            args.truncate(whole);
            if whole == 0 {
                cmds.pop();
            }
        }
    }
    cmds.iter().map(|(c, a)| if a.is_empty() { c.to_string() } else { format!("{c} {}", a.join(" ")) }).collect::<Vec<_>>().join(" ")
}

fn path_arity(c: char) -> usize {
    match c.to_ascii_lowercase() {
        'm' | 'l' | 't' => 2,
        'h' | 'v' => 1,
        'c' => 6,
        's' | 'q' => 4,
        'a' => 7,
        _ => 0, // z
    }
}

fn points_arity(_: char) -> usize {
    2
}

/// A cut attribute's value made whole, or nothing when a half value would be wrong (a colour, a number, a name)
fn salvage(name: &str, value: &str) -> Option<String> {
    let v = match name {
        "d" => trim_path_data(value, path_arity),
        "points" => {
            // numbers only: pairs; reuse the path trimmer under an implicit command
            let t = trim_path_data(&format!("L{value}"), points_arity);
            t.strip_prefix("L ").unwrap_or("").to_string()
        }
        "style" => value.rfind(';').map(|n| value[..=n].to_string()).unwrap_or_default(),
        _ => String::new(),
    };
    let v = v.trim().to_string();
    (!v.is_empty() && !v.contains(['<', '&', '"'])).then_some(v)
}

fn close(svg: &str) -> String {
    let sc = scan(svg);
    let mut out = svg[..sc.keep].trim_end().to_string();
    // a style sheet cut inside keeps its whole rules (the text after the last "}"): the classes the elements will use
    if sc.partial.is_none() && sc.stack.last().map(String::as_str) == Some("style") {
        let tail = &svg[sc.keep..];
        if let Some(n) = tail.rfind('}').filter(|_| !tail.contains(['<', '&'])) {
            out.push_str(&tail[..=n]);
        }
    }
    if let Some(p) = &sc.partial {
        // the start tag cut inside: its whole attributes and the cut one if it can be made whole, closed as empty
        let salvaged = p.cut.as_ref().and_then(|(n, v)| salvage(n, v).map(|v| (n.clone(), v)));
        let head = &svg[p.start..p.whole_to];
        let has_attrs = head.len() > p.name.len() + 1 || salvaged.is_some();
        // an empty element (a bare <path>) says nothing; the root (the stack is empty) is kept with what it has
        if has_attrs || sc.stack.is_empty() {
            out.push_str(svg[sc.keep..p.start].trim_end());
            out.push_str(head);
            if let Some((n, v)) = salvaged {
                out.push_str(&format!(" {n}=\"{v}\""));
            }
            if sc.stack.is_empty() {
                out.push('>');
                out.push_str(&format!("</{}>", p.name));
                return ensure_xmlns(&out);
            }
            out.push_str("/>");
        }
    }
    if sc.stack.is_empty() && sc.partial.is_none() && !out.starts_with("<svg") {
        return format!("<svg xmlns=\"{XMLNS}\"></svg>");
    }
    for name in sc.stack.iter().rev() {
        out.push_str(&format!("</{name}>"));
    }
    ensure_xmlns(&out)
}

/// the root names the SVG namespace (a document cut before its xmlns is not an SVG file without it)
fn ensure_xmlns(s: &str) -> String {
    let end = s.find('>').unwrap_or(s.len());
    if s.starts_with("<svg") && !s[..end].contains("xmlns=") {
        return s.replacen("<svg", &format!("<svg xmlns=\"{XMLNS}\""), 1);
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(f: &Finished) {
        assert!(f.valid && parses(&f.svg), "{}", f.svg);
        assert!(f.svg.trim_end().ends_with("</svg>"), "{}", f.svg);
    }

    #[test]
    fn a_closed_svg_is_untouched() {
        let s = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\"><path d=\"M0 0L5 5z\"/></svg>";
        let f = finish(s);
        assert_eq!(f, Finished { svg: s.into(), repaired: false, valid: true });
        // trailing whitespace stays too
        assert_eq!(finish(&format!("{s}\n")).svg, format!("{s}\n"));
    }

    #[test]
    fn cut_mid_path_data_keeps_the_whole_commands() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><path fill=\"#fff\" d=\"M 10 10 L 20 20 C 30 30 40 40 50 5");
        ok(&f);
        assert!(f.repaired);
        // the cut curve (5 of 6 numbers, the last unfinished) goes; the line stays
        assert!(f.svg.contains("d=\"M 10 10 L 20 20\""), "{}", f.svg);
        assert!(f.svg.contains("fill=\"#fff\"") && f.svg.ends_with("/></svg>"), "{}", f.svg);
    }

    #[test]
    fn cut_after_a_whole_number_still_drops_an_unfinished_set() {
        // "L 20" has one of its two numbers
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M 1 2 L 20 ");
        ok(&f);
        assert!(f.svg.contains("d=\"M 1 2\""), "{}", f.svg);
    }

    #[test]
    fn compact_numbers_and_signs_split_as_svg_does() {
        assert_eq!(trim_path_data("M10-5.5.5l3,4c1 2 3 4 5 6", path_arity), "M 10 -5.5 .5 l 3 4");
        // the last number may be unfinished ("1" might be "12"): dropped; z needs no numbers
        assert_eq!(trim_path_data("M 1 2 L 3 4 z M 5 6 L 7", path_arity), "M 1 2 L 3 4 z M 5 6");
        assert_eq!(trim_path_data("M1 2L3 4z", path_arity), "M 1 2 L 3 4 z");
    }

    #[test]
    fn cut_mid_attribute_drops_it() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><circle cx=\"5\" cy=\"5\" r=\"2\" fill=\"#12a");
        ok(&f);
        assert!(f.svg.contains("<circle cx=\"5\" cy=\"5\" r=\"2\"/>"), "{}", f.svg);
        assert!(!f.svg.contains("#12a"));
        // cut right after the name, after "=", after the opening quote
        for tail in ["<rect x=\"1\" fill", "<rect x=\"1\" fill=", "<rect x=\"1\" fill=\"", "<rect x=\"1\" fill=\"#"] {
            let f = finish(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\">{tail}"));
            ok(&f);
            assert!(f.svg.contains("<rect x=\"1\"/>"), "{tail} -> {}", f.svg);
        }
    }

    #[test]
    fn cut_mid_tag_name_drops_the_tag() {
        for tail in ["<pa", "<", "<path", "</", "</g", "<!-- a comment", "<!"] {
            let f = finish(&format!("<svg xmlns=\"http://www.w3.org/2000/svg\"><g><rect width=\"1\" height=\"1\"/>{tail}"));
            ok(&f);
            assert!(f.svg.ends_with("<rect width=\"1\" height=\"1\"/></g></svg>"), "{tail} -> {}", f.svg);
        }
    }

    #[test]
    fn a_bare_cut_tag_with_no_attributes_is_dropped_but_one_with_attributes_stays_empty() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><path ");
        ok(&f);
        assert_eq!(f.svg, "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><path id=\"a\" ");
        assert!(f.svg.contains("<path id=\"a\"/>"), "{}", f.svg);
    }

    #[test]
    fn nested_groups_are_closed_in_order() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><g id=\"a\"><g id=\"b\"><defs><linearGradient id=\"g\"><stop offset=\"0\"/>");
        ok(&f);
        assert!(f.svg.ends_with("<stop offset=\"0\"/></linearGradient></defs></g></g></svg>"), "{}", f.svg);
        let d = roxmltree::Document::parse(&f.svg).unwrap();
        assert_eq!(d.descendants().filter(|n| n.is_element()).count(), 6);
    }

    #[test]
    fn style_text_and_style_attributes() {
        // a cut style attribute keeps its whole declarations
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><path style=\"fill:#fff;stroke:#00");
        ok(&f);
        assert!(f.svg.contains("style=\"fill:#fff;\""), "{}", f.svg);
        // a style element cut in its text keeps its whole rules, the partial one goes, the element is closed
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><style type=\"text/css\">.a{fill:#fff;}.b{fi");
        ok(&f);
        assert!(f.svg.ends_with("<style type=\"text/css\">.a{fill:#fff;}</style></svg>"), "{}", f.svg);
        // no whole rule: empty
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><style>.a{fi");
        ok(&f);
        assert!(f.svg.ends_with("<style></style></svg>"), "{}", f.svg);
    }

    #[test]
    fn points_are_cut_to_whole_pairs() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><polygon points=\"1,2 3,4 5,");
        ok(&f);
        assert!(f.svg.contains("points=\"1 2 3 4\""), "{}", f.svg);
    }

    #[test]
    fn a_missing_namespace_is_added_and_xlink_declared() {
        let f = finish("<svg width=\"10\" height=\"10\"><image xlink:href=\"a.png\"/>");
        ok(&f);
        assert!(f.svg.contains("xmlns=\"http://www.w3.org/2000/svg\"") && f.svg.contains("width=\"10\""), "{}", f.svg);
        assert!(f.svg.contains("xmlns:xlink"));
    }

    #[test]
    fn a_root_cut_inside_its_own_tag_or_empty_text_still_gives_an_svg() {
        let f = finish("<svg width=\"10\" height=\"10\" viewBox=\"0 0 1");
        ok(&f);
        assert!(f.svg.contains("width=\"10\"") && f.svg.contains("height=\"10\"") && !f.svg.contains("viewBox"), "{}", f.svg);
        for s in ["", "<sv", "<svg", "<svg "] {
            ok(&finish(s));
        }
    }

    #[test]
    fn a_stray_end_tag_cuts_the_text_there() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><g><rect width=\"1\" height=\"1\"/></g></g><path d=\"M0 0\"/></svg>");
        ok(&f);
        assert!(f.repaired && !f.svg.contains("<path"), "{}", f.svg);
    }

    #[test]
    fn a_closed_svg_with_open_groups_inside_is_repaired() {
        // the model wrote </svg> with a group still open: the text does not parse
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"><g><rect width=\"1\" height=\"1\"/></svg>");
        ok(&f);
        assert!(f.repaired);
    }

    #[test]
    fn a_text_after_the_root_is_dropped() {
        let f = finish("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n<!-- more -->");
        assert!(parses(&f.svg));
        assert!(!f.svg.contains("more"));
    }

    #[test]
    fn tabs_newlines_and_single_quotes_are_attributes_too() {
        let f = finish("<svg xmlns='http://www.w3.org/2000/svg'\n\tviewBox='0 0 5 5'>\n\t<path\n\t\td='M1 1 L2 2 L3");
        ok(&f);
        assert!(f.svg.contains("viewBox='0 0 5 5'"), "{}", f.svg);
        assert!(f.svg.contains("d=\"M 1 1 L 2 2\""), "{}", f.svg);
    }
}
