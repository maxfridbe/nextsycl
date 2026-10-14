//! What the model writes, as the OCR page and the other engines write it: tables from OTSL (the model's cell tokens)
//! to HTML, spotted text from its four corners (`<|LOC_n|>`, thousandths of the picture) to Unlimited-OCR's boxes
//! (`<|det|>text [x1, y1, x2, y2]<|/det|>` then the text), and a crop's boxes moved to its page.

/// OTSL to an HTML table: `<fcel>` a cell with text, `<ecel>` an empty one, `<lcel>` the cell to the left spans here,
/// `<ucel>` the one above, `<xcel>` both, `<nl>` the row's end. Text with no cell tokens comes back as it was.
pub fn otsl_to_html(s: &str) -> String {
    if !s.contains("<fcel>") && !s.contains("<ecel>") {
        return s.to_string();
    }
    // the grid: each row's cells (kind, text)
    let mut rows: Vec<Vec<(&str, String)>> = vec![Vec::new()];
    let mut rest = s;
    while let Some(at) = rest.find('<') {
        let tail = &rest[at..];
        let Some(end) = tail.find('>') else { break };
        let tag = &tail[1..end];
        let after = &tail[end + 1..];
        match tag {
            "fcel" | "ecel" | "lcel" | "ucel" | "xcel" => {
                // a cell's text: up to the next tag
                let next = after.find('<').unwrap_or(after.len());
                rows.last_mut().unwrap().push((tag, after[..next].trim().to_string()));
                rest = &after[next..];
            }
            "nl" => {
                rows.push(Vec::new());
                rest = after;
            }
            _ => {
                // not a cell token: part of the text of the cell before it
                if let Some(c) = rows.last_mut().unwrap().last_mut() {
                    c.1.push_str(&tail[..end + 1]);
                }
                rest = after;
            }
        }
    }
    rows.retain(|r| !r.is_empty());
    let w = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let at = |r: usize, c: usize| rows.get(r).and_then(|row| row.get(c)).map(|x| x.0);
    let mut html = String::from("<table>");
    for (r, row) in rows.iter().enumerate() {
        html.push_str("<tr>");
        for (c, (kind, text)) in row.iter().enumerate() {
            if *kind != "fcel" && *kind != "ecel" {
                continue;
            }
            let mut cs = 1;
            while c + cs < w && matches!(at(r, c + cs), Some("lcel") | Some("xcel")) {
                cs += 1;
            }
            let mut rs = 1;
            while matches!(at(r + rs, c), Some("ucel") | Some("xcel")) {
                rs += 1;
            }
            html.push_str("<td");
            if cs > 1 {
                html.push_str(&format!(" colspan=\"{cs}\""));
            }
            if rs > 1 {
                html.push_str(&format!(" rowspan=\"{rs}\""));
            }
            html.push('>');
            html.push_str(text);
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

/// Spotted lines (text then eight `<|LOC_n|>`: four corners) to boxed blocks
pub fn spots_to_det(s: &str) -> String {
    if !s.contains("<|LOC_") {
        return s.to_string();
    }
    let mut out = String::new();
    for line in s.lines() {
        let Some(at) = line.find("<|LOC_") else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let text = line[..at].trim();
        let v: Vec<u32> = line[at..].split("<|LOC_").filter_map(|p| p.split("|>").next()?.parse().ok()).collect();
        if v.len() < 8 || text.is_empty() {
            out.push_str(text);
            out.push('\n');
            continue;
        }
        let xs = [v[0], v[2], v[4], v[6]];
        let ys = [v[1], v[3], v[5], v[7]];
        let c = |x: u32| x.min(999);
        out.push_str(&format!("<|det|>text [{}, {}, {}, {}]<|/det|>{text}\n", c(*xs.iter().min().unwrap()), c(*ys.iter().min().unwrap()),
                              c(*xs.iter().max().unwrap()), c(*ys.iter().max().unwrap())));
    }
    out
}

/// A crop's boxes moved to its page: [x1, y1, x2, y2] in thousandths of the crop (at `x0`, `y0`, `cw` x `ch` pixels of
/// a `w` x `h` page) to thousandths of the page
pub fn to_page(text: &str, (x0, y0, cw, ch): (u32, u32, u32, u32), (w, h): (u32, u32)) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("<|det|>") {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let Some(end) = tail.find("<|/det|>") else {
            out.push_str(tail);
            return out;
        };
        let inner = &tail["<|det|>".len()..end];
        let moved = inner.split_once('[').and_then(|(label, b)| {
            let n: Vec<f64> = b.trim_end_matches(']').split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let [a, b, c, d] = n[..] else { return None };
            let px = |v: f64, o: u32, s: u32, side: u32| ((o as f64 + v / 999.0 * s as f64) / side as f64 * 999.0).round().clamp(0.0, 999.0) as u32;
            Some(format!("{label}[{}, {}, {}, {}]", px(a, x0, cw, w), px(b, y0, ch, h), px(c, x0, cw, w), px(d, y0, ch, h)))
        });
        out.push_str("<|det|>");
        out.push_str(moved.as_deref().unwrap_or(inner));
        out.push_str("<|/det|>");
        rest = &tail[end + "<|/det|>".len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsl_spans() {
        let t = otsl_to_html("<fcel>A<lcel><nl><fcel>b<fcel>c<nl><ucel><ecel><nl>");
        assert_eq!(t, "<table><tr><td colspan=\"2\">A</td></tr><tr><td rowspan=\"2\">b</td><td>c</td></tr><tr><td></td></tr></table>");
        assert_eq!(otsl_to_html("plain"), "plain");
    }

    #[test]
    fn spots_boxed() {
        let s = spots_to_det("Total<|LOC_50|><|LOC_77|><|LOC_338|><|LOC_77|><|LOC_338|><|LOC_832|><|LOC_50|><|LOC_832|>\n");
        assert_eq!(s, "<|det|>text [50, 77, 338, 832]<|/det|>Total\n");
    }

    #[test]
    fn crop_boxes_move_to_the_page() {
        let t = to_page("<|det|>text [0, 0, 999, 999]<|/det|>hello", (500, 0, 500, 1000), (1000, 1000));
        assert_eq!(t, "<|det|>text [500, 0, 999, 999]<|/det|>hello");
    }
}
