//! An ONNX file read for its weights, without a runtime: the protobuf's graph (its nodes' names, types, inputs,
//! outputs and integer attributes) and its initializers (float32 tensors, raw or packed). Enough to load a network
//! whose layers are written in Rust (IS-Net: the convolutions found by the module path in their outputs' names).

use std::collections::HashMap;
use std::path::Path;

use nextsycl_core::{Error, Result};

pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// integer and integer-list attributes (dilations, strides, pads, group, kernel_shape)
    pub ints: HashMap<String, Vec<i64>>,
}

pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Model {
    pub nodes: Vec<Node>,
    pub tensors: HashMap<String, Tensor>,
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for s in 0..10 {
            let c = *self.b.get(self.i).ok_or_else(|| Error("onnx: cut short in a varint".into()))?;
            self.i += 1;
            v |= ((c & 0x7f) as u64) << (7 * s);
            if c & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(Error("onnx: a varint too long".into()))
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        let s = self.b.get(self.i..self.i + n).ok_or_else(|| Error("onnx: a field past the end".into()))?;
        self.i += n;
        Ok(s)
    }
    /// the next (field, wire type), or None at the end
    fn key(&mut self) -> Result<Option<(u64, u8)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let k = self.varint()?;
        Ok(Some((k >> 3, (k & 7) as u8)))
    }
    fn skip(&mut self, wt: u8) -> Result<()> {
        match wt {
            0 => {
                self.varint()?;
            }
            1 => self.i += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.i += 4,
            w => return Err(Error(format!("onnx: wire type {w}"))),
        }
        Ok(())
    }
}

fn string(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn tensor(b: &[u8]) -> Result<(String, Tensor)> {
    let mut r = Rd { b, i: 0 };
    let (mut name, mut dims, mut dtype, mut data) = (String::new(), Vec::new(), 1u64, Vec::new());
    let mut raw: Option<&[u8]> = None;
    while let Some((f, wt)) = r.key()? {
        match (f, wt) {
            (1, 0) => dims.push(r.varint()? as usize),
            (1, 2) => {
                let mut p = Rd { b: r.bytes()?, i: 0 };
                while p.i < p.b.len() {
                    dims.push(p.varint()? as usize);
                }
            }
            (2, 0) => dtype = r.varint()?,
            (4, 2) => data.extend(r.bytes()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
            (4, 5) => {
                let c = r.b.get(r.i..r.i + 4).ok_or_else(|| Error("onnx: a float past the end".into()))?;
                data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                r.i += 4;
            }
            (8, 2) => name = string(r.bytes()?),
            (9, 2) => raw = Some(r.bytes()?),
            (_, w) => r.skip(w)?,
        }
    }
    if let Some(raw) = raw {
        if dtype != 1 {
            return Err(Error(format!("onnx: tensor {name} of type {dtype}, float32 expected")));
        }
        data = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    }
    Ok((name, Tensor { dims, data }))
}

fn attribute(b: &[u8]) -> Result<(String, Vec<i64>)> {
    let mut r = Rd { b, i: 0 };
    let (mut name, mut ints) = (String::new(), Vec::new());
    while let Some((f, wt)) = r.key()? {
        match (f, wt) {
            (1, 2) => name = string(r.bytes()?),
            (3, 0) => ints.push(r.varint()? as i64),
            (8, 0) => ints.push(r.varint()? as i64),
            (8, 2) => {
                let mut p = Rd { b: r.bytes()?, i: 0 };
                while p.i < p.b.len() {
                    ints.push(p.varint()? as i64);
                }
            }
            (_, w) => r.skip(w)?,
        }
    }
    Ok((name, ints))
}

fn node(b: &[u8]) -> Result<Node> {
    let mut r = Rd { b, i: 0 };
    let mut n = Node { op: String::new(), inputs: Vec::new(), outputs: Vec::new(), ints: HashMap::new() };
    while let Some((f, wt)) = r.key()? {
        match (f, wt) {
            (1, 2) => n.inputs.push(string(r.bytes()?)),
            (2, 2) => n.outputs.push(string(r.bytes()?)),
            (4, 2) => n.op = string(r.bytes()?),
            (5, 2) => {
                let (k, v) = attribute(r.bytes()?)?;
                n.ints.insert(k, v);
            }
            (_, w) => r.skip(w)?,
        }
    }
    Ok(n)
}

impl Model {
    pub fn open(path: &Path) -> Result<Model> {
        let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mut r = Rd { b: &b, i: 0 };
        let mut graph = None;
        while let Some((f, wt)) = r.key()? {
            if (f, wt) == (7, 2) {
                graph = Some(r.bytes()?);
            } else {
                r.skip(wt)?;
            }
        }
        let g = graph.ok_or_else(|| Error(format!("{}: no graph", path.display())))?;
        let mut r = Rd { b: g, i: 0 };
        let mut m = Model { nodes: Vec::new(), tensors: HashMap::new() };
        while let Some((f, wt)) = r.key()? {
            match (f, wt) {
                (1, 2) => m.nodes.push(node(r.bytes()?)?),
                (5, 2) => {
                    let (n, t) = tensor(r.bytes()?)?;
                    m.tensors.insert(n, t);
                }
                (_, w) => r.skip(w)?,
            }
        }
        Ok(m)
    }
}
