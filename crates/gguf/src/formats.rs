//! The GGUF block formats the SYCL kernels read, owned here: their layouts (the bytes of a block of each quantized
//! type), the size constants the kernels index by, and the i-quant codebooks (`format_tables`). `ns_gguf_header` writes
//! them as kernels/ns/ns_gguf.h, the header the engines' kernels include; a test keeps that file equal to what this
//! module writes (`NS_WRITE_GGUF_HEADER=1 cargo test -p nextsycl-gguf ns_gguf_header` rewrites it).
//!
//! The layouts are the file format's: a block's fields in their order and widths, as GGUF files store them. Field and
//! type names are the ones the kernels use (`block_q8_1`, `ds`, `QK_K`, `QI6_K` ...), so the kernels' code reads the
//! same against this header.

use crate::format_tables as t;

/// Values a super-block of the K- and i-quants holds
pub const QK_K: usize = 256;
/// Bytes of a Q4_K / Q5_K block's 6-bit scales and mins
pub const K_SCALE_SIZE: usize = 12;
/// Values a block of the 32-value types holds
pub const QK32: usize = 32;
/// IQ1_S: the grid's entries and the shift its signs add
pub const NGRID_IQ1S: usize = 2048;
pub const IQ1S_DELTA: f32 = 0.125;

/// One block type: its name and its fields in file order (C declarations), with the byte size its layout must have
struct Block {
    name: &'static str,
    fields: &'static str,
    bytes: usize,
}

/// A K-quant's (d, dmin) half pair: both halves, or the pair as one sycl::half2 (`dm`)
const DM: &str = "    union {\n        struct {\n            ns_half d;    /* super-block scale of the quantized scales */\n            ns_half dmin; /* super-block scale of the quantized mins */\n        } data;\n        ns_half2 dm;\n    };";

const BLOCKS: &[Block] = &[
    Block { name: "block_q4_0", fields: "    ns_half d;\n    uint8_t qs[QK4_0 / 2];   /* 4-bit values, the low nibbles values 0..15, the high 16..31 */", bytes: 2 + 16 },
    Block { name: "block_q5_0", fields: "    ns_half d;\n    uint8_t qh[4];           /* the 5th bits of the 32 values */\n    uint8_t qs[QK5_0 / 2];", bytes: 2 + 4 + 16 },
    Block { name: "block_q8_0", fields: "    ns_half d;\n    int8_t qs[QK8_0];", bytes: 2 + 32 },
    Block {
        name: "block_q8_1",
        fields: "    union {\n        struct {\n            ns_half d;           /* scale */\n            ns_half s;           /* d * the sum of qs */\n        } data;\n        ns_half2 ds;\n    };\n    int8_t qs[QK8_1];",
        bytes: 4 + 32,
    },
    Block { name: "block_mxfp4", fields: "    uint8_t e;               /* the shared E8M0 exponent */\n    uint8_t qs[QK_MXFP4 / 2];", bytes: 1 + 16 },
    Block { name: "block_q2_K", fields: "    uint8_t scales[QK_K / 16];\n    uint8_t qs[QK_K / 4];\nDM", bytes: 16 + 64 + 4 },
    Block { name: "block_q3_K", fields: "    uint8_t hmask[QK_K / 8];\n    uint8_t qs[QK_K / 4];\n    uint8_t scales[12];\n    ns_half d;", bytes: 32 + 64 + 12 + 2 },
    Block { name: "block_q4_K", fields: "DM\n    uint8_t scales[K_SCALE_SIZE];\n    uint8_t qs[QK_K / 2];", bytes: 4 + 12 + 128 },
    Block { name: "block_q5_K", fields: "DM\n    uint8_t scales[K_SCALE_SIZE];\n    uint8_t qh[QK_K / 8];\n    uint8_t qs[QK_K / 2];", bytes: 4 + 12 + 32 + 128 },
    Block { name: "block_q6_K", fields: "    uint8_t ql[QK_K / 2];\n    uint8_t qh[QK_K / 4];\n    int8_t scales[QK_K / 16];\n    ns_half d;", bytes: 128 + 64 + 16 + 2 },
    Block { name: "block_iq1_s", fields: "    ns_half d;\n    uint8_t qs[QK_K / 8];\n    uint16_t qh[QK_K / 32];", bytes: 2 + 32 + 16 },
    Block {
        name: "block_iq1_m",
        fields: "    uint8_t qs[QK_K / 8];      /* the grid indices' low 8 bits */\n    uint8_t qh[QK_K / 16];     /* a nibble a group of 8: its index's high 3 bits, its delta's sign */\n    uint8_t scales[QK_K / 32]; /* 3-bit scales a 16 values; the block's fp16 scale in the four words' top nibbles */",
        bytes: 32 + 16 + 8,
    },
    Block { name: "block_iq2_xxs", fields: "    ns_half d;\n    uint16_t qs[QK_K / 8];", bytes: 2 + 64 },
    Block { name: "block_iq2_xs", fields: "    ns_half d;\n    uint16_t qs[QK_K / 8];\n    uint8_t scales[QK_K / 32];", bytes: 2 + 64 + 8 },
    Block { name: "block_iq2_s", fields: "    ns_half d;\n    uint8_t qs[QK_K / 4];\n    uint8_t qh[QK_K / 32];\n    uint8_t scales[QK_K / 32];", bytes: 2 + 64 + 8 + 8 },
    Block { name: "block_iq3_xxs", fields: "    ns_half d;\n    uint8_t qs[3 * QK_K / 8];", bytes: 2 + 96 },
    Block {
        name: "block_iq3_s",
        fields: "    ns_half d;\n    uint8_t qs[QK_K / 4];\n    uint8_t qh[QK_K / 32];\n    uint8_t signs[QK_K / 8];\n    uint8_t scales[IQ3S_N_SCALE];",
        bytes: 2 + 64 + 8 + 32 + 4,
    },
    Block { name: "block_iq4_nl", fields: "    ns_half d;\n    uint8_t qs[QK4_NL / 2];", bytes: 2 + 16 },
    Block { name: "block_iq4_xs", fields: "    ns_half d;\n    uint16_t scales_h;\n    uint8_t scales_l[QK_K / 64];\n    uint8_t qs[QK_K / 2];", bytes: 2 + 2 + 4 + 128 },
];

/// The size macros: values a block (QK*), and for the dot products 32-bit words a block (QI*) and values a byte's
/// nibble or crumb stands for (QR*)
const MACROS: &[(&str, &str)] = &[
    ("QK_K", "256"),
    ("K_SCALE_SIZE", "12"),
    ("QK4_0", "32"),
    ("QK5_0", "32"),
    ("QK8_0", "32"),
    ("QK8_1", "32"),
    ("QK4_NL", "32"),
    ("QK_MXFP4", "32"),
    ("IQ3S_N_SCALE", "(QK_K / 64)"),
    ("NGRID_IQ1S", "2048"),
    ("IQ1S_DELTA", "0.125f"),
    ("IQ1M_DELTA", "0.125f"),
    ("QR4_0", "2"),
    ("QI4_0", "(QK4_0 / (4 * QR4_0))"),
    ("QR5_0", "2"),
    ("QI5_0", "(QK5_0 / (4 * QR5_0))"),
    ("QR8_0", "1"),
    ("QI8_0", "(QK8_0 / (4 * QR8_0))"),
    ("QR8_1", "1"),
    ("QI8_1", "(QK8_1 / (4 * QR8_1))"),
    ("QR_MXFP4", "2"),
    ("QI_MXFP4", "(QK_MXFP4 / (4 * QR_MXFP4))"),
    ("QR2_K", "4"),
    ("QI2_K", "(QK_K / (4 * QR2_K))"),
    ("QR3_K", "4"),
    ("QI3_K", "(QK_K / (4 * QR3_K))"),
    ("QR4_K", "2"),
    ("QI4_K", "(QK_K / (4 * QR4_K))"),
    ("QR5_K", "2"),
    ("QI5_K", "(QK_K / (4 * QR5_K))"),
    ("QR6_K", "2"),
    ("QI6_K", "(QK_K / (4 * QR6_K))"),
    ("QR1_S", "8"),
    ("QI1_S", "(QK_K / (4 * QR1_S))"),
    ("QR2_XXS", "4"),
    ("QI2_XXS", "(QK_K / (4 * QR2_XXS))"),
    ("QR2_XS", "4"),
    ("QI2_XS", "(QK_K / (4 * QR2_XS))"),
    ("QR2_S", "4"),
    ("QI2_S", "(QK_K / (4 * QR2_S))"),
    ("QR3_XXS", "4"),
    ("QI3_XXS", "(QK_K / (4 * QR3_XXS))"),
    ("QR3_S", "4"),
    ("QI3_S", "(QK_K / (4 * QR3_S))"),
    ("QR4_NL", "2"),
    ("QI4_NL", "(QK4_NL / (4 * QR4_NL))"),
    ("QR4_XS", "2"),
    ("QI4_XS", "(QK_K / (4 * QR4_XS))"),
];

/// A table as a C array: hex for the grids (their bytes are the lattice points), decimal for the small ones
fn table<T: Copy>(out: &mut String, c_type: &str, name: &str, v: &[T], per_line: usize, fmt: impl Fn(T) -> String) {
    out.push_str(&format!("static const {c_type} {name}[{}] = {{\n", v.len()));
    for row in v.chunks(per_line) {
        out.push_str("    ");
        out.push_str(&row.iter().map(|x| fmt(*x)).collect::<Vec<_>>().join(", "));
        out.push_str(",\n");
    }
    out.push_str("};\n\n");
}

/// kernels/ns/ns_gguf.h as this module defines it
pub fn ns_gguf_header() -> String {
    let mut h = String::new();
    h.push_str(
        "/* ns_gguf.h: the GGUF block formats the kernels read - each quantized type's block layout, the size constants the\n \
         * kernels index by, and the i-quants' codebooks. Written by crates/gguf (formats::ns_gguf_header), which owns them;\n \
         * do not edit: NS_WRITE_GGUF_HEADER=1 cargo test -p nextsycl-gguf ns_gguf_header rewrites it. The codebook values are\n \
         * the formats' own (chosen by the ggml authors, MIT): a reader of these files needs exactly these numbers. */\n\
         #ifndef NS_GGUF_H\n#define NS_GGUF_H\n\n#include <sycl/half_type.hpp>\n#include <cstdint>\n\n\
         typedef sycl::half ns_half;\ntypedef sycl::half2 ns_half2;\n\n",
    );
    for (name, v) in MACROS {
        h.push_str(&format!("#define {name} {v}\n"));
    }
    h.push('\n');
    for b in BLOCKS {
        h.push_str(&format!("typedef struct {{\n{}\n}} {};\n", b.fields.replace("DM", DM), b.name));
        h.push_str(&format!("static_assert(sizeof({0}) == {1}, \"{0}: {1} bytes\");\n\n", b.name, b.bytes));
    }
    table(&mut h, "uint8_t", "kmask_iq2xs", &t::KMASK_IQ2XS, 16, |x| x.to_string());
    table(&mut h, "uint8_t", "ksigns_iq2xs", &t::KSIGNS_IQ2XS, 16, |x| x.to_string());
    table(&mut h, "uint64_t", "iq2xxs_grid", &t::IQ2XXS_GRID, 4, |x| format!("0x{x:016x}"));
    table(&mut h, "uint64_t", "iq2xs_grid", &t::IQ2XS_GRID, 4, |x| format!("0x{x:016x}"));
    table(&mut h, "uint64_t", "iq2s_grid", &t::IQ2S_GRID, 4, |x| format!("0x{x:016x}"));
    table(&mut h, "uint32_t", "iq3xxs_grid", &t::IQ3XXS_GRID, 8, |x| format!("0x{x:08x}"));
    table(&mut h, "uint32_t", "iq3s_grid", &t::IQ3S_GRID, 8, |x| format!("0x{x:08x}"));
    table(&mut h, "int8_t", "kvalues_iq4nl", &t::KVALUES_IQ4NL, 16, |x| x.to_string());
    table(&mut h, "int8_t", "kvalues_mxfp4", &t::KVALUES_MXFP4, 16, |x| x.to_string());
    table(&mut h, "uint32_t", "iq1s_grid_gpu", &t::IQ1S_GRID_GPU, 8, |x| format!("0x{x:08x}"));
    h.push_str("#endif\n");
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn ns_gguf_header_is_what_the_formats_write() {
        let path = repo().join("kernels/ns/ns_gguf.h");
        let want = ns_gguf_header();
        if std::env::var("NS_WRITE_GGUF_HEADER").is_ok_and(|v| v == "1") {
            std::fs::write(&path, &want).unwrap();
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(have == want, "{} differs from formats::ns_gguf_header (NS_WRITE_GGUF_HEADER=1 rewrites it)", path.display());
    }

    #[test]
    fn block_sizes_are_the_formats() {
        // bytes a block of each type, as GGUF files store them (ggml's type traits)
        let want = [
            ("block_q4_0", 18), ("block_q5_0", 22), ("block_q8_0", 34), ("block_q8_1", 36), ("block_mxfp4", 17), ("block_q2_K", 84),
            ("block_q3_K", 110), ("block_q4_K", 144), ("block_q5_K", 176), ("block_q6_K", 210), ("block_iq1_s", 50), ("block_iq1_m", 56), ("block_iq2_xxs", 66),
            ("block_iq2_xs", 74), ("block_iq2_s", 82), ("block_iq3_xxs", 98), ("block_iq3_s", 110), ("block_iq4_nl", 18), ("block_iq4_xs", 136),
        ];
        for (name, bytes) in want {
            assert_eq!(BLOCKS.iter().find(|b| b.name == name).map(|b| b.bytes), Some(bytes), "{name}");
        }
        assert_eq!(QK_K, 256);
        assert_eq!(K_SCALE_SIZE, 12);
        assert_eq!(QK32, 32);
        assert_eq!((NGRID_IQ1S, IQ1S_DELTA), (t::IQ1S_GRID_GPU.len(), 0.125));
    }

    /// The tables against the values of ggml-common.h where it is still in the tree (the one-time check of the import;
    /// skipped once it is gone)
    #[test]
    fn codebooks_equal_ggml_common_h() {
        let Ok(src) = std::fs::read_to_string(repo().join("kernels/strata/third_party/ggml/ggml-common.h")) else { return };
        let values = |name: &str| -> Vec<i128> {
            let at = src.find(&format!(", {name}, ")).unwrap_or_else(|| panic!("{name} not in ggml-common.h"));
            let body = &src[at..];
            let body = &body[body.find(")\n").unwrap() + 1..body.find("GGML_TABLE_END()").unwrap()];
            body.lines()
                .map(|l| l.split("//").next().unwrap())
                .flat_map(|l| l.split(|c: char| c == ',' || c.is_whitespace()))
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
                    let v = s.strip_prefix("0x").map_or_else(|| s.parse::<i128>().unwrap(), |x| i128::from_str_radix(x, 16).unwrap());
                    if neg { -v } else { v }
                })
                .collect()
        };
        let ours: Vec<(&str, Vec<i128>)> = vec![
            ("kmask_iq2xs", t::KMASK_IQ2XS.iter().map(|&x| x as i128).collect()),
            ("ksigns_iq2xs", t::KSIGNS_IQ2XS.iter().map(|&x| x as i128).collect()),
            ("iq2xxs_grid", t::IQ2XXS_GRID.iter().map(|&x| x as i128).collect()),
            ("iq2xs_grid", t::IQ2XS_GRID.iter().map(|&x| x as i128).collect()),
            ("iq2s_grid", t::IQ2S_GRID.iter().map(|&x| x as i128).collect()),
            ("iq3xxs_grid", t::IQ3XXS_GRID.iter().map(|&x| x as i128).collect()),
            ("iq3s_grid", t::IQ3S_GRID.iter().map(|&x| x as i128).collect()),
            ("kvalues_iq4nl", t::KVALUES_IQ4NL.iter().map(|&x| x as i128).collect()),
            ("kvalues_fp4", t::KVALUES_MXFP4.iter().map(|&x| x as i128).collect()),
            ("iq1s_grid_gpu", t::IQ1S_GRID_GPU.iter().map(|&x| x as i128).collect()),
        ];
        for (name, v) in ours {
            assert_eq!(values(name), v, "{name}");
        }
    }
}
