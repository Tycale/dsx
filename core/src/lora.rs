//! Experimental inference-only rank-1 PEFT adapter for attention wo_b.
//! Kept separate from the quantized base; unsupported PEFT variants fail closed.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

#[derive(Debug)]
pub struct RankOne {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug)]
pub struct Adapter {
    pub layers: BTreeMap<usize, RankOne>,
    pub scale: f32,
}

impl Adapter {
    pub fn load(dir: &Path, n_layers: usize, rows: usize, cols: usize, scale: f32) -> Result<Self> {
        let cfg = dir.join("adapter_config.json");
        let weights = dir.join("adapter_model.safetensors");
        ensure!(std::fs::metadata(&cfg)?.len() <= 65536, "adapter config too large");
        ensure!(std::fs::metadata(&weights)?.len() <= 8 * 1024 * 1024, "rank-1 adapter too large");
        Self::parse(&serde_json::from_slice(&std::fs::read(cfg)?)?, &std::fs::read(weights)?, n_layers, rows, cols, scale)
    }

    fn parse(c: &Value, bytes: &[u8], n_layers: usize, rows: usize, cols: usize, scale: f32) -> Result<Self> {
        ensure!(scale.is_finite() && scale >= 0.0, "LoRA scale must be finite and non-negative");
        ensure!(c["peft_type"] == "LORA" && c["r"] == 1 && c["lora_alpha"] == 1, "only rank-1, alpha-1 LORA supported");
        ensure!(c["bias"] == "none" && c["inference_mode"] == true && c["fan_in_fan_out"] == false, "unsupported PEFT mode");
        ensure!(c["lora_dropout"].as_f64() == Some(0.0), "dropout unsupported");
        ensure!(c["target_modules"] == serde_json::json!(["wo_b"]), "only wo_b supported");
        for k in ["use_dora", "use_rslora", "lora_bias"] {
            ensure!(c[k].is_null() || c[k] == false, "{k} unsupported");
        }
        for k in ["rank_pattern", "alpha_pattern", "modules_to_save"] {
            ensure!(c[k].is_null() || c[k].as_object().is_some_and(|x| x.is_empty()), "{k} unsupported");
        }
        let selected = c["layers_to_transform"].as_array().context("layers_to_transform required")?;
        ensure!(!selected.is_empty(), "empty adapter");
        let head_len = u64::from_le_bytes(bytes.get(..8).context("truncated safetensors")?.try_into()?) as usize;
        ensure!(head_len <= 1024 * 1024, "safetensors header too large");
        let head_end = 8usize.checked_add(head_len).context("header overflow")?;
        let head: Value = serde_json::from_slice(bytes.get(8..head_end).context("truncated header")?)?;
        let tensors = head.as_object().context("header object")?;
        ensure!(tensors.keys().filter(|k| *k != "__metadata__").count() == selected.len() * 2, "unexpected/missing tensors");
        let data = bytes.get(head_end..).context("missing tensor data")?;
        let mut layers = BTreeMap::new();
        let mut ranges = Vec::new();
        for v in selected {
            let li = usize::try_from(v.as_u64().context("invalid layer")?)?;
            ensure!(li < n_layers && !layers.contains_key(&li), "duplicate/out-of-range layer {li}");
            let mut read = |part: &str, shape: [usize; 2], count: usize| -> Result<Vec<f32>> {
                let name = format!("base_model.model.language_model.model.layers.{li}.attn.wo_b.lora_{part}.weight");
                let t = tensors.get(&name).with_context(|| format!("missing {name}"))?;
                ensure!(t["dtype"] == "F32" && t["shape"] == serde_json::json!(shape), "invalid dtype/shape: {name}");
                let offsets = t["data_offsets"].as_array().context("offsets")?;
                ensure!(offsets.len() == 2, "invalid offsets");
                let start = usize::try_from(offsets[0].as_u64().context("start")?)?;
                let end = usize::try_from(offsets[1].as_u64().context("end")?)?;
                ensure!(end.checked_sub(start) == count.checked_mul(4), "invalid byte count: {name}");
                let raw = data.get(start..end).context("tensor out of bounds")?;
                ranges.push((start, end));
                let values: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                ensure!(values.iter().all(|v| v.is_finite()), "non-finite adapter: {name}");
                Ok(values)
            };
            let a = read("A", [1, cols], cols)?;
            let b = read("B", [rows, 1], rows)?;
            layers.insert(li, RankOne { a, b });
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (start, next) in ranges {
            ensure!(start == end, "overlap/gap in tensor data");
            end = next;
        }
        ensure!(end == data.len(), "trailing tensor data");
        Ok(Self { layers, scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Value, Vec<u8>) {
        let cfg = serde_json::json!({"peft_type":"LORA","r":1,"lora_alpha":1,"bias":"none","inference_mode":true,"fan_in_fan_out":false,"lora_dropout":0.0,"target_modules":["wo_b"],"layers_to_transform":[0]});
        let h = serde_json::json!({
            "base_model.model.language_model.model.layers.0.attn.wo_b.lora_A.weight":{"dtype":"F32","shape":[1,3],"data_offsets":[0,12]},
            "base_model.model.language_model.model.layers.0.attn.wo_b.lora_B.weight":{"dtype":"F32","shape":[2,1],"data_offsets":[12,20]}
        })
        .to_string();
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend(h.as_bytes());
        for x in [1f32, 2., 3., -0.5, 0.25] {
            b.extend(x.to_le_bytes());
        }
        (cfg, b)
    }
    #[test]
    fn rank_one_math_and_scale() {
        let (c, b) = fixture();
        let a = Adapter::parse(&c, &b, 1, 2, 3, 0.5).unwrap();
        let l = &a.layers[&0];
        let x = [2., -1., 4.];
        let ax: f32 = l.a.iter().zip(x).map(|(a, x)| a * x).sum();
        assert_eq!(a.scale * l.b[0] * ax, -3.);
        assert_eq!(a.scale * l.b[1] * ax, 1.5);
    }
    #[test]
    fn rejects_bad_config_and_shape() {
        let (c, b) = fixture();
        for (key, value) in [
            ("r", serde_json::json!(2)),
            ("use_dora", serde_json::json!(true)),
            ("layers_to_transform", serde_json::json!([0, 0])),
        ] {
            let mut bad = c.clone();
            bad[key] = value;
            assert!(Adapter::parse(&bad, &b, 1, 2, 3, 1.).is_err());
        }
        assert!(Adapter::parse(&c, &b, 1, 2, 4, 1.).is_err());
        assert!(Adapter::parse(&c, &b, 1, 2, 3, f32::NAN).is_err());
    }
    #[test]
    fn rejects_truncation_and_nonfinite() {
        let (c, mut b) = fixture();
        for n in [0, 7, 15, b.len() - 1] {
            assert!(Adapter::parse(&c, &b[..n], 1, 2, 3, 1.).is_err());
        }
        let n = b.len();
        b[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Adapter::parse(&c, &b, 1, 2, 3, 1.).is_err());
    }
}
