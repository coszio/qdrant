//! Benchmark comparing LZ4 dictionary strategies for payload compression.
//!
//! Strategies:
//! - **no_dict**: plain LZ4 (no dictionary at all)
//! - **raw_concat**: dictionary built by concatenating raw serialized payloads (old approach)
//! - **schema_dict**: dictionary built from a merged JSON schema with top-100 frequent strings
//!
//! Each strategy is measured on compress and decompress throughput over 10 000 payloads.

use std::collections::HashMap;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gridstore::fixtures::random_payload;
use lz4_flex::block::{compress_prepend_size, compress_prepend_size_with_dict};
use lz4_flex::decompress_size_prepended;
use lz4_flex::block::decompress_size_prepended_with_dict;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::{Map, Value};

const PAYLOAD_COUNT: usize = 10_000;
const DICT_SAMPLE_COUNT: usize = 1_000;
const MAX_DICT_SIZE: usize = 64 * 1024;
const TOP_STRING_COUNT: usize = 100;

/// Serialise all payloads once so we benchmark compression, not serde.
fn generate_payloads(count: usize) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| {
            let p = random_payload(&mut rng, 2);
            serde_json::to_vec(&p).unwrap()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dictionary builders
// ---------------------------------------------------------------------------

/// Old approach: concatenate raw serialised payloads up to 64 KB.
fn build_raw_concat_dict(samples: &[Vec<u8>]) -> Vec<u8> {
    let mut dict = Vec::with_capacity(MAX_DICT_SIZE);
    for sample in samples {
        let remaining = MAX_DICT_SIZE - dict.len();
        if remaining == 0 {
            break;
        }
        let slice = &sample[..sample.len().min(remaining)];
        dict.extend_from_slice(slice);
    }
    dict
}

/// New approach: merge all sampled payloads into a single JSON schema
/// with default leaf values, plus the top-100 most frequent strings.
fn build_schema_dict(samples: &[Vec<u8>]) -> Vec<u8> {
    let mut schema = Value::Object(Map::new());
    let mut string_counts: HashMap<String, usize> = HashMap::new();

    for sample in samples {
        if let Ok(value) = serde_json::from_slice::<Value>(sample) {
            collect_strings(&value, &mut string_counts);
            merge_into_schema(&mut schema, &value);
        }
    }

    let top_strings = top_n_strings(&string_counts, TOP_STRING_COUNT);
    if !top_strings.is_empty() {
        if let Value::Object(map) = &mut schema {
            map.insert(
                String::new(),
                Value::Array(top_strings.into_iter().map(Value::String).collect()),
            );
        }
    }

    let mut dict = serde_json::to_vec(&schema).unwrap_or_default();
    dict.truncate(MAX_DICT_SIZE);
    dict
}

fn merge_into_schema(schema: &mut Value, sample: &Value) {
    match (schema, sample) {
        (Value::Object(schema_map), Value::Object(sample_map)) => {
            for (key, value) in sample_map {
                match schema_map.get_mut(key) {
                    Some(existing) => merge_into_schema(existing, value),
                    None => {
                        schema_map.insert(key.clone(), default_value(value));
                    }
                }
            }
        }
        (Value::Array(schema_arr), Value::Array(sample_arr)) => {
            if let Some(sample_elem) = sample_arr.first() {
                if schema_arr.is_empty() {
                    schema_arr.push(default_value(sample_elem));
                } else {
                    merge_into_schema(&mut schema_arr[0], sample_elem);
                }
            }
        }
        _ => {}
    }
}

fn default_value(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        Value::Bool(_) => Value::Bool(false),
        Value::Number(_) => Value::Number(0.into()),
        Value::String(_) => Value::String(String::new()),
        Value::Array(arr) => match arr.first() {
            Some(elem) => Value::Array(vec![default_value(elem)]),
            None => Value::Array(vec![]),
        },
        Value::Object(map) => {
            let new_map = map
                .iter()
                .map(|(k, v)| (k.clone(), default_value(v)))
                .collect();
            Value::Object(new_map)
        }
    }
}

fn collect_strings(value: &Value, counts: &mut HashMap<String, usize>) {
    match value {
        Value::String(s) => {
            *counts.entry(s.clone()).or_default() += 1;
        }
        Value::Array(arr) => {
            for elem in arr {
                collect_strings(elem, counts);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                collect_strings(v, counts);
            }
        }
        _ => {}
    }
}

fn top_n_strings(counts: &HashMap<String, usize>, n: usize) -> Vec<String> {
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_unstable_by(|a, b| b.1.cmp(a.1));
    entries.into_iter().take(n).map(|(s, _)| s.clone()).collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn dict_compress_bench(c: &mut Criterion) {
    let payloads = generate_payloads(PAYLOAD_COUNT);
    let samples = &payloads[..DICT_SAMPLE_COUNT.min(payloads.len())];

    let raw_dict = build_raw_concat_dict(samples);
    let schema_dict = build_schema_dict(samples);

    // Print dictionary sizes for context.
    eprintln!(
        "dict sizes: raw_concat={}B, schema={}B",
        raw_dict.len(),
        schema_dict.len()
    );

    let total_uncompressed: usize = payloads.iter().map(|p| p.len()).sum();
    let total_no_dict: usize = payloads
        .iter()
        .map(|p| compress_prepend_size(p).len())
        .sum();
    let total_raw: usize = payloads
        .iter()
        .map(|p| compress_prepend_size_with_dict(p, &raw_dict).len())
        .sum();
    let total_schema: usize = payloads
        .iter()
        .map(|p| compress_prepend_size_with_dict(p, &schema_dict).len())
        .sum();

    eprintln!(
        "total compressed sizes over {PAYLOAD_COUNT} payloads:\n  \
         uncompressed : {total_uncompressed:>10}B\n  \
         no_dict (lz4): {total_no_dict:>10}B ({:.1}%)\n  \
         raw_concat   : {total_raw:>10}B ({:.1}%)\n  \
         schema       : {total_schema:>10}B ({:.1}%)",
        total_no_dict as f64 / total_uncompressed as f64 * 100.0,
        total_raw as f64 / total_uncompressed as f64 * 100.0,
        total_schema as f64 / total_uncompressed as f64 * 100.0,
    );

    let mut group = c.benchmark_group("dict_compress");
    group.throughput(criterion::Throughput::Bytes(total_uncompressed as u64));

    group.bench_function(BenchmarkId::new("compress", "no_dict"), |b| {
        b.iter(|| {
            for payload in &payloads {
                std::hint::black_box(compress_prepend_size(payload));
            }
        });
    });

    group.bench_function(BenchmarkId::new("compress", "raw_concat"), |b| {
        b.iter(|| {
            for payload in &payloads {
                std::hint::black_box(compress_prepend_size_with_dict(payload, &raw_dict));
            }
        });
    });

    group.bench_function(BenchmarkId::new("compress", "schema"), |b| {
        b.iter(|| {
            for payload in &payloads {
                std::hint::black_box(compress_prepend_size_with_dict(payload, &schema_dict));
            }
        });
    });

    group.finish();

    // --- decompress ---
    let compressed_no_dict: Vec<Vec<u8>> = payloads
        .iter()
        .map(|p| compress_prepend_size(p))
        .collect();
    let compressed_raw: Vec<Vec<u8>> = payloads
        .iter()
        .map(|p| compress_prepend_size_with_dict(p, &raw_dict))
        .collect();
    let compressed_schema: Vec<Vec<u8>> = payloads
        .iter()
        .map(|p| compress_prepend_size_with_dict(p, &schema_dict))
        .collect();

    let mut group = c.benchmark_group("dict_decompress");
    group.throughput(criterion::Throughput::Bytes(total_uncompressed as u64));

    group.bench_function(BenchmarkId::new("decompress", "no_dict"), |b| {
        b.iter(|| {
            for compressed in &compressed_no_dict {
                std::hint::black_box(decompress_size_prepended(compressed).unwrap());
            }
        });
    });

    group.bench_function(BenchmarkId::new("decompress", "raw_concat"), |b| {
        b.iter(|| {
            for compressed in &compressed_raw {
                std::hint::black_box(
                    decompress_size_prepended_with_dict(compressed, &raw_dict).unwrap(),
                );
            }
        });
    });

    group.bench_function(BenchmarkId::new("decompress", "schema"), |b| {
        b.iter(|| {
            for compressed in &compressed_schema {
                std::hint::black_box(
                    decompress_size_prepended_with_dict(compressed, &schema_dict).unwrap(),
                );
            }
        });
    });

    group.finish();
}

criterion_group!(benches, dict_compress_bench);
criterion_main!(benches);
