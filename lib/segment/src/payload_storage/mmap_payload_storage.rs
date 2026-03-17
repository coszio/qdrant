use std::collections::HashMap;
use std::path::{Path, PathBuf};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use fs_err as fs;
use gridstore::config::StorageOptions;
use gridstore::{Blob, Gridstore};
use rand::Rng;
use rand::seq::index::sample as sample_indices;
use serde_json::{Map, Value};

use crate::common::Flusher;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::json_path::JsonPath;
use crate::payload_storage::PayloadStorage;
use crate::types::{Payload, PayloadKeyTypeRef};

const STORAGE_PATH: &str = "payload_storage";

impl Blob for Payload {
    fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap()
    }

    fn from_bytes(data: &[u8]) -> Self {
        serde_json::from_slice(data).unwrap()
    }
}

#[derive(Debug)]
pub struct MmapPayloadStorage {
    storage: Gridstore<Payload>,
    populate: bool,
}

impl MmapPayloadStorage {
    pub fn open_or_create(path: PathBuf, populate: bool) -> OperationResult<Self> {
        let path = storage_dir(path);
        if path.exists() {
            Self::open(path, populate)
        } else {
            // create folder if it does not exist
            fs::create_dir_all(&path).map_err(|_| {
                OperationError::service_error("Failed to create mmap payload storage directory")
            })?;
            Ok(Self::new(path, populate)?)
        }
    }

    fn open(path: PathBuf, populate: bool) -> OperationResult<Self> {
        let storage = Gridstore::open(path).map_err(|err| {
            OperationError::service_error(format!("Failed to open mmap payload storage: {err}"))
        })?;

        if populate {
            storage.populate()?;
        }

        Ok(Self { storage, populate })
    }

    fn new(path: PathBuf, populate: bool) -> OperationResult<Self> {
        let storage = Gridstore::new(path, StorageOptions::default())?;

        if populate {
            storage.populate()?;
        }

        Ok(Self { storage, populate })
    }

    /// Populate all pages in the mmap.
    /// Block until all pages are populated.
    pub fn populate(&self) -> OperationResult<()> {
        self.storage.populate()?;
        Ok(())
    }

    /// Drop disk cache.
    pub fn clear_cache(&self) -> OperationResult<()> {
        self.storage.clear_cache()?;
        Ok(())
    }

    /// Maximum number of payloads sampled for dictionary training.
    const DICT_SAMPLE_SIZE: usize = 1000;

    /// LZ4 only uses the last 64KB of the dictionary for matching.
    const MAX_DICT_SIZE: usize = 64 * 1024;

    /// Number of most-frequent string values to embed in the dictionary.
    const TOP_STRING_COUNT: usize = 100;

    /// Optimize compression by switching from plain LZ4 to LZ4 with a trained dictionary.
    ///
    /// This is a no-op when the storage already uses `LZ4Dict` (or `None`).
    ///
    /// Samples up to 1000 random payloads, merges their JSON schemas into a compact
    /// dictionary (keys + top-100 frequent string values), then delegates the
    /// rewrite-and-swap to [`Gridstore::optimize`].
    pub fn optimize<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> OperationResult<()> {
        let sample_count = Self::DICT_SAMPLE_SIZE;
        let max_dict_size = Self::MAX_DICT_SIZE;
        let top_string_count = Self::TOP_STRING_COUNT;

        // Pre-generate the sampled indices so the closure doesn't need the rng.
        let max_offset = self.storage.max_point_offset();
        if max_offset == 0 {
            return Ok(());
        }
        let indices: Vec<PointOffsetType> = sample_indices(
            rng,
            max_offset as usize,
            sample_count.min(max_offset as usize),
        )
        .iter()
        .map(|i| i as PointOffsetType)
        .collect();

        self.storage
            .optimize(|storage| {
                let hw_counter = HardwareCounterCell::disposable();

                let mut schema = Value::Object(Map::new());
                let mut string_counts: HashMap<String, usize> = HashMap::new();

                for &idx in &indices {
                    if let Some(payload) = storage
                        .get_value::<false>(idx, &hw_counter)
                        .ok()
                        .flatten()
                    {
                        let value = Value::Object(payload.0);
                        collect_strings(&value, &mut string_counts);
                        merge_into_schema(&mut schema, &value);
                    }
                }

                // Inject the top most frequent string values so LZ4 can match
                // against common payload values, not just keys.
                let top_strings = top_n_strings(&string_counts, top_string_count);
                if !top_strings.is_empty() {
                    if let Value::Object(map) = &mut schema {
                        map.insert(
                            String::new(),
                            Value::Array(
                                top_strings.into_iter().map(Value::String).collect(),
                            ),
                        );
                    }
                }

                let mut dict = serde_json::to_vec(&schema).unwrap_or_default();
                dict.truncate(max_dict_size);
                dict
            })
            .map_err(|e| {
                OperationError::service_error(format!("Failed to optimize payload storage: {e}"))
            })?;

        if self.populate {
            self.storage.populate()?;
        }

        Ok(())
    }
}

impl PayloadStorage for MmapPayloadStorage {
    fn overwrite(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.storage
            .put_value(point_id, payload, hw_counter.ref_payload_io_write_counter())?;
        Ok(())
    }

    fn set(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        match self.storage.get_value::<false>(point_id, hw_counter)? {
            Some(mut point_payload) => {
                point_payload.merge(payload);
                self.storage.put_value(
                    point_id,
                    &point_payload,
                    hw_counter.ref_payload_io_write_counter(),
                )?;
            }
            None => {
                self.storage.put_value(
                    point_id,
                    payload,
                    hw_counter.ref_payload_io_write_counter(),
                )?;
            }
        }
        Ok(())
    }

    fn set_by_key(
        &mut self,
        point_id: PointOffsetType,
        payload: &Payload,
        key: &JsonPath,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        match self.storage.get_value::<false>(point_id, hw_counter)? {
            Some(mut point_payload) => {
                point_payload.merge_by_key(payload, key);
                self.storage.put_value(
                    point_id,
                    &point_payload,
                    hw_counter.ref_payload_io_write_counter(),
                )?;
            }
            None => {
                let mut dest_payload = Payload::default();
                dest_payload.merge_by_key(payload, key);
                self.storage.put_value(
                    point_id,
                    &dest_payload,
                    hw_counter.ref_payload_io_write_counter(),
                )?;
            }
        }
        Ok(())
    }

    fn get(
        &self,
        point_id: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Payload> {
        match self.storage.get_value::<false>(point_id, hw_counter)? {
            Some(payload) => Ok(payload),
            None => Ok(Default::default()),
        }
    }

    fn get_sequential(
        &self,
        point_id: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Payload> {
        match self.storage.get_value::<true>(point_id, hw_counter)? {
            Some(payload) => Ok(payload),
            None => Ok(Default::default()),
        }
    }

    fn delete(
        &mut self,
        point_id: PointOffsetType,
        key: PayloadKeyTypeRef,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Vec<Value>> {
        match self.storage.get_value::<false>(point_id, hw_counter)? {
            Some(mut payload) => {
                let res = payload.remove(key);
                if !res.is_empty() {
                    self.storage.put_value(
                        point_id,
                        &payload,
                        hw_counter.ref_payload_io_write_counter(),
                    )?;
                }
                Ok(res)
            }
            None => Ok(vec![]),
        }
    }

    fn clear(
        &mut self,
        point_id: PointOffsetType,
        _: &HardwareCounterCell,
    ) -> OperationResult<Option<Payload>> {
        let res = self.storage.delete_value(point_id)?;
        Ok(res)
    }

    #[cfg(test)]
    fn clear_all(&mut self, _: &HardwareCounterCell) -> OperationResult<()> {
        self.storage.clear().map_err(|err| {
            OperationError::service_error(format!("Failed to clear mmap payload storage: {err}"))
        })
    }

    fn flusher(&self) -> Flusher {
        let storage_flusher = self.storage.flusher();
        Box::new(move || {
            storage_flusher().map_err(|err| {
                OperationError::service_error(format!(
                    "Failed to flush mmap payload gridstore: {err}"
                ))
            })
        })
    }

    fn iter<F>(&self, mut callback: F, hw_counter: &HardwareCounterCell) -> OperationResult<()>
    where
        F: FnMut(PointOffsetType, &Payload) -> OperationResult<bool>,
    {
        self.storage.iter(
            |point_id, payload| callback(point_id, &payload),
            hw_counter.ref_payload_io_read_counter(),
        )
    }

    fn files(&self) -> Vec<PathBuf> {
        self.storage.files()
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        self.storage.immutable_files()
    }

    fn get_storage_size_bytes(&self) -> OperationResult<usize> {
        Ok(self.storage.get_storage_size_bytes())
    }

    fn is_on_disk(&self) -> bool {
        !self.populate
    }
}

/// Get storage directory for this payload storage
pub fn storage_dir<P: AsRef<Path>>(segment_path: P) -> PathBuf {
    segment_path.as_ref().join(STORAGE_PATH)
}

/// Merge a JSON value into a schema that keeps only structure and default values.
///
/// Objects are merged by unioning their keys. Arrays keep a single representative
/// element. Leaf values are replaced with type-appropriate defaults.
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

/// Produce a default value mirroring the structure of `value` with zeroed leaves.
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

/// Recursively collect all string values from a JSON value into a frequency map.
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

/// Return the top `n` most frequent strings, sorted by descending frequency.
fn top_n_strings(counts: &HashMap<String, usize>, n: usize) -> Vec<String> {
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_unstable_by(|a, b| b.1.cmp(a.1));
    entries.into_iter().take(n).map(|(s, _)| s.clone()).collect()
}
