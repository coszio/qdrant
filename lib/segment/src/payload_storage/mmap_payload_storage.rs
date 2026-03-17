use std::path::{Path, PathBuf};

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use fs_err as fs;
use gridstore::config::{Compression, StorageOptions};
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

    /// Optimize compression by switching from plain LZ4 to LZ4 with a trained dictionary.
    ///
    /// This is a no-op when the storage already uses `LZ4Dict` (or `None`).
    ///
    /// The method:
    /// 1. Samples up to 1000 random payloads and trains a dictionary.
    /// 2. Creates a secondary storage with `LZ4Dict` in a temporary directory.
    /// 3. Rewrites every payload into the new storage.
    /// 4. Swaps the directories so the optimized storage takes over.
    pub fn optimize<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> OperationResult<()> {
        if self.storage.compression() != Compression::LZ4 {
            return Ok(());
        }

        let hw_counter = HardwareCounterCell::disposable();

        // --- 1. Sample payloads and build dictionary --------------------------------
        let max_offset = self.storage.max_point_offset();
        if max_offset == 0 {
            return Ok(());
        }

        let sample_count = Self::DICT_SAMPLE_SIZE.min(max_offset as usize);
        let indices = sample_indices(rng, max_offset as usize, sample_count);

        let mut schema = Value::Object(Map::new());
        for idx in indices.iter() {
            if let Some(payload) = self
                .storage
                .get_value::<false>(idx as PointOffsetType, &hw_counter)
                .ok()
                .flatten()
            {
                merge_into_schema(&mut schema, &Value::Object(payload.0));
            }
        }
        let dictionary = serde_json::to_vec(&schema).unwrap_or_default();

        // --- 2. Create new storage with LZ4Dict in a tmp directory ------------------
        let base_path = self.storage.base_path().to_path_buf();
        let tmp_dir = tempfile::tempdir_in(base_path.parent().unwrap_or(Path::new(".")))?;
        let tmp_path = tmp_dir.path().to_path_buf();

        Gridstore::<Payload>::write_dictionary(&tmp_path, &dictionary)?;

        let options = StorageOptions {
            compression: Some(Compression::LZ4Dict),
            ..StorageOptions::default()
        };
        let mut new_storage = Gridstore::<Payload>::new(tmp_path.clone(), options)?;

        // --- 3. Rewrite all payloads ------------------------------------------------
        self.storage.iter(
            |point_id, payload: Payload| -> OperationResult<bool> {
                new_storage.put_value(
                    point_id,
                    &payload,
                    hw_counter.ref_payload_io_write_counter(),
                )?;
                Ok(true)
            },
            hw_counter.ref_payload_io_read_counter(),
        )?;

        new_storage.flusher()().map_err(|e| {
            OperationError::service_error(format!("Failed to flush new storage: {e}"))
        })?;

        // --- 4. Swap directories ----------------------------------------------------
        let old_path = base_path.with_extension("lz4_old");
        if old_path.exists() {
            fs::remove_dir_all(&old_path)?;
        }

        // old storage must be dropped before renaming its directory
        drop(std::mem::replace(
            &mut self.storage,
            new_storage,
        ));

        // base_path -> old_path, tmp_path -> base_path
        fs::rename(&base_path, &old_path)?;
        fs::rename(&tmp_path, &base_path)?;
        fs::remove_dir_all(&old_path)?;

        // Reopen from the final location
        self.storage = Gridstore::open(base_path).map_err(|err| {
            OperationError::service_error(format!(
                "Failed to reopen optimized payload storage: {err}"
            ))
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
