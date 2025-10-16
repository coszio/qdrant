# gRPC Compression Fix - Performance Analysis

## Problem Statement

There was a significant latency discrepancy (4x slower) between REST and gRPC interfaces for queries that return points with large payloads. Despite gRPC being generally faster, it was showing much worse performance for large responses.

## Root Cause Analysis

### Investigation

Profiling revealed that compression was a major contributor to the latency issue. Further analysis of the codebase showed:

**REST API (actix/mod.rs):**
```rust
.wrap(Compress::default()) // Only compresses when client requests it via Accept-Encoding header
```

**gRPC API (tonic/mod.rs - BEFORE FIX):**
```rust
.send_compressed(CompressionEncoding::Gzip)      // Always compresses responses
.accept_compressed(CompressionEncoding::Gzip)    // Accepts compressed requests
```

### Why This Was a Problem

1. **Forced Compression**: The `.send_compressed()` configuration forces all responses to be gzip-compressed, regardless of:
   - Payload size
   - Payload compressibility
   - Client preferences
   - Network conditions

2. **Large Payload Characteristics**: Queries returning points with large payloads typically contain:
   - Vector embeddings (high-dimensional float arrays)
   - Binary data
   - These types of data are generally NOT very compressible

3. **CPU vs Network Tradeoff**: For large, incompressible payloads:
   - **CPU Cost**: Gzip compression is expensive (especially at default compression levels)
   - **Network Benefit**: Minimal size reduction (vectors don't compress well)
   - **Result**: Net negative performance - spending CPU cycles for little benefit

4. **REST Advantage**: REST API only compressed when clients explicitly requested it via the `Accept-Encoding` header, so it avoided the compression overhead for these queries.

## Solution

Removed `.send_compressed(CompressionEncoding::Gzip)` from all gRPC services while keeping `.accept_compressed(CompressionEncoding::Gzip)`.

**gRPC API (tonic/mod.rs - AFTER FIX):**
```rust
.accept_compressed(CompressionEncoding::Gzip)    // Still accepts compressed requests
.max_decoding_message_size(usize::MAX)           // No forced send compression
```

### What This Achieves

1. **No Forced Compression**: Responses are sent uncompressed by default
2. **Opt-in Compression**: Clients can still request compression by including the `grpc-accept-encoding: gzip` header in their requests
3. **Consistent Behavior**: Aligns gRPC behavior with REST API
4. **Better Defaults**: Optimizes for the common case (large, incompressible payloads)

## Performance Impact

### Expected Improvements

- **4x faster** for queries returning large payloads (based on initial profiling)
- **Reduced CPU usage** on the server side (no compression overhead)
- **Lower latency** for incompressible data (vectors, embeddings)

### When to Use Compression

Clients should request compression (`grpc-accept-encoding: gzip`) when:
- Network bandwidth is limited
- Payloads are highly compressible (text, JSON metadata)
- Network transfer time dominates over CPU time

Clients should NOT request compression when:
- Payloads contain mostly vectors/embeddings
- Low-latency is critical
- Server or client CPU is constrained
- On fast local networks

## Modified Services

### Public gRPC Services (init function)
- QdrantServer
- CollectionsServer
- PointsServer
- SnapshotsServer
- HealthServer

### Internal gRPC Services (init_internal function)
- QdrantInternalServer
- CollectionsInternalServer
- PointsInternalServer
- ShardSnapshotsServer
- RaftServer

## Testing Recommendations

To verify the fix:

1. **Benchmark large payload queries**: 
   - Compare gRPC vs REST performance for searches returning many points with vectors
   - Should see similar or better gRPC performance

2. **Test with compression enabled**:
   - Client can add `grpc-accept-encoding: gzip` header
   - Verify compression still works when requested

3. **Monitor metrics**:
   - CPU usage should decrease for gRPC endpoints
   - End-to-end latency should improve significantly

## References

- [Tonic Compression Documentation](https://docs.rs/tonic/latest/tonic/transport/server/struct.Server.html)
- gRPC compression is controlled by the `grpc-accept-encoding` header
- `send_compressed` = server always compresses responses
- `accept_compressed` = server accepts compressed requests from clients
