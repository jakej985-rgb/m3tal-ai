// Package tensor defines foundational multi-dimensional tensor abstractions,
// including shapes, strides, data types, hardware devices, and storage ownership contracts.
//
// # Tensor Representation
//
// A [Tensor] represents an N-dimensional array with elements of a uniform [DType].
// Each Tensor maintains:
//   - shape: an N-element slice of integers defining the extent of each dimension.
//   - strides: an N-element slice of integers defining the step size in element counts
//     between adjacent entries along each dimension.
//   - offset: an integer specifying the element offset from the beginning of the underlying [Storage].
//   - dtype: the scalar data type (such as [Float32], [Float64], [Int32], [Int64], or [Uint8]).
//   - device: the target execution device (such as host [CPUDevice] or a [CUDADevice]).
//   - storage: a reference to a contiguous block of bytes on the device.
//
// # Shape and Dimension Semantics
//
// 1. Non-negative Dimensions: All dimension extents must be non-negative (>= 0).
// An invalid negative dimension returns [ErrInvalidShape].
//
// 2. Scalar Tensors: A 0-dimensional tensor (empty shape slice `[]int{}`) represents a scalar.
// Scalars have [Tensor.Rank] 0 and contain 1 element ([Tensor.Numel] == 1) with empty strides.
//
// 3. Zero-size Tensors: Any shape with at least one dimension equal to 0 represents an empty
// tensor with 0 elements ([Tensor.Numel] == 0).
//
// 4. Safe Overflow Detection: Total element count and required byte sizes are verified against
// integer multiplication overflow (e.g. math.MaxInt) and return [ErrStorageOverflow] if overflow occurs.
//
// # Strides and Contiguity
//
// Strides define how multi-dimensional coordinates map to linear storage offsets:
//
//	linear_offset = offset + sum(indices[i] * strides[i])
//
// By default, newly constructed tensors are in standard C-order row-major layout:
//   - The last dimension has stride 1.
//   - Dimension i has stride equal to stride[i+1] * shape[i+1].
//
// A tensor is considered contiguous ([Tensor.IsContiguous]) when its strides match the standard
// C-order contiguous layout for its shape. Non-contiguous tensors occur when creating strided views,
// transpositions, or sub-dimensional slices. Operations that require contiguous layout (such as
// [Tensor.Reshape]) return [ErrNonContiguous] if invoked on a non-contiguous tensor unless it is first
// cloned into a contiguous buffer with [Tensor.Clone].
//
// # Storage Ownership and Views
//
// 1. Shared Storage: Operations like [Tensor.Slice] and [Tensor.Reshape] return new Tensor headers
// that share the underlying [Storage] buffer with the parent tensor without memory allocation or copying.
// Mutations performed on elements of a sliced view directly mutate the parent storage and vice-versa.
//
// 2. Independent Copies: [Tensor.Clone] allocates a new, independent [Storage] buffer and copies
// values according to the source tensor's shape and strides. Cloned tensors are always contiguous.
//
// 3. Storage Ownership: [Storage.OwnsMemory] reports whether the storage owns its buffer.
// When created via [New] or [NewCPUStorage], the storage owns the allocated buffer.
//
// # CPU Device Contract
//
// Tensors created via [New], [FromFloat32], and [FromFloat64] reside on the host CPU ([CPUDevice]).
// CPU storage uses little-endian byte representations and host RAM. Hardware execution bounds are
// checked on all index accesses: out-of-bounds index requests return [ErrIndexOutOfBounds].
package tensor
