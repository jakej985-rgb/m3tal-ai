package tensor

import "fmt"

// Storage represents a contiguous block of bytes on a device.
// Multiple Tensors can share a single Storage via strided views.
type Storage struct {
	device   Device
	byteSize int
	data     []byte
	owns     bool
}

// NewCPUStorage allocates a new contiguous byte buffer on the host CPU.
// It rejects negative byte sizes and guards against overflow.
func NewCPUStorage(byteSize int) (*Storage, error) {
	if byteSize < 0 {
		return nil, fmt.Errorf("%w: byte size %d must be non-negative", ErrInvalidShape, byteSize)
	}
	return &Storage{
		device:   CPUDevice,
		byteSize: byteSize,
		data:     make([]byte, byteSize),
		owns:     true,
	}, nil
}

// NewCPUStorageWithData wraps an existing byte slice as non-owning or owning CPU storage.
func NewCPUStorageWithData(data []byte, owns bool) (*Storage, error) {
	return &Storage{
		device:   CPUDevice,
		byteSize: len(data),
		data:     data,
		owns:     owns,
	}, nil
}

// Device returns the device hosting this storage.
func (s *Storage) Device() Device {
	if s == nil {
		return CPUDevice
	}
	return s.device
}

// ByteSize returns the allocated size in bytes.
func (s *Storage) ByteSize() int {
	if s == nil {
		return 0
	}
	return s.byteSize
}

// OwnsMemory reports whether this Storage allocated and owns its underlying buffer.
func (s *Storage) OwnsMemory() bool {
	if s == nil {
		return false
	}
	return s.owns
}

// Bytes returns the underlying byte slice. For CPU storage, callers can access
// or mutate bytes directly if appropriate.
func (s *Storage) Bytes() []byte {
	if s == nil {
		return nil
	}
	return s.data
}

// Clone creates an independent copy of this Storage on the CPU.
func (s *Storage) Clone() (*Storage, error) {
	if s == nil {
		return nil, nil
	}
	if !s.device.IsCPU() {
		return nil, fmt.Errorf("%w: cloning non-CPU storage %s is not yet supported", ErrUnsupportedDevice, s.device)
	}
	cloned := make([]byte, len(s.data))
	copy(cloned, s.data)
	return &Storage{
		device:   s.device,
		byteSize: s.byteSize,
		data:     cloned,
		owns:     true,
	}, nil
}
