package tensor

import "fmt"

// DeviceType identifies the execution hardware platform hosting tensor storage.
type DeviceType uint8

const (
	// CPU indicates host system RAM and CPU execution.
	CPU DeviceType = iota
	// CUDA indicates an NVIDIA GPU device managed via CUDA.
	CUDA
)

// String returns the string representation of DeviceType.
func (dt DeviceType) String() string {
	switch dt {
	case CPU:
		return "cpu"
	case CUDA:
		return "cuda"
	default:
		return fmt.Sprintf("device(%d)", uint8(dt))
	}
}

// Device encapsulates a device category and hardware ordinal index.
type Device struct {
	Type  DeviceType
	Index int
}

// CPUDevice is the default host CPU execution target.
var CPUDevice = Device{Type: CPU, Index: 0}

// CUDADevice returns a CUDA device descriptor with the given device index.
func CUDADevice(index int) Device {
	return Device{Type: CUDA, Index: index}
}

// String returns a canonical string identifier for the device (e.g., "cpu", "cuda:0").
func (d Device) String() string {
	if d.Type == CPU {
		return "cpu"
	}
	return fmt.Sprintf("%s:%d", d.Type.String(), d.Index)
}

// IsCPU reports whether the device is host CPU memory.
func (d Device) IsCPU() bool {
	return d.Type == CPU
}
