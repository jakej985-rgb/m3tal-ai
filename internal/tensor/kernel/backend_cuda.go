//go:build cuda && cgo

package kernel

/*
#cgo LDFLAGS: -lcudart
#include <cuda_runtime.h>
#include <string.h>

// Returns the number of CUDA-capable devices, or -1 when the runtime
// cannot enumerate devices (no driver, driver/runtime mismatch, etc.).
static int m3tal_cuda_device_count() {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		return -1;
	}
	return count;
}

// Fills name/major/minor for device index; leaves outputs untouched on error.
static void m3tal_cuda_device_info(int device, char *name, int nameLen, int *major, int *minor) {
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
		return;
	}
	strncpy(name, prop.name, (size_t)nameLen - 1);
	name[nameLen - 1] = '\0';
	*major = prop.major;
	*minor = prop.minor;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// cudaProbe enumerates CUDA devices through the CUDA runtime API. It only
// runs in binaries built with `-tags cuda` and cgo enabled.
func cudaProbe() []BackendInfo {
	count := int(C.m3tal_cuda_device_count())
	if count < 0 {
		return []BackendInfo{{
			Name:       "cuda",
			Device:     "cuda",
			CompiledIn: true,
			Available:  false,
			Reason: "CUDA runtime cannot enumerate devices: check that an " +
				"NVIDIA driver is installed and matches the CUDA runtime " +
				"(nvidia-smi; see docs/CUDA.md)",
		}}
	}
	if count == 0 {
		return []BackendInfo{{
			Name:       "cuda",
			Device:     "cuda",
			CompiledIn: true,
			Available:  false,
			Reason:     "no CUDA-capable NVIDIA devices detected",
		}}
	}

	nameBuf := make([]byte, 256)
	var major, minor C.int
	C.m3tal_cuda_device_info(0, (*C.char)(unsafe.Pointer(&nameBuf[0])), C.int(len(nameBuf)), &major, &minor)
	devName := C.GoString((*C.char)(unsafe.Pointer(&nameBuf[0])))

	return []BackendInfo{{
		Name:       "cuda",
		Device:     "cuda:0",
		CompiledIn: true,
		Available:  true,
		Reason: fmt.Sprintf("%d device(s); cuda:0 = %q (compute capability %d.%d)",
			count, devName, int(major), int(minor)),
	}}
}
