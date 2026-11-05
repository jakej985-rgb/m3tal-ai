package kernel

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync/atomic"
)

// Capabilities describes which primitive operations a backend implements
// natively on the device. Operations not listed fall back to the CPU path.
type Capabilities struct {
	MatMulFloat32      bool
	MatMulFloat64      bool
	ElementwiseFloat32 bool
	ElementwiseFloat64 bool
	ScaleFloat32       bool
	ScaleFloat64       bool
	SumFloat32         bool
	SumFloat64         bool
	MaxFloat32         bool
	MaxFloat64         bool
	// AttentionFloat32 and RMSNormFloat32 are populated by the CUDA
	// attention/norm kernels (Stage 1, issue #18).
	AttentionFloat32 bool
	RMSNormFloat32   bool
}

// BackendInfo describes a single execution backend known to the process.
type BackendInfo struct {
	// Name is the canonical backend identifier ("cpu", "cuda").
	Name string
	// Device is the device string reported by the backend ("cpu", "cuda:0").
	Device string
	// CompiledIn reports whether the backend code was compiled into the binary.
	CompiledIn bool
	// Available reports whether the backend can execute right now.
	Available bool
	// Reason explains why the backend is unavailable, or gives extra detail
	// (driver version, device name) when it is available.
	Reason string
	// Capabilities lists the operations the backend runs natively.
	Capabilities Capabilities
}

// cpuCapabilities enumerates every primitive the CPU dispatch layer supports.
var cpuCapabilities = Capabilities{
	MatMulFloat32:      true,
	MatMulFloat64:      true,
	ElementwiseFloat32: true,
	ElementwiseFloat64: true,
	ScaleFloat32:       true,
	ScaleFloat64:       true,
	SumFloat32:         true,
	SumFloat64:         true,
	MaxFloat32:         true,
	MaxFloat64:         true,
	AttentionFloat32:   false, // attention currently runs as composed CPU ops
	RMSNormFloat32:     false,
}

var (
	selectedBackend atomic.Value // stores string
	backendProbe    = probeBackends
)

func init() {
	selectedBackend.Store("cpu")
	_ = SelectFromEnv()
}

// Probe returns the status of every backend compiled into the binary.
// The CPU backend is always available; CUDA availability depends on the
// build tags and on a compatible driver/device being present at runtime.
func Probe() []BackendInfo {
	return backendProbe()
}

// probeBackends is a variable so tests can substitute a fake probe.
func probeBackends() []BackendInfo {
	backends := []BackendInfo{{
		Name:         "cpu",
		Device:       "cpu",
		CompiledIn:   true,
		Available:    true,
		Reason:       "host CPU with SIMD dispatch (AVX2/NEON when present)",
		Capabilities: cpuCapabilities,
	}}
	return append(backends, cudaProbe()...)
}

// Select marks name as the preferred execution backend. Selection is
// validated against Probe(): unknown names, backends that were not compiled
// in, and backends without a usable device all return an actionable error.
//
// name may be "cpu", "cuda", or "cuda:N". The special name "auto" selects
// CUDA when available and falls back to CPU otherwise.
func Select(name string) error {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" || name == "auto" {
		if cuda := findBackend("cuda"); cuda.CompiledIn && cuda.Available {
			selectedBackend.Store("cuda")
			return nil
		}
		selectedBackend.Store("cpu")
		return nil
	}

	base := name
	if idx := strings.IndexByte(name, ':'); idx >= 0 {
		base = name[:idx]
	}
	if base != "cpu" && base != "cuda" {
		return fmt.Errorf("unknown backend %q: supported backends are \"cpu\", \"cuda\", and \"auto\"", name)
	}

	info, ok := lookupBackend(base)
	if !ok {
		return fmt.Errorf("backend %q is not compiled into this binary", base)
	}
	if !info.CompiledIn {
		return fmt.Errorf("backend %q was not compiled in: rebuild with CGO_ENABLED=1 go build -tags cuda (requires the CUDA toolkit; see docs/CUDA.md)", base)
	}
	if !info.Available {
		return fmt.Errorf("backend %q is unavailable: %s", base, info.Reason)
	}
	selectedBackend.Store(base)
	return nil
}

// SelectFromEnv applies the M3TAL_LLM_BACKEND environment variable.
// Empty or unset leaves the default ("cpu"). Invalid values are ignored so
// a typo in the environment cannot break a run; use Select for strict
// validation.
func SelectFromEnv() error {
	return Select(os.Getenv("M3TAL_LLM_BACKEND"))
}

// Selected returns the name of the currently preferred backend.
func Selected() string {
	if v, ok := selectedBackend.Load().(string); ok {
		return v
	}
	return "cpu"
}

// preferCUDA reports whether dispatch should route operations to the CUDA
// backend. Kernel dispatch lands with the CUDA kernels; until then this only
// records intent.
func preferCUDA() bool {
	return Selected() == "cuda"
}

func lookupBackend(name string) (BackendInfo, bool) {
	for _, b := range Probe() {
		if b.Name == name {
			return b, true
		}
	}
	return BackendInfo{}, false
}

func findBackend(name string) BackendInfo {
	b, _ := lookupBackend(name)
	return b
}

// Report renders Probe() results as human-readable lines for CLI output.
func Report() string {
	var sb strings.Builder
	for _, b := range Probe() {
		status := "unavailable"
		if b.Available {
			status = "available"
		}
		fmt.Fprintf(&sb, "%s (%s): %s — %s\n", b.Name, b.Device, status, b.Reason)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// SortedBackendNames returns backend names in deterministic order.
func SortedBackendNames() []string {
	names := make([]string, 0, 2)
	for _, b := range Probe() {
		names = append(names, b.Name)
	}
	sort.Strings(names)
	return names
}
