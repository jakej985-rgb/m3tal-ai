//go:build !amd64 && !arm64

package cpu

func detectFeatures() CPUFeatures {
	return CPUFeatures{}
}
