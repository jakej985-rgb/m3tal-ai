//go:build cuda && cgo && amd64

package gpu

//go:generate nvcc -O3 -std=c++14 -arch=native -c kernels.cu -o kernels_amd64.o
