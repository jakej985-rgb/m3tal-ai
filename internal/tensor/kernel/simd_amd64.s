//go:build amd64

#include "textflag.h"

// func addFloat32AVX2Asm(dst, a, b unsafe.Pointer, n int)
TEXT ·addFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	MOVQ n+24(FP), CX

loop8:
	CMPQ CX, $8
	JL tail
	VMOVUPS (SI), Y0
	VMOVUPS (DX), Y1
	VADDPS Y1, Y0, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $8, CX
	JMP loop8

tail:
	CMPQ CX, $0
	JLE done
	// Single element fallback for remaining < 8 elements
	MOVSS (SI), X0
	ADDSS (DX), X0
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	SUBQ $1, CX
	JMP tail

done:
	VZEROUPPER
	RET

// func mulFloat32AVX2Asm(dst, a, b unsafe.Pointer, n int)
TEXT ·mulFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	MOVQ n+24(FP), CX

mloop8:
	CMPQ CX, $8
	JL mtail
	VMOVUPS (SI), Y0
	VMOVUPS (DX), Y1
	VMULPS Y1, Y0, Y2
	VMOVUPS Y2, (DI)
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, DI
	SUBQ $8, CX
	JMP mloop8

mtail:
	CMPQ CX, $0
	JLE mdone
	MOVSS (SI), X0
	MULSS (DX), X0
	MOVSS X0, (DI)
	ADDQ $4, SI
	ADDQ $4, DX
	ADDQ $4, DI
	SUBQ $1, CX
	JMP mtail

mdone:
	VZEROUPPER
	RET

// func dotFloat32AVX2Asm(a, b unsafe.Pointer, n int) float32
TEXT ·dotFloat32AVX2Asm(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DX
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0 // accumulator

dloop8:
	CMPQ CX, $8
	JL dtail
	VMOVUPS (SI), Y1
	VMOVUPS (DX), Y2
	VFMADD231PS Y1, Y2, Y0 // Y0 = Y0 + (Y1 * Y2)
	ADDQ $32, SI
	ADDQ $32, DX
	SUBQ $8, CX
	JMP dloop8

dtail:
	// Horizontal add of Y0:
	// Extract high 128 to X1, add with low 128 (X0)
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	// Now X0 has 4 floats: [a, b, c, d]
	VMOVHLPS X0, X1, X1 // X1 = [c, d, ?, ?]
	VADDPS X1, X0, X0   // X0 = [a+c, b+d, ...]
	VSHUFPS $1, X0, X0, X1 // X1 = [b+d, ...]
	ADDSS X1, X0        // X0 = (a+c) + (b+d)

dscalar:
	CMPQ CX, $0
	JLE ddone
	MOVSS (SI), X1
	MULSS (DX), X1
	ADDSS X1, X0
	ADDQ $4, SI
	ADDQ $4, DX
	SUBQ $1, CX
	JMP dscalar

ddone:
	MOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// func fmaRowFloat32AVX2Asm(dst, b unsafe.Pointer, alpha float32, n int)
// dst[j] += alpha * b[j]
TEXT ·fmaRowFloat32AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ b+8(FP), SI
	VBROADCASTSS alpha+16(FP), Y0
	MOVQ n+24(FP), CX

fmaloop8:
	CMPQ CX, $8
	JL fmatail
	VMOVUPS (DI), Y1
	VMOVUPS (SI), Y2
	VFMADD231PS Y0, Y2, Y1 // Y1 = Y1 + Y0 * Y2
	VMOVUPS Y1, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $8, CX
	JMP fmaloop8

fmatail:
	CMPQ CX, $0
	JLE fmadone
	MOVSS (DI), X1
	MOVSS (SI), X2
	MULSS X0, X2
	ADDSS X2, X1
	MOVSS X1, (DI)
	ADDQ $4, SI
	ADDQ $4, DI
	SUBQ $1, CX
	JMP fmatail

fmadone:
	VZEROUPPER
	RET
