using AMDGPU
using AMDGPU: @roc
using LinearAlgebra
using Printf

const N_QUBITS::Int = 4 # It gives a 2⁴ = 16 state vector
const DIM::Int = 2^N_QUBITS

# State vector initialisation functions

# Creates the |00..0⟩ state on GPU
function gpu_zero_state(n_qubits::Int)::ROCArray{ComplexF64,1}
    @assert n_qubits >= 0 "n_qubits must be non-negative"
    N = 1 << n_qubits # 2ⁿ via bit shift
    # This way is better for large state vectors
    host = zeros(ComplexF64, N) # CPU array
    host[1] = 1.0 + 0.0im # Ok on cpu
    return ROCArray(host) # copied to GPU
    #return ROCArray([i == 1 ? 1.0 + 0.0im : 0.0 + 0.0im for i in 1:N]) # copied to GPU
end

# Creates an equal superposition of all basis states:
# every amplitude is 1√(2ⁿ)
function gpu_uniform_state(n_qubits::Int)::ROCArray{ComplexF64,1}
    @assert n_qubits >= 0 "n_qubits must be non_negative"
    N = 1 << n_qubits
    amp = inv(sqrt(Float64(N))) # 1/√(2ⁿ) in Float64
    host = fill(ComplexF64(amp, 0.0), N) # fill on cpu
    return ROCArray(host) # copied to GPU
end

# Verification

# Create states on GPU
ψ_zero_gpu = gpu_zero_state(N_QUBITS)
ψ_unif_gpu = gpu_uniform_state(N_QUBITS)

println("Zero type: ", typeof(ψ_zero_gpu))
println("Uniform type: ", typeof(ψ_unif_gpu))

println("Zero norm² (GPU): ", sum(abs2.(ψ_zero_gpu)))
println("Uniform norm² (GPU): ", sum(abs2.(ψ_unif_gpu)))

println("Zero (CPU view): ", Array(ψ_zero_gpu))
println("Uniform (CPU view): ", Array(ψ_unif_gpu))

# Gate definitions

const INV_SQRT2 = inv(sqrt(2.0)) # 1√2 as a global constant

const GATE_H = (
        ComplexF64(INV_SQRT2, 0.0), ComplexF64(INV_SQRT2, 0.0),
        ComplexF64(INV_SQRT2, 0.0), ComplexF64(-INV_SQRT2, 0.0),
    )

const GATE_X = (
    ComplexF64(0.0, 0.0), ComplexF64(1.0, 0.0),
    ComplexF64(1.0, 0.0), ComplexF64(0.0, 0.0),
)

const GATE_Z = (
    ComplexF64(1.0, 0.0), ComplexF64(0.0, 0.0),
    ComplexF64(0.0, 0.0), ComplexF64(-1.0, 0.0),
)

# Single-qubit gate GPU kernel

function apply_single_gate_kernel!(
    ψ, a::ComplexF64, b::ComplexF64, c::ComplexF64, d::ComplexF64,
    k::Int, n_qubits::Int
)
    # 0-based global thread id (convert from Julia's 1-based work * indices)
    tid = (workgroupIdx().x -1) * workgroupDim().x + workitemIdx().x - 1
    n_pairs = Int(1) << (n_qubits - 1)
    if tid >= n_pairs
        return
    end

    # Bit mapping for indices
    low_mask = (Int(1) << k) - 1
    low_bits = tid & low_mask
    high_bits = tid >> k

    i0 = (high_bits << (k + 1)) | low_bits # insert 0 at bit kernel
    i1 = i0 | (Int(1) << k) # same with bit k = 1

    # Convert to 1-based for Julia vector indexing
    i0 += 1
    i1 += 1

    @inbounds begin
        α = ψ[i0]
        β = ψ[i1]
        ψ[i0] = a*α + b*β
        ψ[i1] = c*α + d*β
    end
    return
end

# Host-side launcher

function apply_gate!(
    ψ::ROCArray{ComplexF64, 1},
    gate::NTuple{4, ComplexF64},
    target_qubit::Int,
    n_qubits::Int;
    groupsize::Int = 256, # 4 wavefronts of 64
)
    @assert 0 <= target_qubit < n_qubits "targe_qubit must be in 0..n_qubits-1"

    a, b, c, d = gate
    n_pairs = Int(1) << (n_qubits - 1)
    n_groups = cld(n_pairs, groupsize) # ceiling division

    @roc groupsize=groupsize gridsize=n_groups apply_single_gate_kernel!(
        ψ, a, b, c, d, target_qubit, n_qubits
    )
    AMDGPU.synchronize()
    return ψ
end

# Test

println("Single-qubit gate kernel")

ψ_h = gpu_zero_state(N_QUBITS)
apply_gate!(ψ_h, GATE_H, 0, N_QUBITS) # H on least significant qubit (0-based)

norm2_h = sum(abs2.(ψ_h)) # compute on gpu_uniform_state
ψ_h_cpu = Array(ψ_h) # pull to host to inspect

println("Applied H to qubit to qubit 0 of |0..0⟩")
println("Type (GPU): ", typeof(ψ_h))
println("Normalisation (GPU): Σ|ψ|² = ", norm2_h)
println("State (CPU view):", ψ_h_cpu)

amp_expected = inv(sqrt(Float64(2)))
println(@sprintf("Expected ψ[1]=ψ[2]≈ %.6f + 0.0im, others ≈ 0", amp_expected))

# Sanity checks: X and Z

# X on qubit 0 maps |0000⟩ → |0001⟩
ψ_x = gpu_zero_state(N_QUBITS)
apply_gate!(ψ_x, GATE_X, 0, N_QUBITS)
println("X on qubit 0 of |0..0⟩ → expect only index 2 = 1:", Array(ψ_x))

# Z on qubit 0 keeps |0000⟩ unchanged (phase only affects |...1⟩ branch)
ψ_z = gpu_zero_state(N_QUBITS)
apply_gate!(ψ_z, GATE_Z, 0, N_QUBITS)
println("Z on qubit 0 of |0..0⟩ → expect same as zero state:", Array(ψ_z))