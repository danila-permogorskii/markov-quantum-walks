using AMDGPU
using AMDGPU: @roc
using LinearAlgebra
using Printf
using Random

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

# Two-qubit CNOT kernel

function apply_cnot_kernel!(ψ, c::Int, t::Int, n_qubits::Int)
    # Build a 0-based global thread id
    wg = AMDGPU.workgroupIdx().x # 1-based group index
    wdim = AMDGPU.workgroupDim().x # threads per group
    wi = AMDGPU.workitemIdx().x # 1-based thread index in group
    tid = (wg - 1) * wdim + (wi - 1) # 0-based linear thread id

    # Bounds: number of group-of-four
    n_groups4 = Int(1) << (n_qubits - 2)
    if tid >= n_groups4
        return
    end

    # Order the bit positions
    k_lo = min(c, t)
    k_hi = max(c, t)

    # Double bit insertion
    
    # Insert 0 at k_lo
    low_mask1 = (Int(1) << k_lo) - 1
    low_bits1 = tid & low_mask1
    high_bits1 = tid >> k_lo
    temp = (high_bits1 << (k_lo + 1)) | low_bits1 # n-1 bits with a 0 at k_lo

    # Insert 0 at k_hi
    low_mask2 = (Int(1) << k_hi) - 1
    low_bits2 = temp & low_mask2
    high_bits2 = temp >> k_hi
    i00 = (high_bits2 << (k_hi + 1)) | low_bits2 # n bits, c=t=0

    # build the indices we need for CNOT (control=1 branch)
    i10 = i00 | (Int(1) << c) # control=1, target=0
    i11 = i00 | (Int(1) << c) | (Int(1) << t) # control=1, target=1

    # Convert to 1-based for Julia indexing
    i10 += 1
    i11 += 1

    # Swap amplitudes for the control=1 branch
    @inbounds begin
        α = ψ[i10]
        β = ψ[i11]
        ψ[i10] = β
        ψ[i11] = α
    end
    return
end

# Host-side CNOT launcher

function apply_cnot!(
    ψ::ROCArray{ComplexF64, 1},
    control::Int, target::Int, n_qubits::Int,
    groupsize::Int = 256 # 4 wavefronts by 64
)
    @assert 0 <= control < n_qubits "control must be in 0..n_qubits-1"
    @assert 0 <= target < n_qubits "target must be in 0..n_qubits-1"
    @assert control != target "control and target must be different"
    @assert n_qubits >= 2 "CNOT needs at leas 2 qubits"

    n_groups4 = Int(1) << (n_qubits - 2) # numbxer of groups-of-four
    n_groups = cld(n_groups4, groupsize)

    @roc groupsize=groupsize gridsize=n_groups apply_cnot_kernel!(
        ψ, control, target, n_qubits
    )
    AMDGPU.synchronize()
    return ψ
end

# Bell state test

# Build |0..0⟩ on GPU
ψ_bell = gpu_zero_state(N_QUBITS)

# Apply H on qubit 1 (0-based, LSB=0)
apply_gate!(ψ_bell, GATE_H, 1, N_QUBITS)

# Apply CNOT with control=1, target=0
apply_cnot!(ψ_bell, 1, 0, N_QUBITS)

# Check normalisation on GPU
norm2_bell = sum(abs2.(ψ_bell))
ψ_bell_cpu = Array(ψ_bell)

println("Applied H(1) then CNOT(1→0) to |0..0⟩")
println("Type (GPU): ", typeof(ψ_bell))
println("Normalisation (GPU): Σ|ψ|² = ", norm2_bell)
println("State (CPU view):", ψ_bell_cpu)

amp_expected = inv(sqrt(Float64(2)))
println(@sprintf("Expected non-zeros at indices 1 and 4 ≈ %.6f + 0.0im", amp_expected))


# GPU → probabilities (CPU vector)

# Compute probabilities |ψ[i]|² on the GPU (via broadcasting), then transfer to CPU.
# Ensure the result sums to ~1.0 (renormalizes if needed due to rounding)

function gpu_probabilities(ψ::ROCArray{ComplexF64, 1})::Vector{Float64}
    # Elementwise on gpu_probabilities
    p_gpu = abs2.(ψ) # ROCArray{Float64, 1}
    s = sum(p_gpu) # scalar reduction, host Float64
    p = Array(p_gpu) # pull to cpu

    # Renormalise if off by numerical noise
    if !(isfinite(s)) || s <= 0
        error("Invalid state: sum(|ψ|²) = $s")
    end
    if abs(s - 1.0) > 1e-12
        @inbounds for i in eachindex(p)
            p[i] /= s
        end
    end
    return p
end

# Single-shot measurement (CPU-sampling)

"""
    index_to_bitstring(idx0::Int, n_qubits::Int) -> String

Render a 0-based outcome as an n-bit string with MSB on the left
{q_{n-1} ... q_0}.
"""
function index_to_bitstring(idx0::Int, n_qubits::Int)::String
    @assert 0 <= idx0 < (1 << n_qubits)
    io = IOBuffer()
    for q in (n_qubits - 1):-1:0 # MSB ... LSB
        bit = (idx0 >> q) & 0x1
        write(io, bit == 1 ? '1' : '0')
    end
    return String(take!(io))
end

"""
    measure(ψ::ROCArray{ComplexF64,1}, n_qubits::Int;
    rng=Random.GLOBAL_RNG) -> (outcome0::Int, bits::String)
"""
function measure(ψ::ROCArray{ComplexF64,1}, n_qubits::Int; rng=Random.GLOBAL_RNG)
    probs = gpu_probabilities(ψ)
    cdf = cumsum(probs) # guard against r slightly > last due to FP
    r = rand(rng)
    idx1 = searchsortedfirst(cdf, r) # 1-based index
    outcome0 = idx1 - 1
    return outcome0, index_to_bitstring(outcome0, n_qubits)
end

# Many shots (no collapse)

"""
    measure_many(ψ::ROCArray{ComplexF64,1}, n_qubits::Int,
        n_shots::Int; rng=Random.GLOBAL_RNG) -> Dict{Int, Int}

Sample outcomes n_shots times from the same state distribution (no_collapse).
Returns counds per outcome (0-based).
"""
function measure_many(ψ::ROCArray{ComplexF64,1}, n_qubits::Int,
    n_shots::Int; rng=Random.GLOBAL_RNG)
    @assert n_shots >= 1
    probs = gpu_probabilities(ψ)
    cdf = cumsum(probs)
    cdf[end] = 1.0

    counts = Dict{Int, Int}()
    @inbounds for _ in 1:n_shots
        r = rand(rng)
        idx1 = searchsortedfirst(cdf, r) # 1-based
        k = idx1 - 1 # 0-based
        counts[k] = get(counts, k, 0) + 1
    end
    return counts
end

"""
    print_hist(counts::Dict{Int, Int}, n_qubits::Int; max_bars::Int=16)

Pretty-print a small histogram: binary label, count, percentage, and an ASCII bar.
"""
function print_hist(counts::Dict{Int, Int}, n_qubits::Int;
    max_bars::Int=16)
    total = sum(values(counts))
    pairs = collect(counts)
    sort!(pairs; by = first) # sort by outcome ascending

    # Optionally limit the number of lines shown (show top by count if too many)
    if length(pairs) > max_bars
        sort!(pairs; by = x -> -x[2])
        pairs = pairs[1:max_bars]
        sort!(pairs; by = first)
        println("(showing top $max_bars of $(length(counts))outcomes)")
    end

    for (k, cnt) in pairs
        pct = 100 * cnt / total
        bits = index_to_bitstring(k, n_qubits)
        barlen = clamp(round(Int, pct / 2), 0, 50) # up to 50 chars
        bar = repeat('#', barlen)
        @printf("%s : %6d (%6.2f%%) %s\n", bits, cnt, pct, bar)
    end
end

# Bell state measurement test

# Build Bell: H on qubit 1, then CNOT(1→0) on |0000⟩
ψ_bell = gpu_zero_state(N_QUBITS)
apply_gate!(ψ_bell, GATE_H, 1, N_QUBITS)
apply_cnot!(ψ_bell, 1, 0, N_QUBITS)

# Quick single-shot demo
k, bits = measure(ψ_bell, N_QUBITS)
println("Single-shot outcome: k=$k (bits=$bits)")

# Many shots
n_shots = 10_000
counts = measure_many(ψ_bell, N_QUBITS, n_shots)

println(@sprintf("Histogram over %d shots (expect ~50%% at 0000 and ~50%% at 0011):", n_shots))
print_hist(counts, N_QUBITS)

# Sanity print the two expected bins if present
println("\nCounts for expected outcomes:")
println("   0000 (k=0): ", get(counts, 0, 0))
println("   0011 (k=3): ", get(counts, 3, 0))