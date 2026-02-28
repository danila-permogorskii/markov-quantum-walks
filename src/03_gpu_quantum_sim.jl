using AMDGPU
using AMDGPU: @roc
using LinearAlgebra

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