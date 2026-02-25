## We simulate a discrete-time quantum walk on the same 5-node line graph.
# KEY DIFFERENCE FROM STEG 1:
# Classical: probability vector p ∈ ℝ⁵ (non-negative, sums to 1)
# evolved by stochastic matrix T
# Quantum: amplitude vector Ψ ∈ ℂ¹⁰ (complex, squared magnitudes sum to 1)
# evolved by unitary operator U

# The extra dimension (10 instead of 5) comes from the COIN space.
# Hilbert space = Coin ̇⊗ Position = ℂ² ⊗ C⁵ = ℂ¹⁰

# Basic ordering:
# |0⟩|1⟩, |0⟩|2⟩, |0⟩|3⟩, |0⟩|4⟩, |0⟩|5⟩, ← coin = |0⟩ (move left)
# |1⟩|1⟩, |1⟩|2⟩, |1⟩|3⟩, |1⟩|4⟩, |1⟩|5⟩, ← coin = |1⟩ (move right)

using LinearAlgebra
using UnicodePlots

const N_NODES = 5
const DIM = 2 * N_NODES # coin(2) × position(5) = 10

## BUILD THE COIN OPERATOR

# The Hadamard gate acts on the coin qubit:
# H = (1√2) *   [1  1]
#               [1 -1]

# This is the quantum analogue of flipping a fair coin.
# BUT - and this is crucial - instead of randomly choosing left or right,
# it creates a SUPERPOSITION.

# H|0⟩ = (|0⟩ + |1⟩)/√2 - both left AND right smultaneously
# H|1⟩ = (|0⟩ + |1⟩)/√2 - note the MINUS sign! This is where interference come from

# The full coin operator acts as H ⊗ I₅ (Hadamard on coin, identity on position):

function build_coin_operator(n::Int)
    ## Hadamard gate
    H = (1.0 / sqrt(2.0)) * [1.0 1.0; 1.0 -1.0]

    # This applies H to the coin while leaving position untouched.
    C = kron(H, Matrix{Float64}(I, n ,n))

    return C
end

## BULD THE SHIFT OPERATOR

# The conditional shift moves the walker based on the coin state:
#   If coin = |0⟩: move LEFT (position j → j-1)
#   If coin = |1⟩: move RIGHT (position j → j+1)

# At boundaries, we REFLECT the coin state to maintain unitarity

# Boundary behaviour:
#   Node 1 + coin |0⟩, (wants go left, can't):
#       → stay at node 1, flip coin to |1⟩
#   Node 5 + coin |1⟩, (wants to go right, can't):
#       → stay at node 5, flip coin to |0⟩

# This keeps the operator unitary (U†U = I).

function build_shift_operator(n::Int)
    ## Total dimension: 2n(coin × position)
    S = zeros(Float64, 2n, 2n)

    ## Coin |0⟩ subspace: indices 1 to n (move LEFT)
    for j in 1:n
        if j == 1
            ## Boundary: can't go left, reflect coin 0→1, stay at node 1
            ## Target: coin = |1⟩, positon=1 → index n+1
            S[n + 1, 1] = 1.0
        else
            ## Interior: move left, keep coin state
            ## Target: coin = |0⟩, position=j-1 → index j-1
            S[j - 1, j] = 1.0
        end
    end

    ## Coin |1⟩ subspace: indices n+1 to 2n (move RIGHT)
    for j in 1:n
        source = n + j # index in the full 2n space
        if j == n
            ## Boundary: can't go right, reflect coin 1→0, stay at noden
            ## Target: coin = |0⟩, position=n → index n
            S[n, source] = 1.0
        else
            ## Interior: move right, keep coin state
            ## Target: coin=|1⟩, position=j+1 → index n+j+1
            S[n + j + 1, source] = 1.0
        end
    end
    
    return S
end

## COMPOSE THE WALK OPERATOR

# One step of the quantum walk:
#   U = S ⋅ C

# First apply the coin (Hadamard), then the conditional shift.
# This is analogous to: "flip coin, then move" - but in superposition!

C = build_coin_operator(N_NODES)
S = build_shift_operator(N_NODES)
U = S * C # One step: coin then shift

## VERIFY UNITARITY

# A unitarity matrix satisfies U†U = I.
# This is the quantum equivalent of "probability is conserved"

# Compare with Steg 1 where we checked column sums = 1.

println("=== UNITARITY CHECK ===")
UdagU = U' * U
deviation = norm(UdagU - I(DIM))
println("∥U†U - I∥ = ", round(deviation, digits=12))
println("Unitary?", deviation < 1e-10)
println()

## INITIAL STATE

# We start the walker at node 3 (centre), same as the classicl case.
# But we must also choose a coin state!

# We use coin state |0⟩ for simplicity.
# Initial state: |0⟩|3⟩ - coin=left, position=centre

# In our basis ordering, this is index 3 (coin |0⟩, position 3).

# NOTE: the choice of initial coin state affects the walk's behaviour!
# Unlike the classical case where initial conditions
# wash out, quantum walks remember their initial conditions forever.

ψ0 = zeros(ComplexF64, DIM)
ψ0[3] = 1.0 + 0.0im # |0⟩|3⟩ - coin left, position centre

println("=== INITIAL STATE ===")
println("ψ0 = |0⟩|3⟩ (coin=left, position=centre)")
println("Norm: ∥ψ0∥ = ", norm(ψ0))
println()

## TIME EVOLUTION

# Quantum evolution: ψ(t+1) = U ⋅ ψ(t)
# Same structure as classical p(t+1) = T ⋅ p(t), but with 0
# unitary matrix instead of stochastic, and amplitudes instead
# of probabilities.

# To extract probabilities from amplitudes:
#   P(node j) = |(0|j|ψ)|² + |(1|j|ψ)|²

# We sum over the coin states (trace out the coin).

const N_STEPS = 20

# Store probability history for each node
prob_history = zeros(Float64, N_NODES, N_STEPS + 1)

# Function to extract position probabilities from full state vector
function position_probabilities(ψ::Vector{ComplexF64}, n::Int)
    probs = zeros(Float64, n)
    for j in 1:n
        ## |(0|j|ψ)|² - amplitude for coin=0, position=j
        amp_left = ψ[j]

        ## |(1|j|ψ)|² - amplitude for coin=1, positoin=j
        amp_right = ψ[n + 1]

        ## Total probability at node j: sum over coin states
        probs[j] = abs2(amp_left) + abs2(amp_right)
    end
    return probs
end

# Record initial probabilities
prob_history[:, 1] = position_probabilities(ψ0, N_NODES)

# Evolve the quantum state
ψ = copy(ψ0)
for t in 1:N_STEPS
    # The core operation: one sep of the quantum walk
    # This is UNITARY evolution - amplitudes can interfere!
    global ψ = U * ψ
    prob_history[:, t + 1] = position_probabilities(ψ, N_NODES)
end

## VERIFY PROBABILITY CONSERVATION

# At every time step, total probability must equal 1.

# Classical: guaranteed by column sums = 1
# Quantum: guaranteed by unitarity (U†U = I)

println("=== PROBABILITY CONSERVATION ===")
for t in [0, 5, 10, 20]
    total = sum(prob_history[:, t + 1])
    println(" Step $t: Σ P(j) = ", round(total, digits=10))
end
println()

## VISUALISATION

snapshopts = [0, 1, 2, 5, 10, 20]

for t in snapshopts
    plt = barplot(
        ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"],
        prob_history[:, t + 1],
        title="QUANTUM: Probability at step $t",
        xlim=(0, 1.0),
        width=50
    )
    println(plt)
    println()
end

## TIME SERIES

# This is the quantum counterpart of the convergence plot from Steg 1.
# But here, the curves DON'T converge - they oscillate!

steps = 0:N_STEPS

plt_quantum = lineplot(
    steps, prob_history[1, :],
    title="Quantum Walk: Oscilation",
    xlabel="Time step",
    name="Node 1",
    width=50,
    height=15,
    ylim=(0, 1.0)
)

for node in 2:N_NODES
    lineplot!(plt_quantum, steps, prob_history[node, :], name="Node #node")
end

println(plt_quantum)
println()

## INTERFERENCE ANALYSIS

# Let's look at the RAW AMPLITUDES at one node to see interference.
# This is something that has NO classical analogue.
# We'll track the amplitude (real and imaginary parts) at node 3

println("=== AMPLITUDE ANALYSIS AT NODE 3 ===")
println()
println("Step | Coin=|0⟩ amplitude | Coin=|1⟩ amplitude | P(node 3)")
println("-----|---------------------|---------------------|----------")

# Re-run evolution to capture amplitudes
ψ_track = copy(ψ0)
for t in 0:N_STEPS
    amp0 = ψ_track[3]   # coin=|0⟩, position=3
    amp1 = ψ_track[N_NODES + 3] # coin=|1⟩, position=3
    prob = abs2(amp0) + abs2(amp1)

    # Format complex numbers for display
    a0_str = string(round(real(amp0), digits=3), 
                    (imag(amp0) >= 0 ? "+" : ""),
                    round(imag(amp0), digits=3), "i")

    a1_str = string(round(real(amp1), digits=3), 
                    (imag(amp1) >= 0 ? "+" : ""),
                    round(imag(amp1), digits=3), "i")    

    println(" $( lpad(string(t), 2)) | $(rpad(a0_str, 18)) 
    | $(rpad(a1_str,18)) | $(round(prob, digits=4))")

    if t < N_STEPS
       global ψ_track = U * ψ_track
    end
end
println()

## --------------------------------------------------------------------------
## 11. KEY TAKEAWAY — VIKTIGASTE INSIKTEN
## --------------------------------------------------------------------------
println("=== SUMMARY / SAMMANFATTNING ===")
println()
println("CLASSICAL (Steg 1):                    QUANTUM (Steg 2):")
println("  • Probabilities (non-negative)         • Amplitudes (complex)")
println("  • Stochastic matrix T                  • Unitary matrix U")
println("  • CONVERGES to stationary π            • OSCILLATES forever")
println("  • Information is lost (mixing)          • Information is preserved")
println("  • No interference possible              • Interference is key!")
println()
println("Klassiskt:                              Kvant:")
println("  • Sannolikheter (icke-negativa)        • Amplituder (komplexa)")
println("  • Stokastisk matris T                  • Unitär matris U")
println("  • KONVERGERAR mot stationär π          • OSCILLERAR för evigt")
println("  • Information förloras (blandning)     • Information bevaras")
println("  • Ingen interferens möjlig              • Interferens är nyckeln!")
println()
println("The oscillation is NOT noise — it's a feature!")
println("Oscillationen är INTE brus — det är en egenskap!")
println()
println("Next: Steg 3 — Side-by-side comparison (Jämförelse sida vid sida)")