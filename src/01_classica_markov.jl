using LinearAlgebra
using UnicodePlots

const N_NODES = 5

function build_transition_matrix(n::Int)
    # Create a zero matrix
    T = zeros(Float64, n, n)

    for j in 1:n
        if j == 1
            ## Left boundary: stary or go right
            T[1, 1] = 0.5 # stay at node 1
            T[2, 1] = 0.5 # move to node 2
        elseif j == n
            ## Right boundary: go left or stay
            T[n-1, n] = 0.5 # move to node n-1
            T[n, n] = 0.5   # stay at node n
        else
            ## Interior node: go left or right with equal probability
            T[j-1, j] = 0.5 # move left
            T[j+1, j] = 0.5 # move right
        end
    end

    return T
end

T = build_transition_matrix(N_NODES)

## A valid stochastic matrix must have columns that sum to 1.

println("=== Transition matrix ===")
display(T)
println()

column_sums = sum(T, dims=1)
println("Column sums: ", vec(column_sums))
println("All columns sum to 1? ", all(≈(1.0), column_sums))

## Is T also doubly stochastic?
## If yes, the stationary distribution will be uniform.
row_sums = sum(T, dims=2)
println("Row sums: ", vec(row_sums))
println("Doubly stochastic? ", all(≈(1.0), row_sums))
println()

## We place the walker at node 3 (the middle) with certainty.
## This means p = [0, 0, 1, 0, 0] - all probability mass
## concetrated at a single node.

## In quantum mechanics, this corresponds to a pure state
## localised at a specific position. Soon we'll see how the
## quantum version behaves differently.

p0 = zeros(Float64, N_NODES)
p0[3] = 1.0

println("=== INITIAL STATE ===")
println("p₀ = ", p0)
println()

## The heard of the Markov chain: at each step, multiply by T.
##
## p(t) = Tᵀ · p(0)
##
## This is iterative matrix-vector multiplication.
## Each multiplilcation mixes the probability - it spreads across
## nodes until reaching equilibrium / stationary distribution

const N_STEPS = 20

## Store the full history so we can visualise it
history = zeros(Float64, N_NODES, N_STEPS + 1)
history[:, 1] = p0

p = copy(p0)
for t in 1:N_STEPS
    ## The core operation: one step of the Markov chain
    global p = T * p
    history[:, t+1] = p
end

## The stationary distribution π statisfies T * π = π.
## This means π is an eigenvector with eigenvalue λ = 1.
##
## Perron-Frobenius theorem guarantees that for a stochastic matrix
## with certain properties (irreducible, aperiodic), eigenvalue
## is the largest and the corresponding eigenvector is unique.

eigenvalues = eigvals(T)
println("=== EIGENVALUES ===")
println("λ = ", round.(real.(eigenvalues), digits=6))
println("Largest eigenvalue:", maximum(real.(eigenvalues)))
println()

## Find the stationary distribution from eigenvectors
F = eigen(T)
## Find the eigenvector corresponding to eigenvalue ≈ 1
idx = argmin(abs.(F.values .- 1.0))
π_eigen = real.(F.vectors[:, idx])
## Normalise so entries sum to 1
π_eigen = π_eigen / sum(π_eigen)

println("=== STATIONARY DISTRIBUTION ===")
println("From eigendecomposition:")
println("   π = ", round.(π_eigen, digits=6))
println("From simulation after $(N_STEPS) steps:")
println("   p($(N_STEPS)) = ", round.(history[:, end], digits=6))
println("Difference: ", round.(norm(history[:, end] - π_eigen), digits=10))

## We plot the probability distribution at several time steps
## to see how it evolves from a delta function (all mass at node 3)
## towards the uniform stationary distribution.

println("=== EVOLUTION OF PROBABILITY ===")
println()

## Show snapshopts at specific time steps
snapshots = [0, 1, 2, 5, 10, 20]

for t in snapshots
    step_label = "t=$t"
    plt = barplot(
        ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"],
        history[:, t+1],
        title="Probability at step $t",
        xlim=(0, 1.0),
        width=50
    )
    println(plt)
    println()
end

## Track how each node's probability changes over time.
## All five curves should converge to 0.2 (= 1/5).

steps = 0:N_STEPS

plt_conv = lineplot(
    steps, history[1, :],
    title="Convergence to Stationary Distribution",
    xlabel="Time step",
    ylabel="Probability",
    name="Node 1",
    width=60,
    height=15,
    ylim=(0, 1.0)
)

for node in 2:N_NODES
    lineplot!(plt_conv, steps, history[node, :], name="Node $node")
end

println(plt_conv)
println()

println("=== SUMMARY ===")
println("After $(N_STEPS) steps, the probability has converged to approximately:")
println("   p = ", round.(history[:, end], digits=4))
println("The theoretical stationary distribution is:")
println("   π = ", round.(π_eigen, digits=4))
println()
println("Next: Steg 2 - Quantum Walk")
println("We'll replace T with a unitary operator U and see interference!")
