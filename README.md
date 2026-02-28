# Quantum Walks & GPU-Accelerated Quantum Simulation

An open research project exploring quantum algorithms — from classical Markov chains to GPU-accelerated quantum circuit simulation on AMD Instinct MI300X.

## What is this?

This repository documents a hands-on learning journey through quantum computing fundamentals, implemented entirely in Julia. It starts with classical random walks, transitions to quantum walks, and builds up to a custom GPU-accelerated state-vector quantum circuit simulator running on AMD hardware.

The code is heavily commented — each file is designed to be read as a tutorial, not just executed.

## Why Julia on AMD GPUs?

There is currently no ready-made Julia quantum simulation package with an AMD GPU backend. Existing frameworks like Yao.jl offer CUDA-only GPU support (CuYao.jl), while AMDGPU.jl provides the low-level GPU programming interface but no quantum-specific tooling. This project bridges that gap by building quantum simulation primitives directly on top of AMDGPU.jl with custom ROCm kernels.

## Current capabilities

- **Classical Markov chains** — transition matrices, stationary distributions, eigenvalue analysis
- **Discrete-time quantum walks** — Hadamard coin, conditional shift, boundary reflection, unitarity verification
- **GPU quantum circuit simulator** — custom `@roc` kernels for single-qubit gates (H, X, Z) and CNOT, Bell state generation, multi-shot measurement, benchmarked up to 28 qubits (4 GiB state vector) achieving ~66% of MI300X peak HBM3 bandwidth

## Hardware

GPU simulation runs on an **AMD Instinct MI300X** (192 GiB HBM3, 5.3 TB/s bandwidth, CDNA 3 architecture) provisioned via [DigitalOcean GPU Droplets](https://www.digitalocean.com/products/gpu-droplets). The [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) programme offers complimentary MI300X hours for qualified developers — a practical way to experiment with high-end GPU hardware at zero cost.

## Roadmap

- [ ] Push simulation to 30+ qubits (16–128 GiB state vectors)
- [ ] Add parameterised rotation gates (Rx, Ry, Rz) for variational circuits
- [ ] Implement Grover's search algorithm on GPU
- [ ] Benchmark against Yao.jl CPU baseline
- [ ] Explore Yao.jl + ROCArray integration (experimental)
- [ ] Variational quantum circuits (VQE, QAOA) targeting IonQ cloud simulators
- [ ] Quantum kernel methods for classification

## References

- Scott Aaronson, *Quantum Computing Since Democritus* — complexity theory and conceptual foundations
- John Watrous, *The Theory of Quantum Information* — rigorous mathematical treatment
- Maria Schuld, *Supervised Learning with Quantum Computers* — quantum ML algorithms

## Requirements

- Julia 1.12+ (MI300X/gfx942 requires Julia 1.12 for LLVM support)
- ROCm 6.x
- AMDGPU.jl, LinearAlgebra.jl, HTTP.jl, JSON3.jl

## Licence

MIT
