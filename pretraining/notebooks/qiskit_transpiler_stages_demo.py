"""
Qiskit Transpiler Stages Demonstration
======================================

Walks a Bell-state circuit (H + CNOT) through each of the six transpilation
stages used by Qiskit's preset pass managers — init, layout, routing,
translation, optimization, scheduling — printing what each stage does and
plotting the circuit after every stage. Targets the 156-qubit Heron r2
device `ibm_marrakesh` (Heavy Hex topology); falls back to FakeMarrakesh
when no IBM Quantum credentials are available.

Concludes with a QAOA-on-MaxCut application: builds the cost Hamiltonian
for a 4-node graph, constructs a depth-1 QAOAAnsatz, transpiles it at
optimization_level=3 against the same backend, and reports the gate-count
and depth impact of transpilation.

Compatible with stable Qiskit 1.x (qiskit-ibm-runtime >= 0.25).
"""

# === IMPORTS ===
import warnings

import matplotlib.pyplot as plt
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_gate_map

print(f"Qiskit version: {qiskit.__version__}")


# === BACKEND SETUP ===
# Try the real ibm_marrakesh device; if credentials/network are unavailable,
# fall back to FakeMarrakesh, which carries the same 156-qubit Heavy Hex
# topology and native gate set ({CX/ECR, ID, RZ, SX, X}).
def get_backend():
    """Return ibm_marrakesh if credentials are configured, else FakeMarrakesh."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend("ibm_marrakesh")
        print(f"Using REAL backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except Exception as exc:
        # Any failure (no creds, network error, backend offline) -> FakeMarrakesh.
        print(f"Could not reach ibm_marrakesh ({exc.__class__.__name__}); "
              f"falling back to FakeMarrakesh.")
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        backend = FakeMarrakesh()
        print(f"Using FAKE backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend


backend = get_backend()
print(f"Native gates: {backend.operation_names}")
print(f"Coupling map edges: {len(backend.coupling_map.get_edges())}")


# === BUILD A STAGED PASS MANAGER ===
# generate_preset_pass_manager returns a StagedPassManager whose six named
# stages (init, layout, routing, translation, optimization, scheduling) can
# each be invoked individually via `pm.<stage>.run(circuit)`.
#
# We pick optimization_level=2 (good balance of cost vs. quality) and
# explicitly request ALAP scheduling so stage 6 inserts visible delays.
pm = generate_preset_pass_manager(
    optimization_level=2,
    backend=backend,
    scheduling_method="alap",   # As-Late-As-Possible: makes scheduling stage non-trivial
    seed_transpiler=42,         # Determinism for SabreLayout / SabreSwap
)

print(f"Pass-manager stages: {pm.stages}")


# === BELL STATE CIRCUIT ===
# Two qubits, one Hadamard on q0, one CNOT(q0, q1), then measure all.
# This is the simplest non-trivial entangling circuit and exercises every
# transpilation stage when targeted at a real device.
bell = QuantumCircuit(2, name="bell")
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

print("\nOriginal Bell circuit:")
print(bell)


# Helper: draw a circuit with a clear stage title.
def draw_stage(circuit, title, idle_wires=False):
    """Draw a circuit with `circuit.draw(output='mpl')` and a stage title."""
    fig = circuit.draw(output="mpl", idle_wires=idle_wires, fold=-1)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# === STAGE 1: INIT ===
# =============================================================================
# The init stage performs initial unrolling and high-level synthesis: it
# decomposes any custom gates / 3+-qubit gates down to 1- and 2-qubit gates
# (so layout and routing — which only understand 1q/2q operations — can do
# their job), and runs lightweight optimizations like inverse-cancellation.
# For our Bell circuit there are no exotic gates to unroll, so the change
# is small, but conceptually this is the canonicalization step.
print("\n" + "=" * 70)
print("STAGE 1 - INIT")
print("=" * 70)
print("Unrolls custom gates and decomposes anything wider than 2 qubits into")
print("1- and 2-qubit gates. Applies a few opening optimizations. The output")
print("is a clean, layout-ready circuit on virtual qubits.")

bell_init = pm.init.run(bell)
print(f"  ops after init       : {dict(bell_init.count_ops())}")
print(f"  depth after init     : {bell_init.depth()}")
draw_stage(bell_init, "Stage 1: Init")


# =============================================================================
# === STAGE 2: LAYOUT ===
# =============================================================================
# Layout maps the circuit's *virtual* qubits (q0, q1) onto specific *physical*
# qubits of the backend. Qiskit tries VF2 first (looking for an isomorphic
# subgraph that already satisfies all 2-qubit interactions); if that fails,
# SabreLayout heuristically picks a layout that minimizes expected SWAPs.
# After this stage, the circuit lives on a 156-qubit register, with the two
# virtual qubits parked on a connected pair of physical qubits.
print("\n" + "=" * 70)
print("STAGE 2 - LAYOUT")
print("=" * 70)
print("Assigns each virtual qubit to a specific physical qubit on the backend.")
print("The circuit is embedded into the full 156-qubit register; only the")
print("chosen physical qubits carry instructions, the rest sit idle.")

bell_layout = pm.layout.run(bell_init)
print(f"  total qubits in circuit : {bell_layout.num_qubits}")

# Identify which physical qubits the layout actually selected. Qiskit 1.x
# exposes a TranspileLayout with `initial_index_layout()`, which returns a
# list whose i-th entry is the physical qubit holding the i-th virtual qubit
# of the input. We slice off the first len(bell.qubits) entries to ignore
# any ancillas the pass added.
selected_phys_qubits = []
if bell_layout.layout is not None:
    try:
        # Preferred path: initial_index_layout (Qiskit >= 1.0) returns a
        # plain list indexed by virtual-qubit position.
        full_index = bell_layout.layout.initial_index_layout(filter_ancillas=True)
        selected_phys_qubits = list(full_index)
    except (AttributeError, TypeError):
        # Older fallback: index the Layout object by Qubit.
        init_layout = bell_layout.layout.initial_layout
        for q in bell.qubits:
            try:
                selected_phys_qubits.append(init_layout[q])
            except (KeyError, IndexError):
                pass
print(f"  selected physical qubits: {selected_phys_qubits}")

draw_stage(bell_layout, "Stage 2: Layout (virtual -> physical)")

# --- Plot the backend coupling map with selected qubits highlighted. -------
# plot_gate_map accepts a per-qubit `qubit_color` list. We color the chosen
# physical qubits red and the rest a muted grey so the layout choice pops.
n_qubits = backend.num_qubits
qubit_colors = [
    "#d62728" if i in selected_phys_qubits else "#3c3c3c"
    for i in range(n_qubits)
]
try:
    fig = plot_gate_map(
        backend,
        qubit_color=qubit_colors,
        font_size=8,
        figsize=(14, 8),
    )
    fig.suptitle(
        f"ibm_marrakesh Heavy-Hex topology (156 qubits) — "
        f"selected: {selected_phys_qubits}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
except Exception as exc:
    # plot_gate_map can be picky about non-IBM-shaped backends; fall back to
    # the generic coupling-map plot so the script still completes.
    warnings.warn(f"plot_gate_map failed ({exc}); using plot_coupling_map fallback.")
    from qiskit.visualization import plot_coupling_map
    coords = [[i // 13, i % 13] for i in range(n_qubits)]  # rough 12x13 grid
    fig = plot_coupling_map(
        num_qubits=n_qubits,
        qubit_coordinates=coords,
        coupling_map=backend.coupling_map.get_edges(),
        qubit_color=qubit_colors,
    )
    plt.title(f"Coupling map — selected qubits: {selected_phys_qubits}")
    plt.tight_layout()
    plt.show()


# =============================================================================
# === STAGE 3: ROUTING ===
# =============================================================================
# Routing inserts SWAP gates wherever the layout has placed two interacting
# virtual qubits onto non-adjacent physical qubits. For our Bell circuit the
# two virtual qubits were placed on an adjacent pair (the CNOT only spans
# one edge), so SabreSwap typically inserts ZERO swaps here. On larger,
# more entangled circuits this is where most depth comes from.
print("\n" + "=" * 70)
print("STAGE 3 - ROUTING")
print("=" * 70)
print("Inserts SWAP gates so every 2-qubit gate acts on physically connected")
print("qubits. For a 2-qubit Bell circuit on adjacent qubits, no SWAPs are")
print("needed; for deeper circuits this stage usually adds the most depth.")

bell_routing = pm.routing.run(bell_layout)
ops_routing = dict(bell_routing.count_ops())
print(f"  ops after routing    : {ops_routing}")
print(f"  swap gates inserted  : {ops_routing.get('swap', 0)}")
print(f"  depth after routing  : {bell_routing.depth()}")
draw_stage(bell_routing, "Stage 3: Routing")


# =============================================================================
# === STAGE 4: TRANSLATION ===
# =============================================================================
# Translation rewrites every gate in terms of the backend's *native* gate
# set. Heron r2 devices natively support {ECR (or CZ), ID, RZ, SX, X} —
# anything else has to be expressed as a sequence of these. The Hadamard
# on q0 becomes a short SX/RZ sequence, and the CNOT is rewritten using
# the native two-qubit gate plus single-qubit dressing.
print("\n" + "=" * 70)
print("STAGE 4 - TRANSLATION")
print("=" * 70)
print("Rewrites every gate using only the backend's native basis set")
print(f"(here: {sorted(backend.operation_names)}). After this stage the")
print("circuit is executable in principle, just not yet optimized.")

bell_translation = pm.translation.run(bell_routing)
print(f"  ops after translation: {dict(bell_translation.count_ops())}")
print(f"  depth after trans.   : {bell_translation.depth()}")
draw_stage(bell_translation, "Stage 4: Translation (-> native basis)")


# =============================================================================
# === STAGE 5: OPTIMIZATION ===
# =============================================================================
# The optimization stage runs a fixed-point loop over passes such as
# Optimize1qGatesDecomposition, CommutativeCancellation, InverseCancellation,
# and (at higher levels) 2-qubit unitary resynthesis. The goal is to reduce
# both gate count and depth — particularly the count of expensive 2-qubit
# gates, which dominate error budgets on real hardware.
print("\n" + "=" * 70)
print("STAGE 5 - OPTIMIZATION")
print("=" * 70)
print("Cancels redundant gates, fuses adjacent single-qubit rotations, and")
print("resynthesizes 2-qubit blocks. Same logical operation, fewer/cheaper")
print("physical gates — directly lowers the total error per shot.")

bell_optimization = pm.optimization.run(bell_translation)
print(f"  ops after optimization: {dict(bell_optimization.count_ops())}")
print(f"  depth after opt.      : {bell_optimization.depth()}")
draw_stage(bell_optimization, "Stage 5: Optimization")


# =============================================================================
# === STAGE 6: SCHEDULING ===
# =============================================================================
# Scheduling assigns a concrete start time (in dt units) to every gate.
# ALAP (As-Late-As-Possible) inserts Delay instructions so each gate fires
# as late as it can without violating data dependencies — a useful starting
# point for dynamical-decoupling insertion. After this stage the circuit
# is fully timed and ready to be packaged for the control electronics.
print("\n" + "=" * 70)
print("STAGE 6 - SCHEDULING")
print("=" * 70)
print("Assigns a concrete start time to every instruction (ALAP here) and")
print("inserts Delay gates so qubits idle in lockstep. The output is the")
print("final, hardware-ready, fully-timed circuit.")

bell_scheduling = pm.scheduling.run(bell_optimization)
print(f"  ops after scheduling : {dict(bell_scheduling.count_ops())}")
print(f"  depth after sched.   : {bell_scheduling.depth()}")
# The scheduled circuit has Delay gates; show them.
draw_stage(bell_scheduling, "Stage 6: Scheduling (with delays)")


# =============================================================================
# === QAOA + MAXCUT APPLICATION ===
# =============================================================================
# Now we exercise the same pipeline on something with non-trivial structure.
# MaxCut on a 4-node cycle graph: partition vertices into two sets so the
# number of edges crossing the partition is maximized. On a 4-cycle the
# optimal cut has size 4 (alternate vertices in each partition).
#
# Encoding: each vertex is a qubit; the cost Hamiltonian is
#    H_C = (1/2) * sum over edges (i,j) of (I - Z_i Z_j)
# whose ground state encodes the optimal cut. We drop the constant offset
# and feed the ZZ terms to QAOAAnsatz.
print("\n" + "=" * 70)
print("QAOA + MAXCUT APPLICATION")
print("=" * 70)

# 4-node cycle: edges 0-1, 1-2, 2-3, 3-0
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
n_nodes = 4

# Build the cost Hamiltonian as a SparsePauliOp on n_nodes qubits.
# Qiskit's Pauli string convention: rightmost char is qubit 0.
def maxcut_hamiltonian(num_qubits, edge_list):
    """Return SparsePauliOp = sum_(i,j) Z_i Z_j over edges (constant dropped)."""
    pauli_terms = []
    for (i, j) in edge_list:
        chars = ["I"] * num_qubits
        chars[num_qubits - 1 - i] = "Z"   # rightmost = qubit 0
        chars[num_qubits - 1 - j] = "Z"
        pauli_terms.append(("".join(chars), 1.0))
    return SparsePauliOp.from_list(pauli_terms)


cost_hamiltonian = maxcut_hamiltonian(n_nodes, edges)
print(f"Cost Hamiltonian:\n{cost_hamiltonian}")

# Build a depth-1 QAOA ansatz: alternates exp(-i*gamma*H_C) and exp(-i*beta*H_M).
qaoa_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=1)
qaoa_circuit.measure_all()

# QAOAAnsatz returns a parameterized circuit; bind concrete (random-ish)
# values for plotting purposes. The structure is parameter-free after this.
rng = np.random.default_rng(0)
param_values = rng.uniform(0, 2 * np.pi, size=qaoa_circuit.num_parameters)
qaoa_bound = qaoa_circuit.assign_parameters(param_values)

print(f"\nQAOA ansatz: {qaoa_circuit.num_qubits} qubits, "
      f"reps=1, parameters={qaoa_circuit.num_parameters}")

# Plot the abstract MaxCut graph (no networkx dependency — quick matplotlib).
fig, ax = plt.subplots(figsize=(5, 5))
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) + np.pi / 2
node_xy = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
for (i, j) in edges:
    x = [node_xy[i][0], node_xy[j][0]]
    y = [node_xy[i][1], node_xy[j][1]]
    ax.plot(x, y, "-", color="#666", linewidth=2, zorder=1)
for i, (x, y) in node_xy.items():
    ax.scatter([x], [y], s=900, color="#1f77b4", zorder=2, edgecolors="black")
    ax.text(x, y, str(i), ha="center", va="center", color="white",
            fontsize=14, fontweight="bold", zorder=3)
ax.set_aspect("equal")
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.axis("off")
ax.set_title("MaxCut problem: 4-node cycle graph", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Plot the original (untranspiled) QAOA circuit.
fig = qaoa_bound.decompose().draw(output="mpl", fold=-1)
fig.suptitle("QAOA Ansatz (original, decomposed)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# --- Transpile against the backend at optimization_level=3 ---
# Level 3 is the most aggressive: heavier layout search, 2-qubit unitary
# resynthesis, more passes. Slower, but produces the tightest circuits.
qaoa_pm = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    seed_transpiler=42,
)
qaoa_transpiled = qaoa_pm.run(qaoa_bound)

# Plot the transpiled QAOA circuit. With ~150 idle qubits, suppress them.
fig = qaoa_transpiled.draw(output="mpl", idle_wires=False, fold=-1)
fig.suptitle("QAOA Ansatz (transpiled, optimization_level=3)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# --- Print before-vs-after impact ---
print("\n--- Transpilation impact (QAOA, optimization_level=3) ---")
print(f"{'metric':<20} {'before':>10} {'after':>10}")
print("-" * 42)
print(f"{'num_qubits':<20} {qaoa_bound.num_qubits:>10} "
      f"{qaoa_transpiled.num_qubits:>10}")
print(f"{'depth':<20} {qaoa_bound.decompose().depth():>10} "
      f"{qaoa_transpiled.depth():>10}")
print(f"{'total gates':<20} {sum(qaoa_bound.decompose().count_ops().values()):>10} "
      f"{sum(qaoa_transpiled.count_ops().values()):>10}")

# Per-gate breakdown
ops_before = qaoa_bound.decompose().count_ops()
ops_after = qaoa_transpiled.count_ops()
all_gates = sorted(set(ops_before) | set(ops_after))
print("\nPer-gate counts:")
print(f"{'gate':<12} {'before':>10} {'after':>10}")
print("-" * 34)
for g in all_gates:
    print(f"{g:<12} {ops_before.get(g, 0):>10} {ops_after.get(g, 0):>10}")

print("\nDone.")