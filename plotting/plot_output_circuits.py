"""
Draw the output circuit diagram for each summit benchmark after AI transpilation.
Saves one PNG per circuit to results/circuit_<name>_<method>_output.png.

Usage:
    python plotting/plot_output_circuits.py --method sqr               # SQR HF AI (default)
    python plotting/plot_output_circuits.py --method qiskit_hf         # Qiskit HF AI
    python plotting/plot_output_circuits.py --circuit qft --method sqr # single circuit
    python plotting/plot_output_circuits.py --fold 30                  # columns per row
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

QASM_BASE = os.path.join(os.path.dirname(__file__), "..", "benchpress", "qasm")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")


def _load_qasm(rel_path):
    from qiskit.qasm2 import load as qasm2_load, LEGACY_CUSTOM_INSTRUCTIONS, LEGACY_INCLUDE_PATH
    return qasm2_load(
        os.path.join(QASM_BASE, rel_path),
        include_path=LEGACY_INCLUDE_PATH,
        custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS,
    )


# (display_name, callable returning input circuit)
CIRCUITS = {
    "qft":        ("QFT-100",        lambda: _load_qasm("qft/qft_N100.qasm")),
    "qv":         ("QV-100",         lambda: _load_qasm("qv/qv_N100_12345.qasm")),
    "su2_100":    ("SU2-100",        lambda: __import__("qiskit.circuit.library", fromlist=["EfficientSU2"]).EfficientSU2(100, reps=3, entanglement="circular")),
    "su2_89":     ("SU2-89",         lambda: __import__("qiskit.circuit.library", fromlist=["EfficientSU2"]).EfficientSU2(89,  reps=3, entanglement="circular")),
    "bv":         ("BV-100",         lambda: __import__("benchpress.qiskit_gym.circuits", fromlist=["bv_all_ones"]).bv_all_ones(100)),
    "heisenberg": ("Heisenberg-100", lambda: _load_qasm("square-heisenberg/square_heisenberg_N100.qasm")),
    "qaoa":       ("QAOA-100",       lambda: _load_qasm("qaoa/qaoa_barabasi_albert_N100_3reps.qasm")),
    "bvlike":     ("BVlike-100",     lambda: __import__("benchpress.qiskit_gym.circuits", fromlist=["trivial_bvlike_circuit"]).trivial_bvlike_circuit(100)),
    "clifford":   ("Clifford-100",   lambda: _load_qasm("clifford/clifford_100_12345.qasm")),
}


def build_sqr_transpiler(backend):
    os.environ["QISKIT_TRANSPILER_LINEAR_FUNCTION_REPO_ID"] = "DuruTo/ai-transpiler_linear-functions"
    from qiskit_ibm_transpiler.model_bootstrap import reset_model_repository
    reset_model_repository()
    from qiskit_ibm_transpiler import generate_ai_pass_manager
    from benchpress.config import Configuration
    opt = Configuration.options["qiskit"]["optimization_level"]
    return generate_ai_pass_manager(
        optimization_level=opt,
        ai_optimization_level=opt,
        backend=backend,
        ai_layout_mode="keep",
    )


def build_qiskit_hf_transpiler(backend):
    from qiskit_ibm_transpiler import generate_ai_pass_manager
    return generate_ai_pass_manager(
        coupling_map=backend.coupling_map,
        ai_optimization_level=2,
        optimization_level=2, #test 2 and 3, run  both
        ai_layout_mode="keep", #use keep only for clifford
    )


def prepare_input(key):
    circuit = CIRCUITS[key][1]()
    if key == "clifford":
        # Pre-decompose SWAP→CX; same workaround as test_summit_sqr.py
        from qiskit import transpile
        circuit = transpile(
            circuit,
            basis_gates=["cx", "h", "s", "sdg", "t", "tdg", "x", "y", "z", "rx", "ry", "rz"],
            coupling_map=None,
            optimization_level=0,
        )
    return circuit


def draw_and_save(circuit, title, out_path, fold):
    import matplotlib.pyplot as plt
    fig = circuit.draw("mpl", fold=fold, style="clifford")
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sqr", "qiskit_hf"], default="sqr",
                        help="Which AI transpiler to use (default: sqr)")
    parser.add_argument("--circuit", choices=list(CIRCUITS), default=None,
                        help="Draw a single circuit (default: all)")
    parser.add_argument("--fold", type=int, default=40,
                        help="Gate columns per row in diagram (default: 40)")
    args = parser.parse_args()

    from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_qiskit_bench_backend
    backend = get_qiskit_bench_backend("fake_miami")

    print(f"Setting up {args.method} transpiler...")
    if args.method == "sqr":
        trans_service = build_sqr_transpiler(backend)
        method_label = "SQR HF AI"
    else:
        trans_service = build_qiskit_hf_transpiler(backend)
        method_label = "Qiskit HF AI"

    keys = [args.circuit] if args.circuit else list(CIRCUITS)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    for key in keys:
        label = CIRCUITS[key][0]
        print(f"\n[{key}] Loading input circuit...")
        circuit = prepare_input(key)

        print(f"[{key}] Transpiling with {method_label}...")
        result = trans_service.run(circuit)

        ops = result.count_ops()
        cx = ops.get("cx", ops.get("ecr", ops.get("cz", 0)))
        total = sum(ops.values())
        print(f"[{key}] Done — {result.num_qubits}q, {total} gates total, {cx} 2Q gates, depth={result.depth()}")

        out_path = os.path.join(RESULTS_PATH, f"circuit_{key}_{args.method}_output.png")
        title = f"{label} — {method_label} | {total} gates | {cx} 2Q | depth {result.depth()}"
        print(f"[{key}] Drawing (fold={args.fold})...")
        draw_and_save(result, title, out_path, args.fold)
        print(f"[{key}] Saved → {out_path}")


if __name__ == "__main__":
    main()
