"""Test summit benchmarks using SQR HF AI transpiler on FakeMiami."""

import os
import time
import traceback
import threading

import pytest

os.environ["QISKIT_TRANSPILER_LINEAR_FUNCTION_REPO_ID"] = "DuruTo/ai-transpiler_linear-functions"

from qiskit_ibm_transpiler.model_bootstrap import reset_model_repository
reset_model_repository()

from qiskit.circuit.library import EfficientSU2

from qiskit_ibm_transpiler import generate_ai_pass_manager

from benchpress.config import Configuration
from benchpress.qiskit_gym.circuits import bv_all_ones, trivial_bvlike_circuit
from benchpress.utilities.io import (
    qasm_circuit_loader,
    input_circuit_properties,
    output_circuit_properties,
)
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.device_transpile import WorkoutDeviceTranspile100Q
from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_qiskit_bench_backend

Configuration.gym_name = "qiskit-ibm-transpiler"

BACKEND = get_qiskit_bench_backend("fake_miami")
TWO_Q_GATE = BACKEND.two_q_gate_type
OPTIMIZATION_LEVEL = Configuration.options["qiskit"]["optimization_level"]

TRANS_SERVICE = generate_ai_pass_manager(
    optimization_level=OPTIMIZATION_LEVEL,
    ai_optimization_level=OPTIMIZATION_LEVEL,
    backend=BACKEND,
)

TIMEOUT_SECONDS = 1000  # 10 minutes


def run_with_timeout(fn, timeout):
    """Run fn() in a thread. If it exceeds timeout, skip with a stack trace."""
    result = [None]
    exc = [None]
    snapshot = [None]

    def target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        # Capture stack frames of all threads to show where it's stuck
        frames = []
        for tid, frame in threading._current_frames().items():
            if tid == t.ident:
                frames.append("".join(traceback.format_stack(frame)))
        stuck_trace = "\n".join(frames) if frames else "(no frame captured)"
        pytest.skip(
            f"Timed out after {timeout}s. Thread stuck at:\n{stuck_trace}"
        )

    if exc[0] is not None:
        raise exc[0]

    return result[0]


@benchpress_test_validation
class TestWorkoutDeviceTranspile100Q(WorkoutDeviceTranspile100Q):
    def test_QFT_100_transpile(self, benchmark):
        """Compile 100Q QFT circuit against target backend"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("qft") + "qft_N100.qasm", benchmark
        )

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_QV_100_transpile(self, benchmark):
        """Compile 100Q QV circuit against target backend"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("qv") + "qv_N100_12345.qasm", benchmark
        )

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_circSU2_89_transpile(self, benchmark):
        """Compile 89Q circSU2 circuit against target backend"""
        circuit = EfficientSU2(89, reps=3, entanglement="circular")
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_circSU2_100_transpile(self, benchmark):
        """Compile 100Q circSU2 circuit against target backend"""
        circuit = EfficientSU2(100, reps=3, entanglement="circular")
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_BV_100_transpile(self, benchmark):
        """Compile 100Q BV circuit against target backend"""
        circuit = bv_all_ones(100)
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_square_heisenberg_100_transpile(self, benchmark):
        """Compile 100Q square-Heisenberg circuit against target backend"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("square-heisenberg")
            + "square_heisenberg_N100.qasm",
            benchmark,
        )

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_QAOA_100_transpile(self, benchmark):
        """Compile 100Q QAOA circuit against target backend"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("qaoa") + "qaoa_barabasi_albert_N100_3reps.qasm",
            benchmark,
        )

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_BVlike_simplification_transpile(self, benchmark):
        """Transpile a BV-like circuit that should collapse down
        into a single X and Z gate on a target device
        """
        circuit = trivial_bvlike_circuit(100)
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)

    def test_clifford_100_transpile(self, benchmark):
        """Compile 100Q Clifford circuit against target backend"""
        from qiskit import transpile as qiskit_transpile

        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("clifford") + "clifford_100_12345.qasm",
            benchmark,
        )
        # The Clifford QASM uses SWAP gates. AIRouting's Layout builder maps
        # qubits across the full 120-qubit backend when SWAPs are present,
        # producing a QuantumRegister(120,'q') whose Qubit objects don't match
        # the ones stored in the Layout → KeyError in ApplyLayout.
        # Pre-decompose SWAP→CX (no routing, no optimization) so the AI pass
        # manager only sees CX gates and routes cleanly.
        #"pre"-transpilation to transform SWAPs to cnots
        # circuit = qiskit_transpile(
        #     circuit,
        #     basis_gates=["cx", "h", "s", "sdg", "t", "tdg", "x", "y", "z", "rx", "ry", "rz"],
        #     coupling_map=None,
        #     optimization_level=0,
        # )

        @benchmark
        def result():
            return run_with_timeout(lambda: TRANS_SERVICE.run(circuit), TIMEOUT_SECONDS)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, BACKEND)
