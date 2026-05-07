"""
qasm benchmarks on FakeMiami with selectable transpiler.

Run all tests and save results:

    TRANSPILER_MODE=qiskit     pytest test_qasm_miami.py --benchmark-json=results/qasm_qiskit_3.json
    TRANSPILER_MODE=qiskit_hf  pytest test_qasm_miami.py --benchmark-json=results/qasm_qiskit_hf_3.json
    TRANSPILER_MODE=sqr        pytest test_qasm_miami.py --benchmark-json=results/qasm_sqr_3.json
    TRANSPILER_MODE=all        pytest test_qasm_miami.py   # saves all three automatically

"""

import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from qiskit.circuit.library import EfficientSU2, QuantumVolume

from benchpress.config import Configuration
from benchpress.qiskit_gym.circuits import bv_all_ones, trivial_bvlike_circuit
from benchpress.utilities.io import (
    qasm_circuit_loader,
    input_circuit_properties,
    output_circuit_properties,
)
from benchpress.utilities.progress import (
    benchmark_out_dir,
    progress,
    pytest_command,
    pytest_extra_args,
    run_with_progress,
)
from benchpress.utilities.validation import circuit_validator
from benchpress.workouts.validation import benchpress_test_validation
from benchpress.workouts.device_transpile import WorkoutDeviceTranspile100Q

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
from benchpress.qiskit_gym.utils.qiskit_backend_utils import get_qiskit_bench_backend

MIAMI = get_qiskit_bench_backend("fake_miami")
TWO_Q_GATE = MIAMI.two_q_gate_type

# ---------------------------------------------------------------------------
# Transpiler selection
# ---------------------------------------------------------------------------
_SQR_REPO    = "DuruTo/ai-transpiler_linear-functions"
_QISKIT_REPO = "Qiskit/ai-transpiler_linear-functions"

TRANSPILER_MODE = os.getenv("TRANSPILER_MODE", "qiskit").lower()
_VALID_MODES = ("qiskit", "qiskit_hf", "sqr", "all")
if TRANSPILER_MODE not in _VALID_MODES:
    raise ValueError(
        f"TRANSPILER_MODE={TRANSPILER_MODE!r} is not valid. "
        f"Choose one of: {_VALID_MODES}"
    )

if TRANSPILER_MODE == "all":
    import subprocess

    _out_dir = benchmark_out_dir("results/qasm")
    _extra = pytest_extra_args(sys.argv[1:])

    for _mode in ("qiskit", "qiskit_hf", "sqr"): # for each mode it executes a pytest command
        _out_json = _out_dir / f"qasm_{_mode}.json"
        _cmd = pytest_command(__file__, _out_json, _extra)
        _env = {**os.environ, "TRANSPILER_MODE": _mode}
        print(f"\n{'='*60}\n  Mode: {_mode}  ->  {_out_json}\n{'='*60}", flush=True)
        subprocess.run(_cmd, env=_env, check=True)

    import pytest
    pytest.exit(
        f"All 3 modes done. Results in {_out_dir}/qasm_{{qiskit,qiskit_hf,sqr}}.json",
        returncode=0,
    )

OPT_LEVEL = 3

progress(f"\n[test_qasm_miami] TRANSPILER_MODE={TRANSPILER_MODE}  backend=FakeMiami")

def _make_ai_pass_managers(repo_id):
    os.environ["QISKIT_TRANSPILER_LINEAR_FUNCTION_REPO_ID"] = repo_id
    from qiskit_ibm_transpiler.model_bootstrap import reset_model_repository
    reset_model_repository()
    from qiskit_ibm_transpiler import generate_ai_pass_manager
    pm = generate_ai_pass_manager(
        optimization_level=OPT_LEVEL,
        ai_optimization_level=OPT_LEVEL,
        backend=MIAMI,
    )
    pm_clifford = generate_ai_pass_manager(
        optimization_level=OPT_LEVEL,
        ai_optimization_level=OPT_LEVEL,
        backend=MIAMI,
        ai_layout_mode="keep",
    )
    return pm, pm_clifford


if TRANSPILER_MODE == "qiskit":
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    Configuration.gym_name = "qiskit"
    PM = generate_preset_pass_manager(OPT_LEVEL, MIAMI)
    PM_CLIFFORD = PM

elif TRANSPILER_MODE == "qiskit_hf":
    Configuration.gym_name = "qiskit-ibm-transpiler"
    PM, PM_CLIFFORD = _make_ai_pass_managers(_QISKIT_REPO)

else:  # sqr
    Configuration.gym_name = "qiskit-ibm-transpiler"
    PM, PM_CLIFFORD = _make_ai_pass_managers(_SQR_REPO)

# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------
@benchpress_test_validation
class TestWorkoutDeviceTranspile100Q(WorkoutDeviceTranspile100Q):

    def test_QFT_100_transpile(self, benchmark):
        """Compile 100Q QFT circuit against FakeMiami"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("qft") + "qft_N100.qasm", benchmark
        )

        @benchmark
        def result():
            return run_with_progress("QFT_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_QV_100_transpile(self, benchmark):
        """Compile 100Q QV circuit against FakeMiami"""
        circuit = QuantumVolume(100, 100, seed=12345)

        @benchmark
        def result():
            return run_with_progress("QV_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_circSU2_89_transpile(self, benchmark):
        """Compile 89Q circSU2 circuit against FakeMiami"""
        circuit = EfficientSU2(89, reps=3, entanglement="circular")
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_progress("circSU2_89", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_circSU2_100_transpile(self, benchmark):
        """Compile 100Q circSU2 circuit against FakeMiami"""
        circuit = EfficientSU2(100, reps=3, entanglement="circular")
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_progress("circSU2_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_BV_100_transpile(self, benchmark):
        """Compile 100Q BV circuit against FakeMiami"""
        circuit = bv_all_ones(100)
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_progress("BV_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_square_heisenberg_100_transpile(self, benchmark):
        """Compile 100Q square-Heisenberg circuit against FakeMiami"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("square-heisenberg")
            + "square_heisenberg_N100.qasm",
            benchmark,
        )

        @benchmark
        def result():
            return run_with_progress("square_heisenberg_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_QAOA_100_transpile(self, benchmark):
        """Compile 100Q QAOA circuit against FakeMiami"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("qaoa") + "qaoa_barabasi_albert_N100_3reps.qasm",
            benchmark,
        )

        @benchmark
        def result():
            return run_with_progress("QAOA_100", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_BVlike_simplification_transpile(self, benchmark):
        """Transpile a BV-like circuit that should collapse to a single X and Z gate"""
        circuit = trivial_bvlike_circuit(100)
        input_circuit_properties(circuit, benchmark)

        @benchmark
        def result():
            return run_with_progress("BVlike_simplification", lambda: PM.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)

    def test_clifford_100_transpile(self, benchmark):
        """Compile 100Q Clifford circuit against FakeMiami"""
        circuit = qasm_circuit_loader(
            Configuration.get_qasm_dir("clifford") + "clifford_100_12345.qasm",
            benchmark,
        )

        @benchmark
        def result():
            return run_with_progress("clifford_100", lambda: PM_CLIFFORD.run(circuit), TRANSPILER_MODE)

        output_circuit_properties(result, TWO_Q_GATE, benchmark)
        assert circuit_validator(result, MIAMI)
