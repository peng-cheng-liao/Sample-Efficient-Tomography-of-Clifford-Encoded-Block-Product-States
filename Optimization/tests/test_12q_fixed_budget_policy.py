from copy import deepcopy

import pytest

import main_v2
import Optimization.progressive_search as progressive_module
from Demo.run_6q_fixed_budget_min_error_1m import build_progressive_config
from Optimization import (
    CandidateParameters,
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES,
    OptimizationConfig,
    ProgressiveSearchConfig,
    derive_candidate,
    is_sufficient_relative_improvement,
    relative_loss_improvement,
)
from Optimization.checkpoint import CheckpointStore
from Optimization.objective import require_trace_distance_objective_available
from Optimization.parameterization import (
    candidate_to_end_to_end_config,
    preflight_resource_check,
)
from Optimization.progressive_search import _progressive_checkpoint_metadata
from Optimization.run_parameter_optimization import (
    _parser,
    build_smoke_instance,
    smoke_search_space,
)


BUDGET = 1_000_000
PARAMETERS = CandidateParameters(
    alpha_peel=0.02,
    alpha_rank=0.02,
    alpha_sgn=0.001,
    c_peel=1.25,
    c_rank=1.25,
    h_min=0.70,
    h_span=0.10,
    theta=0.12,
    eta_test=0.10,
    kappa_ratio=0.60,
    epsilon_tom=1.0,
)


def _optimization_config(**values):
    defaults = dict(
        total_copies=BUDGET,
        tuning_seeds=tuple(range(101, 117)),
        holdout_seeds=(901, 902),
        halving_seed_counts=(1, 2),
        number_of_candidates=2,
        max_enumeration_workspace_bytes=2_000_000_000,
        max_score_array_bytes=200_000_000,
        max_structured_bell_workspace_bytes=300_000_000,
    )
    defaults.update(values)
    return OptimizationConfig(**defaults)


def test_qubit_policy_defaults_are_12_without_raising_dense_debug_defaults():
    assert main_v2.MAX_SUPPORTED_END_TO_END_QUBITS == 12
    assert main_v2.PeelingConfig().max_enumeration_qubits == 12
    assert main_v2.RecoveryConfig(theta=0.1).max_enumeration_qubits == 12
    end_to_end = main_v2.EndToEndConfig(0.5, 0.1)
    assert end_to_end.max_enumeration_qubits == 12

    optimization = OptimizationConfig()
    assert optimization.max_enumeration_qubits == 12
    assert optimization.max_oracle_dense_qubits == 12

    # Dense materialization remains an explicit debug-only policy.
    assert end_to_end.max_dense_qubits == 8
    assert end_to_end.max_dense_debug_qubits == 8
    assert optimization.max_dense_qubits == 8
    assert main_v2.TomographyConfig(0.5, 0.1).max_dense_qubits == 8


def test_qubit_policy_accepts_12_and_rejects_13_before_execution():
    config = _optimization_config()
    derived = derive_candidate(
        PARAMETERS,
        n=12,
        d=2,
        total_copies=BUDGET,
        optimization_config=config,
    )
    check = preflight_resource_check(
        derived,
        n=12,
        d=2,
        total_copies=BUDGET,
        optimization_config=config,
    )
    assert check.safe_to_execute

    learner = candidate_to_end_to_end_config(
        derived,
        d=2,
        learner_seed=101,
        total_copies=BUDGET,
        optimization_config=config,
    )
    assert learner.max_enumeration_qubits == 12
    assert learner.peeling_override.max_enumeration_qubits == 12
    assert learner.recovery_override.max_enumeration_qubits == 12
    assert learner.max_dense_debug_qubits == 12

    with pytest.raises(ValueError, match=r"n=13 exceeds max_enumeration_qubits=12"):
        derive_candidate(
            PARAMETERS,
            n=13,
            d=2,
            total_copies=BUDGET,
            optimization_config=config,
        )
    with pytest.raises(ValueError, match=r"n=13 exceeds max_enumeration_qubits=12"):
        preflight_resource_check(
            derived,
            n=13,
            d=2,
            total_copies=BUDGET,
            optimization_config=config,
        )


def test_12q_fixed_budget_oracle_policy_is_available_without_dense_evaluation():
    instance = main_v2.random_cebp_state(
        12,
        2,
        block_sizes=(2,) * 6,
        pure=True,
        clifford_steps=0,
        seed=17,
    )
    config = _optimization_config()
    require_trace_distance_objective_available(
        instance, config, config.effective_objective
    )

    estimate = main_v2.estimate_enumeration_workspace(
        12,
        "recovery",
        simulation_backend="batched_counts",
        safety_factor=config.enumeration_workspace_safety_factor,
    )
    assert estimate.predicted_peak_bytes <= config.max_enumeration_workspace_bytes


def test_end_to_end_rejects_13q_at_policy_guard_before_schedule_or_sampling():
    instance = main_v2.random_cebp_state(
        13,
        2,
        block_sizes=(1,) * 13,
        pure=True,
        clifford_steps=0,
        seed=18,
    )
    with pytest.raises(ValueError, match=r"n=13 exceeds max_enumeration_qubits=12"):
        main_v2.full_cebp_tomography(
            instance.learner_view(),
            config=main_v2.EndToEndConfig(0.5, 0.1),
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: main_v2.EndToEndConfig(0.5, 0.1, max_enumeration_qubits=13),
        lambda: OptimizationConfig(max_enumeration_qubits=13),
        lambda: OptimizationConfig(max_oracle_dense_qubits=13),
    ),
)
def test_user_policy_limits_cannot_raise_supported_workflow_above_12(factory):
    with pytest.raises(ValueError, match=r"cannot exceed.*12"):
        factory()


def test_cli_defaults_to_12_and_rejects_13_qubit_limits():
    parser = _parser()
    defaults = parser.parse_args([])
    assert defaults.max_enumeration_qubits == 12
    assert defaults.max_oracle_dense_qubits == 12
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-enumeration-qubits", "13"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-oracle-dense-qubits", "13"])


def test_fixed_budget_improvement_policy_boundaries_preserve_strict_comparison():
    assert DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO == 0.0
    assert ProgressiveSearchConfig().relative_improvement_threshold == 0.0
    threshold = 0.05

    below = relative_loss_improvement(100.0, 95.1)
    exact = relative_loss_improvement(100.0, 95.0)
    above = relative_loss_improvement(100.0, 94.9)
    assert below == pytest.approx(0.049)
    assert exact == pytest.approx(0.05)
    assert above == pytest.approx(0.051)
    assert not is_sufficient_relative_improvement(below, threshold)
    assert is_sufficient_relative_improvement(exact, threshold)
    assert is_sufficient_relative_improvement(above, threshold)
    assert not is_sufficient_relative_improvement(
        relative_loss_improvement(100.0, 97.9), threshold
    )


def test_fixed_budget_candidate_hard_cap_accepts_256_and_rejects_257():
    assert MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES == 256
    assert ProgressiveSearchConfig().max_candidates == 256
    assert ProgressiveSearchConfig(max_candidates=256).max_candidates == 256
    with pytest.raises(ValueError, match=r"hard cap of 256"):
        ProgressiveSearchConfig(max_candidates=257)

    # The unrelated fixed-size optimizer retains its independent candidate count.
    assert OptimizationConfig(number_of_candidates=257).number_of_candidates == 257


def test_progressive_search_requests_exactly_256_catalog_entries(monkeypatch):
    requested = []

    class CatalogProbe(RuntimeError):
        pass

    def probe_catalog(**kwargs):
        requested.append(kwargs["target_count"])
        raise CatalogProbe

    monkeypatch.setattr(progressive_module, "_sample_valid_candidates", probe_catalog)
    config = _optimization_config(total_copies=300_000)
    with pytest.raises(CatalogProbe):
        progressive_module.optimize_cebp_parameters_progressive(
            build_smoke_instance(),
            300_000,
            config,
            smoke_search_space(),
            progressive_config=ProgressiveSearchConfig(),
        )
    assert requested == [256]


def test_demo_uses_central_fixed_budget_policy_defaults():
    progressive = build_progressive_config()
    assert progressive.max_candidates == MAX_FIXED_BUDGET_MIN_ERROR_CANDIDATES
    assert (
        progressive.relative_improvement_threshold
        == DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO
    )


def test_checkpoint_records_new_policy_and_rejects_old_policy_metadata(tmp_path):
    instance = build_smoke_instance()
    config = _optimization_config(total_copies=300_000)
    progressive = ProgressiveSearchConfig()
    metadata = _progressive_checkpoint_metadata(
        instance=instance,
        objective=config.effective_objective,
        config=config,
        search_space=smoke_search_space(),
        records=(),
        progressive_config=progressive,
    )
    assert metadata["qubit_policy"] == {
        "max_dense_qubits_compatibility_alias": 8,
        "max_enumeration_qubits": 12,
        "max_oracle_dense_qubits": 12,
    }
    assert metadata["progressive_config"]["max_candidates"] == 256
    assert metadata["progressive_config"]["relative_improvement_threshold"] == 0.0

    matching_path = tmp_path / "matching.json"
    matching = CheckpointStore(
        str(matching_path), expected_metadata=metadata, every_n_evaluations=1
    )
    matching.save({}, {}, status="running")
    assert matching.load() == ({}, {})

    old_metadata = deepcopy(metadata)
    old_metadata["qubit_policy"]["max_enumeration_qubits"] = 8
    old_metadata["qubit_policy"]["max_oracle_dense_qubits"] = 8
    old_metadata["progressive_config"]["max_candidates"] = 1024
    old_metadata["progressive_config"]["relative_improvement_threshold"] = 0.02
    old_path = tmp_path / "old-policy.json"
    CheckpointStore(
        str(old_path), expected_metadata=old_metadata, every_n_evaluations=1
    ).save({}, {}, status="running")
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        CheckpointStore(
            str(old_path), expected_metadata=metadata, every_n_evaluations=1
        ).load()
