from dataclasses import replace
from types import SimpleNamespace

import pytest

import main_v2
import Optimization.objective as objective_module
from Optimization.checkpoint import CheckpointStore, SCHEMA_VERSION
from Optimization.objective import evaluation_identity, evaluate_candidate_on_seed
from Optimization.parameterization import (
    CandidateParameters,
    OptimizationConfig,
    candidate_to_end_to_end_config,
    derive_candidate,
)
from Optimization.run_parameter_optimization import (
    _parser,
    build_smoke_instance,
    smoke_search_space,
)
from Optimization.search import _checkpoint_metadata, _sample_valid_candidates


BUDGET = 300_000
PARAMETERS = CandidateParameters(
    alpha_peel=0.0334,
    alpha_rank=0.20,
    alpha_sgn=0.001,
    c_peel=1.25,
    c_rank=1.30,
    h_min=0.80,
    h_span=0.10,
    theta=0.12,
    eta_test=0.10,
    kappa_ratio=0.60,
    epsilon_tom=0.80,
)


def _config(**values):
    defaults = dict(
        total_copies=BUDGET,
        tuning_seeds=(11, 12),
        holdout_seeds=(21,),
        halving_seed_counts=(1, 2),
        number_of_candidates=1,
    )
    defaults.update(values)
    return OptimizationConfig(**defaults)


def _derived(config):
    return derive_candidate(
        PARAMETERS,
        n=3,
        d=2,
        total_copies=BUDGET,
        optimization_config=config,
    )


def test_optimization_propagates_every_enumeration_memory_control():
    config = _config(
        inner_enumeration_workers=4,
        max_score_array_bytes=111_111,
        max_structured_bell_workspace_bytes=222_222,
        max_enumeration_workspace_bytes=333_333,
        enumeration_workspace_safety_factor=1.75,
    )
    learner = candidate_to_end_to_end_config(
        _derived(config),
        d=2,
        learner_seed=11,
        total_copies=BUDGET,
        optimization_config=config,
    )
    execution = learner.enumeration_execution
    assert execution == learner.peeling_override.enumeration_execution
    assert execution == learner.recovery_override.enumeration_execution
    assert execution.workers == 4
    assert execution.max_score_array_bytes == 111_111
    assert execution.max_structured_bell_workspace_bytes == 222_222
    assert execution.max_enumeration_workspace_bytes == 333_333
    assert execution.enumeration_workspace_safety_factor == 1.75


@pytest.mark.parametrize("backend", ("batched_counts", "legacy_shotwise"))
def test_cli_parses_backends_and_all_byte_caps(backend):
    args = _parser().parse_args(
        [
            "--simulation-backend",
            backend,
            "--max-score-array-bytes",
            "1000",
            "--max-structured-bell-workspace-bytes",
            "2000",
            "--max-enumeration-workspace-bytes",
            "3000",
            "--enumeration-workspace-safety-factor",
            "1.5",
        ]
    )
    assert args.simulation_backend == backend
    assert args.max_score_array_bytes == 1000
    assert args.max_structured_bell_workspace_bytes == 2000
    assert args.max_enumeration_workspace_bytes == 3000
    assert args.enumeration_workspace_safety_factor == 1.5


def test_optimization_defaults_to_compressed_backend_but_end_to_end_compatibility_does_not():
    assert OptimizationConfig().simulation_backend == "batched_counts"
    assert main_v2.EndToEndConfig(0.5, 0.1).simulation_backend == "legacy_shotwise"


def test_evaluation_identity_changes_with_backend_or_workspace_policy():
    base = _config()
    derived = _derived(base)
    objective = base.effective_objective
    base_key = evaluation_identity(derived, 11, BUDGET, base, objective)
    assert base_key != evaluation_identity(
        derived,
        11,
        BUDGET,
        replace(base, simulation_backend="legacy_shotwise"),
        objective,
    )
    assert base_key != evaluation_identity(
        derived,
        11,
        BUDGET,
        replace(base, max_enumeration_workspace_bytes=123_456),
        objective,
    )


def test_checkpoint_records_policy_and_rejects_changed_workspace_cap(tmp_path):
    instance = build_smoke_instance()
    base = _config(max_enumeration_workspace_bytes=1_000_000)
    records, _rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=BUDGET,
        optimization_config=base,
        search_space=smoke_search_space(),
        search_seed=base.search_seed,
        target_count=1,
    )
    metadata = _checkpoint_metadata(
        instance=instance,
        objective=base.effective_objective,
        config=base,
        search_space=smoke_search_space(),
        records=records,
    )
    assert metadata["evaluation_identity_schema"] == 7
    assert metadata["enumeration_memory_policy"] == {
        "max_score_array_bytes": None,
        "max_structured_bell_workspace_bytes": None,
        "max_enumeration_workspace_bytes": 1_000_000,
        "enumeration_workspace_safety_factor": 1.25,
    }
    path = tmp_path / "policy-checkpoint.json"
    CheckpointStore(
        str(path), expected_metadata=metadata, every_n_evaluations=1
    ).save({}, {}, status="running")

    changed = replace(base, max_enumeration_workspace_bytes=2_000_000)
    changed_metadata = _checkpoint_metadata(
        instance=instance,
        objective=changed.effective_objective,
        config=changed,
        search_space=smoke_search_space(),
        records=records,
    )
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        CheckpointStore(
            str(path), expected_metadata=changed_metadata, every_n_evaluations=1
        ).load()
    assert SCHEMA_VERSION == 7


def test_seed_evaluation_records_active_resource_policy(monkeypatch):
    instance = build_smoke_instance()
    config = _config(
        max_score_array_bytes=101,
        max_structured_bell_workspace_bytes=202,
        max_enumeration_workspace_bytes=303,
        enumeration_workspace_safety_factor=1.5,
    )
    ledger = main_v2.CopyLedger((("fake", 1),))

    def fake_full(_view, *, config):
        return SimpleNamespace(
            theorem_certified=False,
            realized_total=1,
            realized_copy_ledger=ledger,
            success=False,
            estimator_available=False,
            failure_stage="fake",
            failure_reason="fake",
            peeling=None,
            grouping=None,
            localization=None,
            budget_truncated=False,
            truncated_stages=(),
        )

    monkeypatch.setattr(objective_module, "full_cebp_tomography", fake_full)
    evaluation = evaluate_candidate_on_seed(
        instance,
        _derived(config),
        11,
        BUDGET,
        config,
    )
    policy = dict(evaluation.execution_resource_policy)
    assert policy["simulation_backend"] == "batched_counts"
    assert policy["max_score_array_bytes"] == "101"
    assert policy["max_structured_bell_workspace_bytes"] == "202"
    assert policy["max_enumeration_workspace_bytes"] == "303"
    assert policy["enumeration_workspace_safety_factor"] == "1.5"
