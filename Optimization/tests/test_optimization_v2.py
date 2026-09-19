from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import main_v2
import Optimization.objective as objective_module
import Optimization.search as search_module
from Optimization import (
    CandidateParameters,
    InvalidCandidateError,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    SearchSpace,
    candidate_to_end_to_end_config,
    derive_candidate,
    evaluate_candidate_on_seed,
    optimize_cebp_parameters,
    predict_tomography_pool_copies,
    preflight_resource_check,
)
from Optimization.run_parameter_optimization import build_smoke_instance


BUDGET = 300_000
BASE = CandidateParameters(
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


def config(*, objective=None, **values):
    fixed_error_mode = bool(
        isinstance(objective, OptimizationObjective)
        and objective.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES
    )
    defaults = dict(
        total_copies=BUDGET,
        tuning_seeds=(tuple(range(11, 27)) if fixed_error_mode else (11, 12)),
        holdout_seeds=((121,) if fixed_error_mode else (21,)),
        halving_seed_counts=(
            (1, 2, 4, 8, 16) if fixed_error_mode else (1, 2)
        ),
        number_of_candidates=2,
        objective=objective,
    )
    defaults.update(values)
    return OptimizationConfig(**defaults)


def derived(parameters=BASE, cfg=None):
    cfg = cfg or config()
    return derive_candidate(
        parameters,
        n=3,
        d=2,
        total_copies=BUDGET,
        optimization_config=cfg,
    )


def fixed_error(target=0.05):
    return OptimizationObjective(
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
        copy_ceiling=BUDGET,
        error_target=target,
    )


def install_fake_learner(monkeypatch, trace_model, *, fail_model=lambda _e, _s: False):
    calls = []

    def fake_full(_view, *, config):
        epsilon = float(config.tomography_override.epsilon_tom)
        seed = int(config.seed)
        calls.append((epsilon, seed))
        peel = 2 * int(config.peeling_override.M1)
        recovery = 2 * int(config.recovery_override.M2)
        grouping = 1_000
        sign = 100
        tomography = predict_tomography_pool_copies(
            (2,), epsilon, config.tomography_override.zeta_tom
        )
        ledger = main_v2.CopyLedger(
            (
                ("peeling_bell_pool", peel),
                ("recovery_bell_pool", recovery),
                ("grouping_ordinary_pool", grouping),
                ("syndrome_sign_pool", sign),
                ("block_tomography_pool", tomography),
            )
        )
        failed = bool(fail_model(epsilon, seed))
        return SimpleNamespace(
            theorem_certified=False,
            realized_total=ledger.total,
            realized_copy_ledger=ledger,
            success=not failed,
            failure_stage="fake" if failed else None,
            failure_reason="fake_failure" if failed else None,
            peeling=SimpleNamespace(t=1),
            grouping=SimpleNamespace(clusters=((0,),)),
            localization=SimpleNamespace(J_C=(((0,), (0, 1)),), J_aux=()),
            _trace=float(trace_model(epsilon, seed)),
        )

    monkeypatch.setattr(objective_module, "full_cebp_tomography", fake_full)
    monkeypatch.setattr(
        objective_module,
        "debug_end_to_end_trace_error",
        lambda result, _instance, **_kwargs: 2.0 * result._trace,
    )
    return calls


def test_objective_validation_and_legacy_default():
    assert config().effective_objective.mode is OptimizationMode.FIXED_BUDGET_MIN_ERROR
    assert config().effective_objective.budget_utilization_target == pytest.approx(0.99)
    with pytest.raises(ValueError, match="requires.*error_target"):
        OptimizationObjective(mode="fixed_error_min_copies", copy_ceiling=10)
    for target in (0.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="error_target"):
            OptimizationObjective(
                mode="fixed_error_min_copies", copy_ceiling=10, error_target=target
            )
    assert "require_error_target_on_all_tuning_seeds" not in (
        OptimizationObjective.__dataclass_fields__
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        OptimizationObjective(
            mode="fixed_error_min_copies",
            copy_ceiling=10,
            error_target=0.05,
            require_error_target_on_all_tuning_seeds=False,
        )


def test_optimizer_backend_wiring_and_batched_default():
    default_cfg = config()
    assert default_cfg.simulation_backend == "batched_counts"

    legacy_cfg = config(simulation_backend="legacy_shotwise")
    assert legacy_cfg.simulation_backend == "legacy_shotwise"
    legacy_learner = candidate_to_end_to_end_config(
        derived(cfg=legacy_cfg),
        d=2,
        learner_seed=11,
        total_copies=BUDGET,
        optimization_config=legacy_cfg,
    )
    assert legacy_learner.simulation_backend == "legacy_shotwise"
    assert legacy_learner.max_realized_copies == BUDGET

    fast_cfg = default_cfg
    fast_learner = candidate_to_end_to_end_config(
        derived(cfg=fast_cfg),
        d=2,
        learner_seed=11,
        total_copies=BUDGET,
        optimization_config=fast_cfg,
    )
    assert fast_learner.simulation_backend == "batched_counts"
    assert fast_learner.max_realized_copies == BUDGET

    with pytest.raises(ValueError, match="simulation_backend"):
        config(simulation_backend="not-a-backend")


def test_tomography_prediction_exact_max_rule_and_edge_cases():
    epsilon, zeta = 0.8, 0.1
    predicted = predict_tomography_pool_copies((2, 3), epsilon, zeta)
    local = [
        main_v2.register_tomography_budget(
            (index,), tuple(range(size)), epsilon / 2, zeta / 2
        ).L_C
        for index, size in enumerate((2, 3))
    ]
    assert predicted == max(local)
    assert predicted != sum(local)
    assert predict_tomography_pool_copies((2,), epsilon, zeta) == (
        main_v2.register_tomography_budget((0,), (0, 1), epsilon, zeta).L_C
    )
    assert predict_tomography_pool_copies((), epsilon, zeta) == 0


def test_graceful_tomography_uses_physical_remainder_not_epsilon_prediction():
    cfg = config(number_of_candidates=1, tuning_seeds=(11,), halving_seed_counts=(1,))
    evaluation = evaluate_candidate_on_seed(
        build_smoke_instance(), derived(cfg=cfg), 31, BUDGET, cfg
    )
    actual = dict(evaluation.realized_copy_ledger)["block_tomography_pool"]
    assert actual > 0
    structural = sum(
        count
        for name, count in evaluation.realized_copy_ledger
        if name != "block_tomography_pool"
    )
    assert actual == BUDGET - structural
    assert evaluation.estimator_available and evaluation.budget_feasible


def test_fixed_budget_epsilon_refinement_is_disabled(monkeypatch):
    calls = install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    cfg = config(
        number_of_candidates=1,
        tuning_seeds=(11,),
        halving_seed_counts=(1,),
        tomography_refinement_enabled=True,
        tomography_refinement_steps=4,
    )
    space = SearchSpace(epsilon_tom=(0.35, 1.8))
    result = optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, cfg, space, initial_candidates=(BASE,)
    )
    assert result.analytical_refinement_trial_count == 0
    assert result.refinement_actual_learner_run_count == 0
    assert not result.search_metadata.tomography_refinement_enabled
    assert "disabled" in result.search_metadata.tomography_refinement_note
    diagnostic_epsilon = cfg.fixed_budget_epsilon_tom_diagnostic
    assert calls == [(diagnostic_epsilon, 11), (diagnostic_epsilon, 21)]


@pytest.mark.parametrize(
    ("parameters", "limits"),
    (
        (
            replace(BASE, kappa_ratio=0.001),
            {"max_predicted_grouping_copies": 1_000_000},
        ),
        (
            replace(BASE, epsilon_tom=0.01),
            {"max_predicted_tomography_copies": 1_000_000},
        ),
    ),
)
def test_graceful_preflight_does_not_reject_nominal_tolerance_copy_counts(
    parameters, limits
):
    cfg = config(number_of_candidates=1, tuning_seeds=(11,), halving_seed_counts=(1,), **limits)
    candidate = derived(parameters, cfg)
    check = preflight_resource_check(
        candidate, n=3, d=2, total_copies=BUDGET, optimization_config=cfg
    )
    assert check.safe_to_execute and not check.runtime_safety_rejected
    assert "diagnostic only" in check.guarantee_level


def test_baseline_candidate_passes_default_runtime_safety():
    check = preflight_resource_check(
        derived(), n=3, d=2, total_copies=BUDGET, optimization_config=config()
    )
    assert check.safe_to_execute
    assert not check.runtime_safety_rejected
    assert "rigorously bound actual execution" in check.guarantee_level


def test_fixed_budget_weight_catalog_has_nontrivial_valid_fraction():
    rng = np.random.default_rng(123)
    cfg = config(total_copies=1_000_000)
    valid = 0
    for _ in range(100):
        try:
            derive_candidate(
                search_module.sample_candidate(rng, SearchSpace()),
                n=6,
                d=3,
                total_copies=1_000_000,
                optimization_config=cfg,
            )
            valid += 1
        except InvalidCandidateError:
            pass
    assert valid >= 25


def test_fixed_budget_sampler_omits_inert_compatibility_coordinates():
    first = search_module.sample_candidate(
        np.random.default_rng(123),
        SearchSpace(),
        mode=OptimizationMode.FIXED_BUDGET_MIN_ERROR,
    )
    second = search_module.sample_candidate(
        np.random.default_rng(456),
        SearchSpace(),
        mode=OptimizationMode.FIXED_BUDGET_MIN_ERROR,
    )
    assert not hasattr(first, "alpha_peel")
    assert not hasattr(first, "epsilon_tom")
    assert first.peel_weight != second.peel_weight


def test_fixed_budget_semantic_identity_ignores_only_inert_coordinates():
    cfg = config()
    inert_variant = replace(
        BASE,
        alpha_peel=0.01,
        alpha_rank=0.02,
        alpha_sgn=0.003,
        epsilon_tom=1.7,
    )
    first = derived(BASE, cfg)
    second = derived(inert_variant, cfg)
    identity = lambda value: objective_module.evaluation_identity(
        value, 11, BUDGET, cfg, cfg.effective_objective
    )
    assert identity(first) == identity(second)
    active = derived(replace(BASE, tomography_weight=3.0), cfg)
    assert identity(first) != identity(active)

    fixed_cfg = config(objective=fixed_error())
    fixed_first = derived(BASE, fixed_cfg)
    fixed_second = derived(replace(BASE, epsilon_tom=1.2), fixed_cfg)
    fixed_identity = lambda value: objective_module.evaluation_identity(
        value, 11, BUDGET, fixed_cfg, fixed_cfg.effective_objective
    )
    assert fixed_identity(fixed_first) == fixed_identity(fixed_second)


def test_fixed_budget_semantic_duplicates_canonicalize_before_catalog(monkeypatch):
    calls = install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    duplicate = replace(
        BASE,
        alpha_peel=0.01,
        alpha_rank=0.02,
        alpha_sgn=0.003,
        epsilon_tom=1.7,
    )
    cfg = config(
        number_of_candidates=2,
        tuning_seeds=(11,),
        halving_seed_counts=(1,),
    )
    result = optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, cfg, initial_candidates=(BASE, duplicate)
    )
    assert sum(
        record.candidate_id.startswith("initial-")
        for record in result.candidate_catalog
    ) == 1
    assert result.actual_learner_run_count == 3  # two tuning candidates + holdout
    assert result.cache_hit_count == 0
    assert len(calls) == 3


def test_fixed_error_converter_uses_shared_graceful_backend():
    cfg = config(objective=fixed_error())
    value = derived(cfg=cfg)
    learner = candidate_to_end_to_end_config(
        value,
        d=2,
        learner_seed=11,
        total_copies=BUDGET,
        optimization_config=cfg,
    )
    assert learner.execution_policy is main_v2.ExecutionPolicy.FIXED_BUDGET_GRACEFUL
    assert learner.fixed_budget_stage_weights is not None
    assert learner.tomography_override.epsilon_tom == pytest.approx(
        cfg.fixed_budget_epsilon_tom_diagnostic
    )


def test_checkpoint_resume_avoids_completed_learner_reruns(monkeypatch, tmp_path):
    checkpoint = tmp_path / "optimizer.json"
    calls = install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    original = objective_module.full_cebp_tomography
    interrupted_calls = 0

    def interrupting(*args, **kwargs):
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == 3:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(objective_module, "full_cebp_tomography", interrupting)
    first = config(checkpoint_path=str(checkpoint), checkpoint_key="resume-test")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        optimize_cebp_parameters(
            build_smoke_instance(),
            BUDGET,
            first,
            initial_candidates=(BASE, replace(BASE, tomography_weight=3.0)),
        )
    payload = json.loads(checkpoint.read_text())
    assert payload["status"] == "running"
    assert len(payload["evaluation_cache"]) == 2

    monkeypatch.setattr(objective_module, "full_cebp_tomography", original)
    before_resume = len(calls)
    resumed = replace(first, resume_from_checkpoint=True)
    result = optimize_cebp_parameters(
        build_smoke_instance(),
        BUDGET,
        resumed,
        initial_candidates=(BASE, replace(BASE, tomography_weight=3.0)),
    )
    assert len(calls) - before_resume == 2
    assert result.actual_learner_run_count == 4
    assert result.cache_hit_count >= 2
    assert result.search_metadata.resumed_from_checkpoint
    assert json.loads(checkpoint.read_text())["status"] == "complete"
    assert not list(tmp_path.glob("*.tmp"))

    uninterrupted_cfg = replace(
        first,
        checkpoint_path=None,
        resume_from_checkpoint=False,
    )
    uninterrupted = optimize_cebp_parameters(
        build_smoke_instance(),
        BUDGET,
        uninterrupted_cfg,
        initial_candidates=(BASE, replace(BASE, tomography_weight=3.0)),
    )
    assert result.best_candidate == uninterrupted.best_candidate
    assert result.tuning_evaluation == uninterrupted.tuning_evaluation
    assert result.holdout_evaluation == uninterrupted.holdout_evaluation


def test_checkpoint_key_required_only_when_checkpointing_enabled(tmp_path):
    disabled = OptimizationConfig()
    assert disabled.checkpoint_path is None and disabled.checkpoint_key == ""
    with pytest.raises(ValueError, match="uniquely identify the target state/run"):
        OptimizationConfig(checkpoint_path=str(tmp_path / "empty.json"))
    with pytest.raises(ValueError, match="uniquely identify the target state/run"):
        OptimizationConfig(
            checkpoint_path=str(tmp_path / "whitespace.json"), checkpoint_key="   "
        )
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        OptimizationConfig(
            checkpoint_path=str(tmp_path / "padded.json"), checkpoint_key=" state-a "
        )
    enabled = OptimizationConfig(
        checkpoint_path=str(tmp_path / "valid.json"), checkpoint_key="state-a"
    )
    assert enabled.checkpoint_key == "state-a"


def test_checkpoint_incompatible_metadata_is_refused(monkeypatch, tmp_path):
    calls = install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    checkpoint = tmp_path / "optimizer.json"
    first = config(
        number_of_candidates=1,
        tuning_seeds=(11,),
        halving_seed_counts=(1,),
        checkpoint_path=str(checkpoint),
        checkpoint_key="first",
    )
    optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, first, initial_candidates=(BASE,)
    )
    calls_before_incompatible_resume = len(calls)
    incompatible = replace(first, checkpoint_key="different", resume_from_checkpoint=True)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        optimize_cebp_parameters(
            main_v2.random_cebp_state(
                3, 2, block_sizes=(1, 2), clifford_steps=3, seed=9876
            ),
            BUDGET,
            incompatible,
            initial_candidates=(BASE,),
        )
    # Same (n,d), different caller-supplied target key: no cached result is reused.
    assert len(calls) == calls_before_incompatible_resume


def test_pre_remediation_checkpoint_schema_is_rejected(monkeypatch, tmp_path):
    install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    checkpoint = tmp_path / "optimizer.json"
    cfg = config(
        number_of_candidates=1,
        tuning_seeds=(11,),
        halving_seed_counts=(1,),
        checkpoint_path=str(checkpoint),
        checkpoint_key="schema-guard",
    )
    optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, cfg, initial_candidates=(BASE,)
    )
    payload = json.loads(checkpoint.read_text())
    assert payload["schema_version"] == 7
    payload["schema_version"] = 3
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        optimize_cebp_parameters(
            build_smoke_instance(),
            BUDGET,
            replace(cfg, resume_from_checkpoint=True),
            initial_candidates=(BASE,),
        )


def test_counters_are_exposed_and_consistent(monkeypatch):
    install_fake_learner(monkeypatch, lambda _epsilon, _seed: 0.04)
    cfg = config(number_of_candidates=1, tuning_seeds=(11,), halving_seed_counts=(1,))
    result = optimize_cebp_parameters(
        build_smoke_instance(), BUDGET, cfg, initial_candidates=(BASE,)
    )
    assert result.evaluation_attempt_count == 2
    assert result.actual_learner_run_count == 2
    assert result.cache_hit_count == 0
    assert result.preflight_rejection_count == 0
    assert result.analytical_refinement_trial_count == 0
    assert result.refinement_actual_learner_run_count == 0
