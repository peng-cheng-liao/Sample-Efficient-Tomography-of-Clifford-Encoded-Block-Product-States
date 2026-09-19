from dataclasses import replace
import inspect
import json

import pytest

import Optimization.objective as objective_module
import Optimization.progressive_search as progressive_module
from Optimization import (
    DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO,
    OptimizationConfig,
    OptimizationMode,
    OptimizationObjective,
    ProgressiveSearchConfig,
    optimize_cebp_parameters_progressive,
)
from Optimization.checkpoint import SCHEMA_VERSION
from Optimization.objective import SeedEvaluation, evaluation_identity
from Optimization.parameterization import PreflightResourceCheck, SearchSpace
from Optimization.run_parameter_optimization import build_smoke_instance, smoke_search_space
from Optimization.search import _sample_valid_candidates
from Optimization.progressive_search import _validate_progressive_call


BUDGET = 300_000
TUNING_SEEDS = tuple(range(101, 117))
HOLDOUT_SEEDS = (901, 902)


def _optimization_config(**values):
    defaults = dict(
        total_copies=BUDGET,
        search_seed=2468,
        tuning_seeds=TUNING_SEEDS,
        holdout_seeds=HOLDOUT_SEEDS,
        number_of_candidates=2,
        halving_seed_counts=(1, 2),
    )
    defaults.update(values)
    return OptimizationConfig(**defaults)


def _progressive(**values):
    defaults = dict(
        initial_candidates=16,
        max_candidates=64,
        seed_fidelities=(1, 2, 4, 8, 16),
        comparison_seed_count=16,
        relative_improvement_threshold=(
            DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO
        ),
        improvement_patience=2,
        min_rounds=3,
    )
    defaults.update(values)
    return ProgressiveSearchConfig(**defaults)


@pytest.fixture(scope="module")
def instance():
    return build_smoke_instance()


@pytest.fixture(scope="module")
def catalog(instance):
    records, _rejected = _sample_valid_candidates(
        instance=instance,
        total_copies=BUDGET,
        optimization_config=_optimization_config(),
        search_space=smoke_search_space(),
        search_seed=2468,
        target_count=256,
    )
    return records


def _check():
    return PreflightResourceCheck(
        mandatory_bell_copies=1,
        worst_case_sign_reservation=1,
        preflight_fixed_reservation=2,
        grouping_safety_estimate=1,
        tomography_safety_estimate=1,
        preflight_safety_estimate=4,
        rigorous_grouping_upper_bound=1,
        rigorous_tomography_upper_bound=1,
        rigorous_total_upper_bound=4,
        single_grouping_query_shots_estimate=1,
        safety_limit=BUDGET,
        runtime_safety_rejected=False,
        mathematical_budget_rejected=False,
        safe_to_execute=True,
        reason="mock-safe",
        guarantee_level="mock",
    )


def _seed_evaluation(seed, loss):
    return SeedEvaluation(
        learner_seed=int(seed),
        preflight_rejected=False,
        preflight_resource_check=_check(),
        operational_success=True,
        budget_feasible=True,
        post_run_realized_budget_feasible=True,
        loss=float(loss),
        loss_computed=True,
        trace_distance=float(loss),
        failure_stage=None,
        failure_reason=None,
        realized_copy_ledger=(("mock", BUDGET),),
        realized_total=BUDGET,
        copy_utilization=1.0,
        recovered_t=1,
        cluster_sizes=(1, 2),
        register_sizes=(1, 2),
        j_aux_size=0,
        error_feasible=None,
        error_excess=None,
        estimator_available=True,
    )


def _install_mock_evaluator(monkeypatch, catalog, loss_model=None):
    id_by_derived = {id(record.derived): record.candidate_id for record in catalog}
    calls = []
    seen = set()

    def catalog_provider(**kwargs):
        target = int(kwargs["target_count"])
        return tuple(catalog[:target]), 0

    def evaluator(
        _instance,
        derived,
        learner_seed,
        total_copies,
        optimization_config,
        *,
        objective=None,
    ):
        resolved = objective or optimization_config.effective_objective
        key = evaluation_identity(
            derived,
            learner_seed,
            total_copies,
            optimization_config,
            resolved,
        )
        assert key not in seen, "semantic candidate-seed learner work was duplicated"
        seen.add(key)
        candidate_id = id_by_derived[id(derived)]
        calls.append((candidate_id, int(learner_seed), key))
        index = int(candidate_id.split("-")[-1])
        loss = (
            loss_model(candidate_id, int(learner_seed))
            if loss_model is not None
            else 0.1 + index / 10_000.0 + (int(learner_seed) % 7) / 1_000_000.0
        )
        return _seed_evaluation(learner_seed, loss)

    monkeypatch.setattr(progressive_module, "_sample_valid_candidates", catalog_provider)
    monkeypatch.setattr(objective_module, "evaluate_candidate_on_seed", evaluator)
    return calls, seen


def _run(instance, progressive, *, config=None):
    return optimize_cebp_parameters_progressive(
        instance,
        BUDGET,
        config or _optimization_config(),
        smoke_search_space(),
        progressive_config=progressive,
    )


@pytest.mark.parametrize(
    "values, message",
    [
        ({"initial_candidates": 0}, "positive integer"),
        ({"max_candidates": 0}, "positive integer"),
        ({"initial_candidates": 8, "max_candidates": 4}, "must not exceed"),
        ({"candidate_growth_factor": 1}, "at least 2"),
        ({"seed_fidelities": ()}, "nonempty"),
        ({"seed_fidelities": (1, 1)}, "strictly increasing"),
        ({"seed_fidelities": (2, 1)}, "strictly increasing"),
        ({"comparison_seed_count": 0}, "positive integer"),
        (
            {"seed_fidelities": (1, 2, 4), "comparison_seed_count": 2},
            "at least max",
        ),
        ({"relative_improvement_threshold": -0.1}, "lie in"),
        ({"relative_improvement_threshold": 1.0}, "lie in"),
        ({"improvement_patience": 0}, "positive integer"),
        ({"min_rounds": 0}, "positive integer"),
        ({"retention_fraction": 1.0}, "lie in"),
        ({"checkpoint_every_n_new_evaluations": 0}, "positive integer"),
    ],
)
def test_progressive_config_validation(values, message):
    with pytest.raises(ValueError, match=message):
        ProgressiveSearchConfig(**values)


def test_progressive_validation_is_deferred_and_fixed_error_is_supported(
    monkeypatch, instance, catalog
):
    assert OptimizationConfig().tuning_seeds == (10001, 10002)
    with pytest.raises(ValueError, match="fidelity exceeds"):
        _run(
            instance,
            ProgressiveSearchConfig(),
            config=OptimizationConfig(total_copies=BUDGET),
        )
    fixed_error = OptimizationObjective(
        mode=OptimizationMode.FIXED_ERROR_MIN_COPIES,
        copy_ceiling=BUDGET,
        error_target=0.1,
    )
    config = _optimization_config(objective=fixed_error)
    resolved = _validate_progressive_call(
        instance,
        BUDGET,
        config,
        _progressive(max_candidates=16),
        (),
    )
    assert resolved.mode is OptimizationMode.FIXED_ERROR_MIN_COPIES


def test_nonempty_initial_candidates_are_supported(monkeypatch, instance, catalog):
    _install_mock_evaluator(monkeypatch, catalog)
    result = optimize_cebp_parameters_progressive(
        instance,
        BUDGET,
        _optimization_config(),
        progressive_config=_progressive(max_candidates=16),
        initial_candidates=(catalog[0].parameters,),
    )
    assert result.total_candidate_count == 16


def test_nested_catalog_prefixes_ids_and_search_seed_reproducibility(instance):
    cfg = _optimization_config()
    args = dict(
        instance=instance,
        total_copies=BUDGET,
        optimization_config=cfg,
        search_space=SearchSpace(),
    )
    c16, r16 = _sample_valid_candidates(**args, search_seed=77, target_count=16)
    c32, _ = _sample_valid_candidates(**args, search_seed=77, target_count=32)
    c64, _ = _sample_valid_candidates(**args, search_seed=77, target_count=64)
    repeated, repeated_rejected = _sample_valid_candidates(
        **args, search_seed=77, target_count=16
    )
    changed, _ = _sample_valid_candidates(**args, search_seed=78, target_count=16)
    assert c16 == c32[:16] == c64[:16] == repeated
    assert r16 == repeated_rejected
    assert [item.candidate_id for item in c16] == [
        f"candidate-{index:04d}" for index in range(16)
    ]
    assert tuple(item.parameters for item in c16) != tuple(
        item.parameters for item in changed
    )


def test_round_schedules_nested_seeds_and_maximum_cap(
    monkeypatch, instance, catalog
):
    calls, seen = _install_mock_evaluator(monkeypatch, catalog)
    result = _run(
        instance,
        _progressive(
            max_candidates=64,
            relative_improvement_threshold=0.0,
            min_rounds=99,
        ),
    )
    schedules = [
        [(s.candidates_entering, s.seed_fidelity) for s in summary.halving_summaries]
        for summary in result.search_metadata.round_summaries
    ]
    assert schedules == [
        [(16, 1), (8, 2), (4, 4)],
        [(32, 1), (16, 2), (8, 4), (4, 8)],
        [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16)],
    ]
    assert result.search_metadata.termination_reason == "max_candidates"
    assert result.actual_learner_run_count == len(calls) == len(seen)
    assert result.unique_cached_seed_evaluation_count == len(seen)
    assert result.cache_hit_count > 0
    assert result.evaluation_attempt_count == len(seen) + result.cache_hit_count


def test_large_pool_never_repeats_or_exceeds_sixteen_seeds(
    monkeypatch, instance, catalog
):
    calls, seen = _install_mock_evaluator(monkeypatch, catalog)
    result = _run(
        instance,
        _progressive(
            max_candidates=128,
            relative_improvement_threshold=0.0,
            min_rounds=99,
        ),
    )
    final = result.search_metadata.round_summaries[-1]
    assert [(s.candidates_entering, s.seed_fidelity) for s in final.halving_summaries] == [
        (128, 1),
        (64, 2),
        (32, 4),
        (16, 8),
        (8, 16),
    ]
    assert max(seed for _candidate, seed, _key in calls if seed in TUNING_SEEDS) <= 116
    assert len(calls) == len(seen) == result.actual_learner_run_count


def test_worse_common_seed_challenger_does_not_replace_incumbent(
    monkeypatch, instance, catalog
):
    def losses(candidate_id, seed):
        if candidate_id == "candidate-0000":
            return 0.1
        if candidate_id == "candidate-0016":
            return 0.05 if seed in TUNING_SEEDS[:8] else 0.35
        return 0.5

    _install_mock_evaluator(monkeypatch, catalog, losses)
    result = _run(
        instance,
        _progressive(max_candidates=32, min_rounds=99),
    )
    second = result.search_metadata.round_summaries[1]
    assert second.round_search_winner_id == "candidate-0016"
    assert second.incumbent_before_id == second.incumbent_after_id == "candidate-0000"
    assert second.relative_improvement == pytest.approx(0.0)
    assert second.selected_incumbent_comparison_mean_loss == pytest.approx(0.1)


def test_better_challenger_replaces_incumbent_and_reports_improvement(
    monkeypatch, instance, catalog
):
    def losses(candidate_id, _seed):
        if candidate_id == "candidate-0000":
            return 0.2
        if candidate_id == "candidate-0016":
            return 0.1
        return 0.5

    _install_mock_evaluator(monkeypatch, catalog, losses)
    result = _run(instance, _progressive(max_candidates=32, min_rounds=99))
    second = result.search_metadata.round_summaries[1]
    assert second.incumbent_after_id == "candidate-0016"
    assert second.relative_improvement == pytest.approx(0.5)


def test_convergence_patience_waits_two_expansions(monkeypatch, instance, catalog):
    _install_mock_evaluator(monkeypatch, catalog)
    result = _run(
        instance,
        _progressive(max_candidates=128, relative_improvement_threshold=0.05),
    )
    assert result.search_metadata.termination_reason == "converged"
    assert result.search_metadata.rounds_completed == 3
    assert [item.low_improvement_streak for item in result.search_metadata.round_summaries] == [
        0,
        1,
        2,
    ]


def test_above_threshold_improvement_resets_streak(monkeypatch, instance, catalog):
    def losses(candidate_id, _seed):
        index = int(candidate_id.split("-")[-1])
        if index == 0:
            return 0.1
        if index == 32:
            return 0.05
        return 0.5

    _install_mock_evaluator(monkeypatch, catalog, losses)
    result = _run(
        instance,
        _progressive(max_candidates=64, relative_improvement_threshold=0.05),
    )
    rounds = result.search_metadata.round_summaries
    assert rounds[1].low_improvement_streak == 1
    assert rounds[2].relative_improvement == pytest.approx(0.5)
    assert rounds[2].low_improvement_streak == 0
    assert result.search_metadata.termination_reason == "max_candidates"


def test_perfect_zero_stops_deterministically(monkeypatch, instance, catalog):
    _install_mock_evaluator(
        monkeypatch,
        catalog,
        lambda candidate_id, _seed: 0.0 if candidate_id == "candidate-0000" else 0.5,
    )
    result = _run(
        instance,
        _progressive(max_candidates=128, relative_improvement_threshold=0.05),
    )
    assert result.search_metadata.termination_reason == "perfect_zero_loss"
    assert result.search_metadata.rounds_completed == 1


def test_default_zero_threshold_perfect_loss_still_reaches_256(
    monkeypatch, instance, catalog
):
    _install_mock_evaluator(monkeypatch, catalog, lambda _candidate, _seed: 0.0)
    result = _run(instance, _progressive(max_candidates=256))
    assert DEFAULT_FIXED_BUDGET_MIN_ERROR_IMPROVEMENT_RATIO == 0.0
    assert result.search_metadata.termination_reason == "max_candidates"
    assert result.search_metadata.final_candidate_pool_size == 256


def test_holdout_is_once_after_freeze_and_never_controls_search(
    monkeypatch, instance, catalog
):
    calls, _seen = _install_mock_evaluator(
        monkeypatch,
        catalog,
        lambda candidate_id, seed: (
            0.99 if seed in HOLDOUT_SEEDS else 0.1 + int(candidate_id[-4:]) / 10_000
        ),
    )
    result = _run(instance, _progressive(max_candidates=32, min_rounds=99))
    holdout_calls = [item for item in calls if item[1] in HOLDOUT_SEEDS]
    assert len(holdout_calls) == len(HOLDOUT_SEEDS)
    assert {item[0] for item in holdout_calls} == {result.best_candidate_id}
    assert result.search_metadata.holdout_post_selection_only
    assert result.search_metadata.rounds_completed == 2


def test_checkpoint_resume_equivalence_without_learner_reruns(
    monkeypatch, tmp_path, instance, catalog
):
    checkpoint = tmp_path / "progressive.json"
    calls, seen = _install_mock_evaluator(monkeypatch, catalog)
    cfg = _optimization_config(
        checkpoint_path=str(checkpoint),
        checkpoint_key="progressive-resume",
        checkpoint_every_n_evaluations=1,
    )
    progressive = _progressive(max_candidates=32, min_rounds=99)
    original_save = progressive_module.CheckpointStore.save
    interrupted = False

    def save_then_interrupt(store, cache, run_state, *, status):
        nonlocal interrupted
        original_save(store, cache, run_state, status=status)
        completed = run_state.get("completed_progressive_round_summaries", ())
        if not interrupted and len(completed) == 1 and run_state.get("termination_reason") is None:
            interrupted = True
            raise RuntimeError("interrupt after progressive round one")

    monkeypatch.setattr(progressive_module.CheckpointStore, "save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="round one"):
        _run(instance, progressive, config=cfg)
    assert json.loads(checkpoint.read_text())["schema_version"] == SCHEMA_VERSION == 7
    completed_calls = len(calls)

    monkeypatch.setattr(progressive_module.CheckpointStore, "save", original_save)
    resumed = _run(
        instance,
        progressive,
        config=replace(cfg, resume_from_checkpoint=True),
    )
    assert resumed.search_metadata.resumed_from_checkpoint
    assert len(calls) == len(seen)
    assert len(calls) > completed_calls

    uninterrupted_calls, _ = _install_mock_evaluator(monkeypatch, catalog)
    uninterrupted = _run(
        instance,
        progressive,
        config=replace(cfg, checkpoint_path=None, checkpoint_key=""),
    )
    assert uninterrupted_calls
    assert resumed.best_candidate_id == uninterrupted.best_candidate_id
    assert resumed.search_metadata.round_summaries == uninterrupted.search_metadata.round_summaries
    assert resumed.comparison_evaluation == uninterrupted.comparison_evaluation
    assert resumed.holdout_evaluation == uninterrupted.holdout_evaluation
    assert resumed.actual_learner_run_count == uninterrupted.actual_learner_run_count
    assert resumed.candidate_catalog == uninterrupted.candidate_catalog


def test_checkpoint_progressive_compatibility_guard(
    monkeypatch, tmp_path, instance, catalog
):
    checkpoint = tmp_path / "guard.json"
    _install_mock_evaluator(monkeypatch, catalog)
    cfg = _optimization_config(
        checkpoint_path=str(checkpoint), checkpoint_key="guard"
    )
    _run(instance, _progressive(max_candidates=16), config=cfg)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        _run(
            instance,
            _progressive(max_candidates=32),
            config=replace(cfg, resume_from_checkpoint=True),
        )


def _assert_exact_resumed_result(resumed, uninterrupted):
    assert resumed.best_candidate_id == uninterrupted.best_candidate_id
    assert resumed.best_candidate == uninterrupted.best_candidate
    assert resumed.best_derived_candidate == uninterrupted.best_derived_candidate
    assert (
        resumed.search_metadata.round_summaries
        == uninterrupted.search_metadata.round_summaries
    )
    assert resumed.comparison_evaluation == uninterrupted.comparison_evaluation
    assert resumed.holdout_evaluation == uninterrupted.holdout_evaluation
    assert (
        resumed.search_metadata.termination_reason
        == uninterrupted.search_metadata.termination_reason
    )
    assert (
        resumed.search_metadata.final_candidate_pool_size
        == uninterrupted.search_metadata.final_candidate_pool_size
    )
    assert (
        resumed.search_metadata.total_valid_candidates_pregenerated
        == uninterrupted.search_metadata.total_valid_candidates_pregenerated
    )
    assert resumed.rejected_invalid_count == uninterrupted.rejected_invalid_count
    assert resumed.evaluation_attempt_count == uninterrupted.evaluation_attempt_count
    assert resumed.actual_learner_run_count == uninterrupted.actual_learner_run_count
    assert resumed.cache_hit_count == uninterrupted.cache_hit_count
    assert (
        resumed.unique_cached_seed_evaluation_count
        == uninterrupted.unique_cached_seed_evaluation_count
    )
    assert resumed.preflight_rejection_count == uninterrupted.preflight_rejection_count
    assert resumed.candidate_catalog == uninterrupted.candidate_catalog


def _interrupt_resume_exact(
    monkeypatch,
    tmp_path,
    instance,
    catalog,
    *,
    name,
    predicate,
    progressive=None,
    optimization_values=None,
    loss_model=None,
):
    progressive = progressive or _progressive(
        max_candidates=64,
        relative_improvement_threshold=0.0,
        min_rounds=99,
        checkpoint_every_n_new_evaluations=2,
    )
    values = dict(optimization_values or {})
    _install_mock_evaluator(monkeypatch, catalog, loss_model)
    uninterrupted = _run(
        instance,
        progressive,
        config=_optimization_config(**values),
    )

    checkpoint = tmp_path / f"{name}.json"
    calls, seen = _install_mock_evaluator(monkeypatch, catalog, loss_model)
    checkpoint_config = _optimization_config(
        **values,
        checkpoint_path=str(checkpoint),
        checkpoint_key=f"exact-{name}",
    )
    original_save = progressive_module.CheckpointStore.save
    interrupted = False

    def save_then_interrupt(store, cache, run_state, *, status):
        nonlocal interrupted
        original_save(store, cache, run_state, status=status)
        if not interrupted and predicate(run_state):
            interrupted = True
            raise RuntimeError(f"interrupt-{name}")

    monkeypatch.setattr(
        progressive_module.CheckpointStore, "save", save_then_interrupt
    )
    with pytest.raises(RuntimeError, match=f"interrupt-{name}"):
        _run(instance, progressive, config=checkpoint_config)
    assert interrupted
    calls_at_checkpoint = len(calls)

    monkeypatch.setattr(
        progressive_module.CheckpointStore, "save", original_save
    )
    resumed = _run(
        instance,
        progressive,
        config=replace(checkpoint_config, resume_from_checkpoint=True),
    )
    assert resumed.search_metadata.resumed_from_checkpoint
    assert len(calls) == len(seen)
    assert len(calls) >= calls_at_checkpoint
    _assert_exact_resumed_result(resumed, uninterrupted)
    return resumed, uninterrupted


def test_exact_resume_mid_round2_32x2_stage(
    monkeypatch, tmp_path, instance, catalog
):
    _interrupt_resume_exact(
        monkeypatch,
        tmp_path,
        instance,
        catalog,
        name="round2-32x2",
        predicate=lambda state: (
            state.get("controller_phase") == "halving"
            and state.get("current_progressive_round") == 2
            and state.get("current_seed_fidelity") == 2
            and 0 < state.get("current_stage_candidate_index", 0) < 16
        ),
    )


def test_exact_resume_mid_round3_8x8_stage(
    monkeypatch, tmp_path, instance, catalog
):
    _interrupt_resume_exact(
        monkeypatch,
        tmp_path,
        instance,
        catalog,
        name="round3-8x8",
        predicate=lambda state: (
            state.get("controller_phase") == "halving"
            and state.get("current_progressive_round") == 3
            and state.get("current_seed_fidelity") == 8
            and 0 < state.get("current_stage_candidate_index", 0) < 8
        ),
    )


def test_exact_resume_mid_round3_4x16_stage(
    monkeypatch, tmp_path, instance, catalog
):
    _interrupt_resume_exact(
        monkeypatch,
        tmp_path,
        instance,
        catalog,
        name="round3-4x16",
        predicate=lambda state: (
            state.get("controller_phase") == "halving"
            and state.get("current_progressive_round") == 3
            and state.get("current_seed_fidelity") == 16
            and 0 < state.get("current_stage_candidate_index", 0) < 4
        ),
    )


def test_exact_resume_during_common_comparison(
    monkeypatch, tmp_path, instance, catalog
):
    def losses(candidate_id, _seed):
        if candidate_id == "candidate-0000":
            return 0.2
        if candidate_id == "candidate-0016":
            return 0.1
        return 0.5

    _interrupt_resume_exact(
        monkeypatch,
        tmp_path,
        instance,
        catalog,
        name="comparison",
        predicate=lambda state: (
            state.get("controller_phase") == "comparison_challenger"
            and state.get("current_progressive_round") == 2
            and 8 < state.get("current_evaluation_next_seed_index", 0) < 16
        ),
        loss_model=losses,
    )


def test_exact_resume_mid_holdout(monkeypatch, tmp_path, instance, catalog):
    holdout = tuple(range(901, 909))
    _interrupt_resume_exact(
        monkeypatch,
        tmp_path,
        instance,
        catalog,
        name="holdout",
        predicate=lambda state: (
            state.get("controller_phase") == "holdout"
            and 0 < state.get("current_evaluation_next_seed_index", 0) < len(holdout)
        ),
        optimization_values={"holdout_seeds": holdout},
    )


def test_progressive_schema_one_checkpoint_is_rejected(
    monkeypatch, tmp_path, instance, catalog
):
    checkpoint = tmp_path / "old-progressive-schema.json"
    _install_mock_evaluator(monkeypatch, catalog)
    cfg = _optimization_config(
        checkpoint_path=str(checkpoint), checkpoint_key="progressive-schema"
    )
    progressive = _progressive(max_candidates=16)
    _run(instance, progressive, config=cfg)
    payload = json.loads(checkpoint.read_text())
    assert payload["compatibility"]["progressive_search_schema"] == 5
    payload["compatibility"]["progressive_search_schema"] = 1
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        _run(
            instance,
            progressive,
            config=replace(cfg, resume_from_checkpoint=True),
        )


def test_progressive_cadence_reduces_checkpoint_rewrites(
    monkeypatch, tmp_path, instance, catalog
):
    write_counts = {"cadence-1.json": 0, "cadence-16.json": 0}
    original_save = progressive_module.CheckpointStore.save

    def counting_save(store, cache, run_state, *, status):
        write_counts[store.path.name] += 1
        return original_save(store, cache, run_state, status=status)

    monkeypatch.setattr(progressive_module.CheckpointStore, "save", counting_save)
    for cadence in (1, 16):
        _install_mock_evaluator(monkeypatch, catalog)
        cfg = _optimization_config(
            checkpoint_path=str(tmp_path / f"cadence-{cadence}.json"),
            checkpoint_key=f"cadence-{cadence}",
        )
        _run(
            instance,
            _progressive(
                max_candidates=32,
                min_rounds=99,
                checkpoint_every_n_new_evaluations=cadence,
            ),
            config=cfg,
        )
    assert write_counts["cadence-16.json"] < write_counts["cadence-1.json"]


def test_reproducible_outcome_no_refinement_and_no_oracle_sampling(
    monkeypatch, instance, catalog
):
    _install_mock_evaluator(monkeypatch, catalog)
    first = _run(instance, _progressive(max_candidates=16))
    _install_mock_evaluator(monkeypatch, catalog)
    second = _run(instance, _progressive(max_candidates=16))
    assert first.best_candidate_id == second.best_candidate_id
    assert first.candidate_catalog == second.candidate_catalog
    assert not first.search_metadata.fixed_budget_epsilon_refinement_enabled
    source = inspect.getsource(progressive_module)
    assert "hidden_partition" not in source
    assert "latent block" not in source
    assert "encoder truth" not in source
