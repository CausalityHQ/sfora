"""Core tools for group-based similarity learning experiments."""

from importlib import import_module
from typing import cast

from sfora.ablation import (
    SyntheticAblationConfig,
    SyntheticAblationResult,
    SyntheticAblationTrial,
    run_synthetic_ablation,
    write_ablation_report,
)
from sfora.api import SforaProjector, fit_sfora_projection
from sfora.compose import (
    Head,
    Identity,
    Join,
    L2Normalize,
    Pca,
    Pipeline,
    Projection,
    RetrievalReport,
    compare,
    evaluate,
    grid,
)
from sfora.data import (
    ImageExample,
    TextExample,
    TextGroupTriplet,
    TextTriplet,
    load_image_retrieval_examples,
    load_imdb_examples,
    mine_group_triplets,
    mine_triplets,
    select_balanced_examples,
    select_labeled_image_examples,
)
from sfora.encoder_ablation import (
    EncoderAblationConfig,
    EncoderAblationResult,
    EncoderAblationTrial,
    run_encoder_ablation,
    write_encoder_ablation_report,
)
from sfora.encoder_training import (
    EncoderTrainingConfig,
    EncoderTrainingMethodMetrics,
    EncoderTrainingResult,
    run_encoder_training,
    run_encoder_training_on_split,
    write_encoder_training_report,
)
from sfora.evaluation import (
    EmbeddingSpaceDiagnostics,
    ProbeScore,
    RetrievalScore,
    embedding_space_diagnostics_on_split,
    linear_probe_score,
    linear_probe_score_on_split,
    retrieval_score_on_split,
)
from sfora.experiments import (
    ExperimentResult,
    MethodMetrics,
    SyntheticExperimentConfig,
    TrainableSyntheticExperimentConfig,
    run_synthetic_experiment,
    run_trainable_synthetic_experiment,
    write_experiment_report,
)
from sfora.image_benchmark import (
    ImageBenchmarkConfig,
    ImageBenchmarkMethodMetrics,
    ImageBenchmarkResult,
    ImageObjective,
    ImageRetrievalMetrics,
    image_self_retrieval_score,
    objective_display_name,
    run_image_benchmark,
    write_image_benchmark_report,
)
from sfora.losses import group_triplet_margin_loss, triplet_margin_loss
from sfora.publication import (
    HfPublishBundle,
    HfPublishConfig,
    HfPublishResult,
    build_hf_publish_bundle,
    publish_hf_bundle,
)
from sfora.remote import (
    RemoteRunConfig,
    RemoteRunPlan,
    RemoteStep,
    build_remote_run_plan,
    write_remote_run_plan,
)
from sfora.report import (
    ReportConfig,
    build_html_report,
    build_markdown_report,
    build_site_data,
    write_hf_model_card,
    write_html_report,
    write_markdown_report,
    write_site_data,
)
from sfora.text_baselines import (
    SentenceTransformerBaselineConfig,
    SentenceTransformerModelSuiteConfig,
    TextBaselineConfig,
    TextBaselineResult,
    TextMethodMetrics,
    run_sentence_transformer_baseline,
    run_sentence_transformer_model_suite,
    run_text_baseline,
    write_text_baseline_report,
)
from sfora.training import (
    ProjectionHeadTrainingConfig,
    ProjectionHeadTrainingResult,
    ProjectionTrainingConfig,
    ProjectionTrainingResult,
    train_embedding_table,
    train_projection_head,
)

_PACKED_INT4_EXPORTS = frozenset(
    {
        "PackedInt4Embeddings",
        "ResidentInt4Gallery",
        "pack_int4_unit_embeddings",
    }
)

_RELATIONAL_COMPACTION_EXPORTS = frozenset(
    {
        "PackedInt8Embeddings",
        "RelationalLinearEncoder",
        "RelationalLinearTrainingConfig",
        "fit_relational_linear_compaction",
        "fit_relational_linear_encoder",
        "pack_int8_unit_embeddings",
    }
)

_REPRESENTATION_CEILING_EXPORTS = frozenset(
    {
        "AffineMap",
        "CenteredPcaTransform",
        "ClassDisjointPartition",
        "TeacherGuidedProjection",
        "apply_normalized_affine",
        "deterministic_class_partition",
        "fit_centered_pca",
        "fit_ridge_affine",
        "fit_teacher_guided_projection",
    }
)

_TEACHER_ANCHORED_EXPORTS = frozenset(
    {
        "TeacherAnchoredConfig",
        "TeacherAnchorSchedule",
        "TeacherNeighborBatches",
        "TeacherNeighborRanking",
        "TeacherAnchoredLoss",
        "TeacherAnchoredNumericalError",
        "EmbeddingGeometryDiagnostics",
        "cross_dimensional_anchor_distillation_loss",
        "cross_dimensional_relational_distillation_loss",
        "cross_dimensional_similarity_distillation_loss",
        "embedding_geometry_diagnostics",
        "retrieval_local_rank_distillation_loss",
        "retrieval_impact_weighted_pairwise_loss",
        "teacher_anchor_schedule",
        "teacher_anchored_forward",
        "teacher_anchored_input_sha256",
        "teacher_anchored_loss",
        "teacher_neighbor_batches",
        "teacher_neighbor_ranking",
        "verify_teacher_anchor_schedule",
        "verify_teacher_neighbor_batches",
    }
)

_TEACHER_ANCHORED_SCHEDULE_IO_EXPORTS = frozenset(
    {
        "SealedTeacherAnchoredSchedule",
        "TeacherAnchoredScheduleBinding",
        "build_teacher_anchored_schedule",
        "canonical_teacher_anchored_schedule_bytes",
        "parse_teacher_anchored_schedule_bytes",
        "parse_teacher_anchored_schedule_for_inputs",
        "teacher_anchored_schedule_rows",
    }
)


def __getattr__(name: str) -> object:
    """Load optional PyTorch compaction symbols only when explicitly requested."""

    if name in _PACKED_INT4_EXPORTS:
        module = import_module("sfora.packed_int4")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _RELATIONAL_COMPACTION_EXPORTS:
        module = import_module("sfora.joint_relational_compaction")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _REPRESENTATION_CEILING_EXPORTS:
        module = import_module("sfora.representation_ceiling")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _TEACHER_ANCHORED_EXPORTS:
        module = import_module("sfora.teacher_anchored_distillation")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _TEACHER_ANCHORED_SCHEDULE_IO_EXPORTS:
        module = import_module("sfora.teacher_anchored_schedule_io")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExperimentResult",
    "EncoderTrainingConfig",
    "EncoderTrainingMethodMetrics",
    "EncoderTrainingResult",
    "EncoderAblationConfig",
    "EncoderAblationResult",
    "EncoderAblationTrial",
    "EmbeddingSpaceDiagnostics",
    "AffineMap",
    "CenteredPcaTransform",
    "ClassDisjointPartition",
    "SforaProjector",
    "Projection",
    "Identity",
    "L2Normalize",
    "Pca",
    "Head",
    "Pipeline",
    "Join",
    "RetrievalReport",
    "compare",
    "evaluate",
    "grid",
    "HfPublishBundle",
    "HfPublishConfig",
    "HfPublishResult",
    "ImageBenchmarkConfig",
    "ImageBenchmarkMethodMetrics",
    "ImageBenchmarkResult",
    "ImageExample",
    "ImageObjective",
    "ImageRetrievalMetrics",
    "MethodMetrics",
    "PackedInt4Embeddings",
    "PackedInt8Embeddings",
    "ProjectionHeadTrainingConfig",
    "ProjectionHeadTrainingResult",
    "ProjectionTrainingConfig",
    "ProjectionTrainingResult",
    "ProbeScore",
    "RetrievalScore",
    "RemoteRunConfig",
    "RemoteRunPlan",
    "RemoteStep",
    "RelationalLinearEncoder",
    "RelationalLinearTrainingConfig",
    "ReportConfig",
    "ResidentInt4Gallery",
    "SyntheticAblationConfig",
    "SyntheticAblationResult",
    "SyntheticAblationTrial",
    "SentenceTransformerBaselineConfig",
    "SentenceTransformerModelSuiteConfig",
    "SyntheticExperimentConfig",
    "TextExample",
    "TextGroupTriplet",
    "TextTriplet",
    "TextBaselineConfig",
    "TextBaselineResult",
    "TextMethodMetrics",
    "TeacherGuidedProjection",
    "TeacherAnchoredConfig",
    "TeacherAnchorSchedule",
    "TeacherNeighborBatches",
    "TeacherNeighborRanking",
    "TeacherAnchoredLoss",
    "TeacherAnchoredNumericalError",
    "TeacherAnchoredScheduleBinding",
    "SealedTeacherAnchoredSchedule",
    "build_teacher_anchored_schedule",
    "EmbeddingGeometryDiagnostics",
    "TrainableSyntheticExperimentConfig",
    "build_html_report",
    "build_markdown_report",
    "build_site_data",
    "build_hf_publish_bundle",
    "build_remote_run_plan",
    "fit_sfora_projection",
    "fit_relational_linear_compaction",
    "fit_relational_linear_encoder",
    "apply_normalized_affine",
    "deterministic_class_partition",
    "fit_centered_pca",
    "fit_ridge_affine",
    "fit_teacher_guided_projection",
    "cross_dimensional_anchor_distillation_loss",
    "cross_dimensional_relational_distillation_loss",
    "cross_dimensional_similarity_distillation_loss",
    "embedding_geometry_diagnostics",
    "retrieval_local_rank_distillation_loss",
    "teacher_anchor_schedule",
    "teacher_anchored_forward",
    "teacher_anchored_input_sha256",
    "teacher_anchored_loss",
    "teacher_neighbor_batches",
    "teacher_neighbor_ranking",
    "verify_teacher_anchor_schedule",
    "verify_teacher_neighbor_batches",
    "canonical_teacher_anchored_schedule_bytes",
    "parse_teacher_anchored_schedule_bytes",
    "parse_teacher_anchored_schedule_for_inputs",
    "teacher_anchored_schedule_rows",
    "group_triplet_margin_loss",
    "embedding_space_diagnostics_on_split",
    "image_self_retrieval_score",
    "load_imdb_examples",
    "load_image_retrieval_examples",
    "linear_probe_score",
    "linear_probe_score_on_split",
    "retrieval_score_on_split",
    "mine_group_triplets",
    "mine_triplets",
    "objective_display_name",
    "pack_int4_unit_embeddings",
    "pack_int8_unit_embeddings",
    "run_sentence_transformer_baseline",
    "run_sentence_transformer_model_suite",
    "run_encoder_ablation",
    "run_encoder_training",
    "run_encoder_training_on_split",
    "run_image_benchmark",
    "run_synthetic_ablation",
    "run_synthetic_experiment",
    "run_text_baseline",
    "run_trainable_synthetic_experiment",
    "select_balanced_examples",
    "select_labeled_image_examples",
    "publish_hf_bundle",
    "train_embedding_table",
    "train_projection_head",
    "triplet_margin_loss",
    "write_ablation_report",
    "write_experiment_report",
    "write_encoder_training_report",
    "write_encoder_ablation_report",
    "write_hf_model_card",
    "write_html_report",
    "write_image_benchmark_report",
    "write_markdown_report",
    "write_remote_run_plan",
    "write_site_data",
    "write_text_baseline_report",
]
