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

_COMPACT_METRIC_EXPORTS = frozenset(
    {
        "CompactMetricConfig",
        "CompactMetricEncoder",
        "CompactMetricFitResult",
        "CompactMetricModule",
        "CompactMetricSelectionFold",
        "CompactMetricSelectionResult",
        "choose_compact_metric_projection",
        "fit_compact_metric_projection",
        "select_compact_metric_projection",
    }
)

_FOLDABLE_LINEAR_EXPORTS = frozenset({"FoldableLinear"})

_FACTORIZED_RESIDUAL_ANN_EXPORTS = frozenset(
    {
        "CandidateResult",
        "FactorizedResidualArtifact",
        "FactorizedResidualComponents",
        "FactorizedResidualPostings",
        "FactorizedResidualSpec",
        "PortableCandidateIndex",
        "VectorStoreIdentity",
        "write_factorized_residual_artifact",
        "write_factorized_residual_artifact_from_role_files",
    }
)

_FACTORIZED_RESIDUAL_NATIVE_EXPORTS = frozenset(
    {
        "NativeBackend",
        "NativeCandidateIndex",
        "NativeDirectBusyError",
        "NativeDirectQuiescenceError",
        "NativeDirectUnsupportedError",
        "NativeExactReranker",
        "compile_factorized_residual_backend",
    }
)

_FACTORIZED_RESIDUAL_INDEX_EXPORTS = frozenset(
    {
        "FactorizedResidualIndex",
        "FactorizedResidualMemoryLedger",
        "FactorizedResidualSearchResult",
    }
)

_MODEL_SOUP_EXPORTS = frozenset({"average_compatible_model_states"})

_VECTOR_STORE_EXPORTS = frozenset(
    {
        "ExactReranker",
        "ExactSearchResult",
        "DirectIoVectorStore",
        "InsufficientCandidatesError",
        "MemoryVectorStore",
        "PreadVectorStore",
        "VectorReadEvidence",
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

_RATE_MATCHED_PRODUCT_QUANTIZATION_EXPORTS = frozenset(
    {
        "RateMatchedProductQuantizer",
        "fit_rate_matched_product_quantizer",
    }
)

_PRODUCT_QUANTIZATION_EXPORTS = frozenset(
    {
        "ProductQuantizationSpec",
        "balanced_product_quantization_spec",
    }
)

_PQ_CANDIDATE_SCORING_EXPORTS = frozenset(
    {
        "CompiledPqCandidateScorer",
        "PqCandidateResult",
        "PqCandidateScoringSpec",
        "compile_pq_candidate_scorer",
    }
)

_PROGRESSIVE_RESIDUAL_QUANTIZATION_EXPORTS = frozenset(
    {
        "ProgressiveCandidateResult",
        "ProgressiveResidualCodes",
        "ProgressiveResidualQuantizer",
        "ProgressiveResidualSpec",
        "fit_progressive_residual_quantizer",
        "pack_residual_bitplanes",
        "unpack_residual_prefix",
    }
)

_PROGRESSIVE_RESIDUAL_SCORING_EXPORTS = frozenset(
    {
        "CompiledProgressiveCandidateScorer",
        "ProgressiveScoringSpec",
        "ProgressiveScoringResult",
        "compile_progressive_candidate_scorer",
        "progressive_candidate_scores",
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
        "ClassBalancedAnchorSchedule",
        "class_balanced_anchor_schedule",
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
        "exposure_normalized_update_count",
        "multi_similarity_hard_negative_loss",
        "retrieval_local_rank_distillation_loss",
        "retrieval_impact_weighted_pairwise_loss",
        "positive_coverage_hard_negative_loss",
        "stable_different_class_topk",
        "supervised_contrastive_hard_negative_loss",
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
    if name in _COMPACT_METRIC_EXPORTS:
        module = import_module("sfora.compact_metric")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _FOLDABLE_LINEAR_EXPORTS:
        module = import_module("sfora.foldable_linear")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _FACTORIZED_RESIDUAL_ANN_EXPORTS:
        module = import_module("sfora.factorized_residual_ann")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _FACTORIZED_RESIDUAL_NATIVE_EXPORTS:
        module = import_module("sfora.factorized_residual_native")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _FACTORIZED_RESIDUAL_INDEX_EXPORTS:
        module = import_module("sfora.factorized_residual_index")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _MODEL_SOUP_EXPORTS:
        module = import_module("sfora.model_soup")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _VECTOR_STORE_EXPORTS:
        module = import_module("sfora.vector_store")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _RELATIONAL_COMPACTION_EXPORTS:
        module = import_module("sfora.joint_relational_compaction")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _RATE_MATCHED_PRODUCT_QUANTIZATION_EXPORTS:
        module = import_module("sfora.rate_matched_product_quantization")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _PRODUCT_QUANTIZATION_EXPORTS:
        module = import_module("sfora.product_quantization")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _PQ_CANDIDATE_SCORING_EXPORTS:
        module = import_module("sfora.pq_candidate_scoring")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _PROGRESSIVE_RESIDUAL_QUANTIZATION_EXPORTS:
        module = import_module("sfora.progressive_residual_quantization")
        value = cast(object, getattr(module, name))
        globals()[name] = value
        return value
    if name in _PROGRESSIVE_RESIDUAL_SCORING_EXPORTS:
        module = import_module("sfora.progressive_residual_scoring")
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
    "CandidateResult",
    "CompactMetricConfig",
    "CompactMetricEncoder",
    "CompactMetricFitResult",
    "CompactMetricModule",
    "CompactMetricSelectionFold",
    "CompactMetricSelectionResult",
    "ExperimentResult",
    "FactorizedResidualArtifact",
    "FactorizedResidualComponents",
    "FactorizedResidualPostings",
    "FactorizedResidualSpec",
    "FactorizedResidualIndex",
    "FactorizedResidualMemoryLedger",
    "FactorizedResidualSearchResult",
    "ExactReranker",
    "ExactSearchResult",
    "DirectIoVectorStore",
    "InsufficientCandidatesError",
    "MemoryVectorStore",
    "NativeBackend",
    "NativeCandidateIndex",
    "NativeDirectBusyError",
    "NativeDirectQuiescenceError",
    "NativeDirectUnsupportedError",
    "NativeExactReranker",
    "PreadVectorStore",
    "PortableCandidateIndex",
    "compile_factorized_residual_backend",
    "FoldableLinear",
    "average_compatible_model_states",
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
    "compile_pq_candidate_scorer",
    "compile_progressive_candidate_scorer",
    "evaluate",
    "grid",
    "progressive_candidate_scores",
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
    "PqCandidateResult",
    "PqCandidateScoringSpec",
    "ProjectionHeadTrainingConfig",
    "ProjectionHeadTrainingResult",
    "ProjectionTrainingConfig",
    "ProjectionTrainingResult",
    "ProbeScore",
    "ProductQuantizationSpec",
    "CompiledPqCandidateScorer",
    "CompiledProgressiveCandidateScorer",
    "ProgressiveCandidateResult",
    "ProgressiveResidualCodes",
    "ProgressiveResidualQuantizer",
    "ProgressiveResidualSpec",
    "ProgressiveScoringSpec",
    "ProgressiveScoringResult",
    "RetrievalScore",
    "RemoteRunConfig",
    "RemoteRunPlan",
    "RemoteStep",
    "RateMatchedProductQuantizer",
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
    "ClassBalancedAnchorSchedule",
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
    "balanced_product_quantization_spec",
    "choose_compact_metric_projection",
    "fit_sfora_projection",
    "fit_compact_metric_projection",
    "fit_relational_linear_compaction",
    "fit_relational_linear_encoder",
    "fit_rate_matched_product_quantizer",
    "fit_progressive_residual_quantizer",
    "apply_normalized_affine",
    "deterministic_class_partition",
    "fit_centered_pca",
    "fit_ridge_affine",
    "fit_teacher_guided_projection",
    "select_compact_metric_projection",
    "cross_dimensional_anchor_distillation_loss",
    "cross_dimensional_relational_distillation_loss",
    "cross_dimensional_similarity_distillation_loss",
    "class_balanced_anchor_schedule",
    "embedding_geometry_diagnostics",
    "exposure_normalized_update_count",
    "multi_similarity_hard_negative_loss",
    "retrieval_local_rank_distillation_loss",
    "positive_coverage_hard_negative_loss",
    "stable_different_class_topk",
    "supervised_contrastive_hard_negative_loss",
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
    "pack_residual_bitplanes",
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
    "unpack_residual_prefix",
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
    "VectorStoreIdentity",
    "VectorReadEvidence",
    "write_factorized_residual_artifact",
    "write_factorized_residual_artifact_from_role_files",
]
