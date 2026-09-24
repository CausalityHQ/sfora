import SforaProofs.Core
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

namespace SforaProofs

/-- An ideal quantized-code cosine `c` after both stored inverse norms have
    relative errors `δq` and `δg`. The integer dot product is included in `c`. -/
def roundedNormScore (c δq δg : ℝ) : ℝ := c * (1 + δq) * (1 + δg)

/-- Conditional bound for the f16 inverse-norm stage, in exact real arithmetic.
    The premise bounds each stored scale's relative error by `u`. -/
theorem roundedNormScore_error (c δq δg u : ℝ)
    (hu : 0 ≤ u) (hc : |c| ≤ 1) (hq : |δq| ≤ u) (hg : |δg| ≤ u) :
    |roundedNormScore c δq δg - c| ≤ 2 * u + u * u := by
  have hproduct : |δq| * |δg| ≤ u * u :=
    (mul_le_mul_of_nonneg_right hq (abs_nonneg _)).trans
      (mul_le_mul_of_nonneg_left hg hu)
  have hinner : |δq + δg + δq * δg| ≤ 2 * u + u * u := by
    calc
      |δq + δg + δq * δg| ≤ |δq + δg| + |δq * δg| := abs_add_le _ _
      _ ≤ |δq| + |δg| + |δq| * |δg| := by
        rw [abs_mul]
        simpa [add_comm, add_left_comm, add_assoc] using
          (add_le_add_right (abs_add_le δq δg) (|δq| * |δg|))
      _ ≤ 2 * u + u * u := by linarith
  have hpositive : 0 ≤ 2 * u + u * u := by nlinarith [sq_nonneg u]
  have hid : roundedNormScore c δq δg - c = c * (δq + δg + δq * δg) := by
    unfold roundedNormScore
    ring
  rw [hid, abs_mul]
  calc
    |c| * |δq + δg + δq * δg|
        ≤ |c| * (2 * u + u * u) := mul_le_mul_of_nonneg_left hinner (abs_nonneg _)
    _ ≤ 2 * u + u * u := by nlinarith [mul_nonneg (sub_nonneg.mpr hc) hpositive]

/-- A fixed positive query-scale factor has no effect on score and ordinal
    ordering, even when gallery scales vary by row. -/
theorem commonQueryScale_topK_eq {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (k : ℕ) (ideal δg : ι → ℝ) (δq : ℝ)
    (hpositive : 0 < 1 + δq) :
    topK (fun i => roundedNormScore (ideal i) δq (δg i)) S k =
      topK (fun i => ideal i * (1 + δg i)) S k := by
  ext x
  rw [mem_topK, mem_topK]
  have hfilter :
      S.filter (fun y => beats (fun i => roundedNormScore (ideal i) δq (δg i)) y x) =
        S.filter (fun y => beats (fun i => ideal i * (1 + δg i)) y x) := by
    apply Finset.filter_congr
    intro y _
    rw [beats_iff, beats_iff]
    have hscore (i : ι) :
        roundedNormScore (ideal i) δq (δg i) =
          (1 + δq) * (ideal i * (1 + δg i)) := by
      unfold roundedNormScore
      ring
    rw [hscore y, hscore x]
    constructor
    · rintro (h | ⟨h, hOrdinal⟩)
      · left
        by_contra hnot
        have hreverse := le_of_not_gt hnot
        nlinarith [mul_nonneg hpositive.le (sub_nonneg.mpr hreverse)]
      · right; exact ⟨(mul_left_cancel₀ hpositive.ne' h), hOrdinal⟩
    · rintro (h | ⟨h, hOrdinal⟩)
      · left
        nlinarith [mul_pos hpositive (sub_pos.mpr h)]
      · right; exact ⟨congrArg ((1 + δq) * ·) h, hOrdinal⟩
  rw [hfilter]

/-- A rounded gallery inverse norm alone changes ideal code cosine by at most
    its relative-scale error, assuming ideal code cosine has magnitude ≤ 1. -/
theorem galleryScale_error (c δg u : ℝ) (hc : |c| ≤ 1) (hg : |δg| ≤ u)
    (hu : 0 ≤ u) :
    |c * (1 + δg) - c| ≤ u := by
  have hid : c * (1 + δg) - c = c * δg := by ring
  rw [hid, abs_mul]
  calc
    |c| * |δg| ≤ |c| * u := mul_le_mul_of_nonneg_left hg (abs_nonneg _)
    _ ≤ u := by nlinarith [mul_nonneg (sub_nonneg.mpr hc) hu]

/-- In exact score arithmetic, the row-varying gallery scale alone supplies
    an `ErrWithin` radius `u` for ideal quantized-code cosine. -/
theorem galleryScale_errWithin {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (ideal δg : ι → ℝ) (u : ℝ) (hu : 0 ≤ u)
    (h : ∀ i ∈ S, |ideal i| ≤ 1 ∧ |δg i| ≤ u) :
    ErrWithin ideal (fun i => ideal i * (1 + δg i)) u S := by
  refine ⟨hu, ?_⟩
  intro i hi
  obtain ⟨hc, hg⟩ := h i hi
  have herr := galleryScale_error (ideal i) (δg i) u hc hg hu
  have hle := abs_le.mp herr
  constructor <;> linarith

/-- A robust ideal-code-cosine winner survives the two rounded scales when
    score arithmetic is exact and the fixed query scale remains positive. -/
theorem robust_mem_roundedNormScore {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (k : ℕ) (ideal δg : ι → ℝ) (δq u : ℝ) (x : ι)
    (hu : 0 ≤ u) (hu1 : u < 1) (hq : |δq| ≤ u)
    (h : ∀ i ∈ S, |ideal i| ≤ 1 ∧ |δg i| ≤ u)
    (hx : x ∈ robust ideal u S k) :
    x ∈ topK (fun i => roundedNormScore (ideal i) δq (δg i)) S k := by
  have hpositive : 0 < 1 + δq := by
    have hlow := (abs_le.mp hq).1
    linarith
  rw [commonQueryScale_topK_eq S k ideal δg δq hpositive]
  exact robust_mem_topK_approx (galleryScale_errWithin S ideal δg u hu h) hx

/-- An additional absolute error `ρ` covers any difference between the observed
    score and the real-valued formula, including f32 score arithmetic. -/
theorem observedPackedScore_error (c δq δg observed u ρ : ℝ)
    (hu : 0 ≤ u) (hc : |c| ≤ 1) (hq : |δq| ≤ u) (hg : |δg| ≤ u)
    (hobserved : |observed - roundedNormScore c δq δg| ≤ ρ) :
    |observed - c| ≤ ρ + (2 * u + u * u) := by
  have hnorm := roundedNormScore_error c δq δg u hu hc hq hg
  calc
    |observed - c| =
        |(observed - roundedNormScore c δq δg) +
          (roundedNormScore c δq δg - c)| := by congr 1; ring
    _ ≤ |observed - roundedNormScore c δq δg| +
          |roundedNormScore c δq δg - c| := abs_add_le _ _
    _ ≤ ρ + (2 * u + u * u) := add_le_add hobserved hnorm

/-- The packed-score error premises instantiate `ErrWithin` for quantized-code
    cosine, so its robust-margin top-k recall theorem applies to that score. -/
theorem packedNorm_errWithin {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (ideal observed δq δg : ι → ℝ) (u ρ : ℝ)
    (hu : 0 ≤ u) (hρ : 0 ≤ ρ)
    (h : ∀ i ∈ S, |ideal i| ≤ 1 ∧ |δq i| ≤ u ∧ |δg i| ≤ u ∧
      |observed i - roundedNormScore (ideal i) (δq i) (δg i)| ≤ ρ) :
    ErrWithin ideal observed (ρ + (2 * u + u * u)) S := by
  refine ⟨by nlinarith [sq_nonneg u], ?_⟩
  intro i hi
  obtain ⟨hc, hq, hg, hround⟩ := h i hi
  have herr := observedPackedScore_error (ideal i) (δq i) (δg i) (observed i)
    u ρ hu hc hq hg hround
  have hle := abs_le.mp herr
  constructor <;> linarith

/-- With the explicit `2⁻¹¹` premise on each stored scale's total relative
    error, the real-valued norm contribution is at most `4097/4194304`, about
    `0.000977`. The premise must account for scale construction and storage. -/
theorem stored_scale_error_of_2pow11_premise (c δq δg : ℝ)
    (hc : |c| ≤ 1) (hq : |δq| ≤ (1 / 2048 : ℝ))
    (hg : |δg| ≤ (1 / 2048 : ℝ)) :
    |roundedNormScore c δq δg - c| ≤ (4097 / 4194304 : ℝ) := by
  have h := roundedNormScore_error c δq δg (1 / 2048) (by norm_num) hc hq hg
  norm_num at h ⊢
  exact h

end SforaProofs
