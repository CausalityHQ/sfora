import SforaProofs.Core
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace SforaProofs

/-- Squared Euclidean distance between finite real vectors. -/
def squaredDistance {n : ℕ} (q g : Fin n → ℝ) : ℝ :=
  ∑ i, (q i - g i) ^ 2

/-- The score used for Euclidean retrieval after dropping a fixed query norm. -/
def euclideanScore {n : ℕ} (q g : Fin n → ℝ) : ℝ :=
  2 * ∑ i, q i * g i - ∑ i, (g i) ^ 2

theorem euclideanScore_eq_queryNorm_sub_distance {n : ℕ} (q g : Fin n → ℝ) :
    euclideanScore q g = (∑ i, (q i) ^ 2) - squaredDistance q g := by
  unfold euclideanScore squaredDistance
  simp_rw [sub_sq, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have h : (∑ i, 2 * q i * g i) = 2 * ∑ i, q i * g i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [h]
  ring

/-- For one query, score-descending and squared-distance-ascending have identical
    comparisons, including the lower-ordinal rule for ties. -/
theorem euclidean_beats_iff {n : ℕ} (q : Fin n → ℝ)
    (gallery : ℕ → Fin n → ℝ) (ordinalX ordinalY : ℕ) :
    beats (fun i => euclideanScore q (gallery i)) ordinalX ordinalY ↔
      beats (fun i => -squaredDistance q (gallery i)) ordinalX ordinalY := by
  rw [beats_iff, beats_iff]
  rw [euclideanScore_eq_queryNorm_sub_distance q (gallery ordinalX),
    euclideanScore_eq_queryNorm_sub_distance q (gallery ordinalY)]
  constructor
  · rintro (h | ⟨h, hOrdinal⟩)
    · left; linarith
    · right; constructor <;> [linarith; exact hOrdinal]
  · rintro (h | ⟨h, hOrdinal⟩)
    · left; linarith
    · right; constructor <;> [linarith; exact hOrdinal]

/-- Exact real-valued SOP score selects the same rows as minimum squared distance. -/
theorem euclidean_topK_eq {n : ℕ} (q : Fin n → ℝ) (gallery : ℕ → Fin n → ℝ)
    (S : Finset ℕ) (k : ℕ) :
    topK (fun i => euclideanScore q (gallery i)) S k =
      topK (fun i => -squaredDistance q (gallery i)) S k := by
  ext x
  rw [mem_topK, mem_topK]
  have hfilter :
      S.filter (fun y => beats (fun i => euclideanScore q (gallery i)) y x) =
        S.filter (fun y => beats (fun i => -squaredDistance q (gallery i)) y x) := by
    apply Finset.filter_congr
    intro y _
    exact euclidean_beats_iff q gallery y x
  rw [hfilter]

end SforaProofs
