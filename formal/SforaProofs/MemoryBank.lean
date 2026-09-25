import SforaProofs.Core
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

namespace SforaProofs

/-- Exact real-valued dot score used by unit-vector retrieval. -/
def dotScore {n : ℕ} (q g : Fin n → ℝ) : ℝ :=
  ∑ i, q i * g i

/-- A stale gallery member changes its dot score by at most the query-weighted
    coordinate drift. This is a conditional arithmetic bound, not a bound on
    model drift or on floating-point CUDA arithmetic. -/
theorem dotScore_stale_error {n : ℕ} (q stale fresh : Fin n → ℝ) :
    |dotScore q fresh - dotScore q stale| ≤
      ∑ i, |q i| * |fresh i - stale i| := by
  unfold dotScore
  rw [← Finset.sum_sub_distrib]
  calc
    |∑ i, (q i * fresh i - q i * stale i)|
        ≤ ∑ i, |q i * fresh i - q i * stale i| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |q i| * |fresh i - stale i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [← mul_sub, abs_mul]

/-- Per-coordinate drift premises yield a computable score radius. -/
theorem dotScore_stale_error_of_coordinate_bounds {n : ℕ}
    (q stale fresh radius : Fin n → ℝ)
    (h : ∀ i, |fresh i - stale i| ≤ radius i) :
    |dotScore q fresh - dotScore q stale| ≤
      ∑ i, |q i| * radius i := by
  calc
    |dotScore q fresh - dotScore q stale|
        ≤ ∑ i, |q i| * |fresh i - stale i| := dotScore_stale_error q stale fresh
    _ ≤ ∑ i, |q i| * radius i := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (h i) (abs_nonneg _)

/-- Uniformly bounded gallery drift supplies the `ErrWithin` premise used by
    the existing conditional top-k and label-recall theorems. -/
theorem staleBank_errWithin {n : ℕ} {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (q : Fin n → ℝ) (fresh stale : ι → Fin n → ℝ)
    (radius : ι → Fin n → ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (hcoordinate : ∀ row ∈ S, ∀ i, |fresh row i - stale row i| ≤ radius row i)
    (haggregate : ∀ row ∈ S, (∑ i, |q i| * radius row i) ≤ ε) :
    ErrWithin (fun row => dotScore q (fresh row))
      (fun row => dotScore q (stale row)) ε S := by
  refine ⟨hε, ?_⟩
  intro row hrow
  have hbound := dotScore_stale_error_of_coordinate_bounds
    q (stale row) (fresh row) (radius row) (hcoordinate row hrow)
  have hle := (abs_le.mp (hbound.trans (haggregate row hrow)))
  constructor <;> linarith

end SforaProofs
