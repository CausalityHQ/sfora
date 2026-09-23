import Mathlib.Data.Finset.Sort
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace SforaProofs

/-- The number of gallery blocks for a positive block width. -/
def blockCount (rows width : ℕ) : ℕ := (rows + width - 1) / width

/-- Positive-width blocks cover the entire row count. -/
theorem rows_le_blockCount_mul (rows width : ℕ) (hw : 0 < width) :
    rows ≤ blockCount rows width * width := by
  unfold blockCount
  have hmod := Nat.mod_lt (rows + width - 1) hw
  have hdiv := Nat.div_add_mod (rows + width - 1) width
  rw [Nat.mul_comm width ((rows + width - 1) / width)] at hdiv
  omega

/-- With the production 128-row pad bound, every materialized lane index is
strictly below the signed `i32::MAX` sentinel. This is an arithmetic contract;
the Rust constructor must enforce its premise. -/
theorem padded_ordinal_lt_i32_sentinel (rows ordinal : ℕ)
    (hpad : blockCount rows 128 * 128 ≤ 2147483520)
    (hord : ordinal < blockCount rows 128 * 128) :
    ordinal < 2147483647 := by
  omega

/-- The same guard bounds the number of valid gallery rows. -/
theorem gallery_rows_lt_i32_sentinel (rows : ℕ)
    (hpad : blockCount rows 128 * 128 ≤ 2147483520) :
    rows < 2147483647 := by
  have hcover := rows_le_blockCount_mul rows 128 (by omega)
  omega

/-- Every block emits at most `k` candidates, including the short tail block. -/
theorem candidate_count_le (blocks k : ℕ) (emitted : Fin blocks → ℕ)
    (h : ∀ b, emitted b ≤ k) :
    (∑ b : Fin blocks, emitted b) ≤ blocks * k := by
  calc
    (∑ b : Fin blocks, emitted b) ≤ ∑ _b : Fin blocks, k :=
      Finset.sum_le_sum (fun b _ => h b)
    _ = blocks * k := by simp

theorem blocked_candidate_count_le (rows width k : ℕ)
    (emitted : Fin (blockCount rows width) → ℕ)
    (_hw : 0 < width) (h : ∀ b, emitted b ≤ k) :
    (∑ b : Fin (blockCount rows width), emitted b) ≤
      blockCount rows width * k :=
  candidate_count_le (blockCount rows width) k emitted h

/-- Storage for fixed-size emitted candidate records. -/
theorem candidate_bytes_le (rows width k bytesPerCandidate : ℕ)
    (emitted : Fin (blockCount rows width) → ℕ)
    (hw : 0 < width) (h : ∀ b, emitted b ≤ k) :
    (∑ b : Fin (blockCount rows width), emitted b) * bytesPerCandidate ≤
      blockCount rows width * k * bytesPerCandidate := by
  exact Nat.mul_le_mul_right bytesPerCandidate
    (blocked_candidate_count_le rows width k emitted hw h)

/-- Physical per-block candidate lanes can exceed the logical retained `k`. -/
def allocatedCandidateSlots (rows width tileWidth : ℕ) : ℕ :=
  blockCount rows width * tileWidth

/-- Payload bytes for one pair of candidate tensors across a query batch. -/
def allocatedCandidateBytes (rows width tileWidth bytesPerLane batch : ℕ) : ℕ :=
  allocatedCandidateSlots rows width tileWidth * bytesPerLane * batch

theorem logical_candidates_le_allocated_slots (rows width k tileWidth : ℕ)
    (emitted : Fin (blockCount rows width) → ℕ)
    (hw : 0 < width) (h : ∀ b, emitted b ≤ k) (hk : k ≤ tileWidth) :
    (∑ b : Fin (blockCount rows width), emitted b) ≤
      allocatedCandidateSlots rows width tileWidth := by
  exact (blocked_candidate_count_le rows width k emitted hw h).trans
    (Nat.mul_le_mul_left (blockCount rows width) hk)

/-- The exact score-product count for a dense row-by-dimension scan. -/
def denseScoreProducts (rows dimensions : ℕ) : ℕ := rows * dimensions

def paddedScoreProducts (rows width dimensions : ℕ) : ℕ :=
  blockCount rows width * width * dimensions

theorem denseScoreProducts_le_padded (rows width dimensions : ℕ)
    (hw : 0 < width) :
    denseScoreProducts rows dimensions ≤
      paddedScoreProducts rows width dimensions := by
  exact Nat.mul_le_mul_right dimensions (rows_le_blockCount_mul rows width hw)

example : blockCount 1000003 128 = 7813 := by decide
example : blockCount 1000003 128 * 10 = 78130 := by decide
example : allocatedCandidateSlots 1000003 128 16 = 125008 := by decide
example : allocatedCandidateBytes 1000003 128 16 8 1 = 1000064 := by decide
example : allocatedCandidateBytes 1000003 128 16 8 32 = 32002048 := by decide

/-- End-to-end latency is bounded by the sum of explicit stage bounds. -/
theorem latency_le_of_stage_bounds
    (score select merge transfer host
      scoreBound selectBound mergeBound transferBound hostBound : ℕ)
    (hs : score ≤ scoreBound) (hsel : select ≤ selectBound)
    (hm : merge ≤ mergeBound) (ht : transfer ≤ transferBound)
    (hh : host ≤ hostBound) :
    score + select + merge + transfer + host ≤
      scoreBound + selectBound + mergeBound + transferBound + hostBound := by
  omega

/-- A symbolic end-to-end envelope. Each stage premise must be established
for the actual implementation and hardware before this bounds wall time. -/
theorem symbolic_latency_le
    (rows dimensions width k scoreUnit selectUnit mergeBound
      transferUnit hostBound score select merge transfer host : ℕ)
    (_hw : 0 < width)
    (hs : score ≤ scoreUnit * blockCount rows width * width * dimensions)
    (hsel : select ≤ selectUnit * blockCount rows width * width * k)
    (hm : merge ≤ mergeBound)
    (ht : transfer ≤ transferUnit * blockCount rows width * k)
    (hh : host ≤ hostBound) :
    score + select + merge + transfer + host ≤
      scoreUnit * blockCount rows width * width * dimensions +
      selectUnit * blockCount rows width * width * k +
      mergeBound +
      transferUnit * blockCount rows width * k + hostBound := by
  exact latency_le_of_stage_bounds
    score select merge transfer host
    (scoreUnit * blockCount rows width * width * dimensions)
    (selectUnit * blockCount rows width * width * k)
    mergeBound
    (transferUnit * blockCount rows width * k)
    hostBound hs hsel hm ht hh

end SforaProofs
