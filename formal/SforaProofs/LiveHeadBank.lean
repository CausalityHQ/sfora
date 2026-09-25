import SforaProofs.MemoryBank
import Mathlib.Tactic.Ring

namespace SforaProofs

/-- The trainable affine projection applied to an already normalized source
    feature. Output normalization is intentionally outside this definition. -/
def affineHead {m n : ℕ} (weight : Fin m → Fin n → ℝ)
    (bias : Fin m → ℝ) (source : Fin n → ℝ) : Fin m → ℝ :=
  fun output => dotScore (weight output) source + bias output

/-- Reprojecting a cached source row through the current head removes head
    staleness exactly. Any remaining discrepancy must come from source drift. -/
theorem affineHead_equal_of_source_equal {m n : ℕ}
    (weight : Fin m → Fin n → ℝ) (bias : Fin m → ℝ)
    (cached fresh : Fin n → ℝ) (h : cached = fresh) :
    affineHead weight bias cached = affineHead weight bias fresh := by
  rw [h]

/-- For a fixed current head, source drift bounds each projected coordinate.
    The affine bias cancels, so only the current weight row contributes. -/
theorem affineHead_coordinate_drift {m n : ℕ}
    (weight : Fin m → Fin n → ℝ) (bias : Fin m → ℝ)
    (stale fresh : Fin n → ℝ) (output : Fin m) :
    |affineHead weight bias fresh output - affineHead weight bias stale output| ≤
      ∑ input, |weight output input| * |fresh input - stale input| := by
  have hcancel :
      affineHead weight bias fresh output - affineHead weight bias stale output =
        dotScore (weight output) fresh - dotScore (weight output) stale := by
    dsimp [affineHead]
    ring
  rw [hcancel]
  exact dotScore_stale_error (weight output) stale fresh

/-- The exact real-valued query score after the live affine head has a source
    drift radius weighted by both the query and the current head. This does
    not bound encoder drift, final L2 normalization, or CUDA roundoff. -/
theorem affineHead_dotScore_drift {m n : ℕ}
    (query : Fin m → ℝ) (weight : Fin m → Fin n → ℝ)
    (bias : Fin m → ℝ) (stale fresh : Fin n → ℝ) :
    |dotScore query (affineHead weight bias fresh) -
      dotScore query (affineHead weight bias stale)| ≤
      ∑ output, |query output| *
        (∑ input, |weight output input| * |fresh input - stale input|) := by
  exact dotScore_stale_error_of_coordinate_bounds
    query (affineHead weight bias stale) (affineHead weight bias fresh)
    (fun output =>
      ∑ input, |weight output input| * |fresh input - stale input|)
    (affineHead_coordinate_drift weight bias stale fresh)

end SforaProofs
