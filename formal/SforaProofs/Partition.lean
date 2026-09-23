import SforaProofs.Core
import SforaProofs.Cost

namespace SforaProofs

/-- Valid gallery ordinals assigned to a quotient-indexed block. -/
def blockRows (rows width block : ℕ) : Finset ℕ :=
  (Finset.range rows).filter (fun i => i / width = block)

theorem blockRows_subset (rows width block : ℕ) :
    blockRows rows width block ⊆ Finset.range rows :=
  Finset.filter_subset _ _

/-- A positive width puts every valid ordinal in one indexed block. -/
theorem blockRows_cover (rows width : ℕ) (hw : 0 < width)
    {i : ℕ} (hi : i ∈ Finset.range rows) :
    ∃ block ∈ Finset.range (blockCount rows width),
      i ∈ blockRows rows width block := by
  refine ⟨i / width, ?_, ?_⟩
  · rw [Finset.mem_range]
    apply (Nat.div_lt_iff_lt_mul hw).2
    exact lt_of_lt_of_le (Finset.mem_range.mp hi)
      (rows_le_blockCount_mul rows width hw)
  · exact Finset.mem_filter.mpr ⟨hi, rfl⟩

/-- Concrete fixed-width partition instance of the abstract top-k merge. -/
theorem topK_merge_blocks (score : ℕ → β) [LinearOrder β]
    (rows width k : ℕ) (hw : 0 < width) :
    topK score
      ((Finset.range (blockCount rows width)).biUnion
        (fun block => topK score (blockRows rows width block) k)) k =
      topK score (Finset.range rows) k := by
  apply topK_merge score _ _ _ k
  · intro block _
    exact blockRows_subset rows width block
  · intro i hi
    exact blockRows_cover rows width hw hi

theorem orderedTopK_merge_blocks (score : ℕ → β) [LinearOrder β]
    (rows width k : ℕ) (hw : 0 < width) :
    orderedTopK score
      ((Finset.range (blockCount rows width)).biUnion
        (fun block => topK score (blockRows rows width block) k)) k =
      orderedTopK score (Finset.range rows) k := by
  apply orderedTopK_merge score _ _ _ k
  · intro block _
    exact blockRows_subset rows width block
  · intro i hi
    exact blockRows_cover rows width hw hi

/-- The logical candidates emitted by this fixed-width partition. -/
theorem block_candidates_le (score : ℕ → β) [LinearOrder β]
    (rows width k : ℕ) (_hw : 0 < width) :
    ((Finset.range (blockCount rows width)).biUnion
      (fun block => topK score (blockRows rows width block) k)).card ≤
      blockCount rows width * k := by
  simpa using card_candidates_le score
    (Finset.range (blockCount rows width))
    (blockRows rows width) k

end SforaProofs
