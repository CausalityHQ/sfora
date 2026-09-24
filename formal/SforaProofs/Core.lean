import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Prod.Lex
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

namespace SforaProofs

/-! ## Layer 1: generic "k smallest" of a finset in a linear order. -/
section Generic
variable {α : Type*} [LinearOrder α]

/-- 0-based rank of `x` in `S`: number of elements of `S` strictly below `x`. -/
def rank (S : Finset α) (x : α) : ℕ := (S.filter (· < x)).card

/-- The `k` smallest elements of `S` (all of `S` when `#S ≤ k`). -/
def bottomK (S : Finset α) (k : ℕ) : Finset α := S.filter (fun x => rank S x < k)

theorem mem_bottomK {S : Finset α} {k : ℕ} {x : α} :
    x ∈ bottomK S k ↔ x ∈ S ∧ rank S x < k := by
  simp [bottomK]

theorem rank_mono {T S : Finset α} (h : T ⊆ S) (x : α) : rank T x ≤ rank S x :=
  Finset.card_le_card (Finset.filter_subset_filter _ h)

/-- Global winners survive local selection (holds for any subset, any `k`). -/
theorem mem_bottomK_of_subset {T S : Finset α} {k : ℕ} {x : α} (h : T ⊆ S)
    (hx : x ∈ T) (hS : x ∈ bottomK S k) : x ∈ bottomK T k := by
  rw [mem_bottomK] at *
  exact ⟨hx, lt_of_le_of_lt (rank_mono h x) hS.2⟩

theorem rank_lt_rank {S : Finset α} {x y : α} (hx : x ∈ S) (hxy : x < y) :
    rank S x < rank S y := by
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset]
  · exact ⟨x, by simp [hx, hxy], by simp⟩
  · intro a ha
    rw [Finset.mem_filter] at ha ⊢
    exact ⟨ha.1, lt_trans ha.2 hxy⟩

theorem rank_lt_card {S : Finset α} {x : α} (hx : x ∈ S) : rank S x < S.card :=
  Finset.card_lt_card (Finset.filter_ssubset.2 ⟨x, hx, lt_irrefl x⟩)

/-- `bottomK` is downward closed: a winner is below every non-winner of `S`. -/
theorem lt_of_mem_bottomK_of_notMem {S : Finset α} {k : ℕ} {x y : α}
    (hy : y ∈ bottomK S k) (hxS : x ∈ S) (hx : x ∉ bottomK S k) : y < x := by
  rw [mem_bottomK] at hy hx
  rcases lt_trichotomy y x with h | h | h
  · exact h
  · subst h; exact absurd hy hx
  · exact absurd ⟨hxS, lt_trans (rank_lt_rank hxS h) hy.2⟩ hx

/-- Cardinality: exactly `min k #S` elements. Induction inserting a new maximum. -/
theorem card_bottomK (S : Finset α) (k : ℕ) : (bottomK S k).card = min k S.card := by
  induction S using Finset.induction_on_max with
  | empty => simp [bottomK]
  | insert a s ha ih =>
    have haS : a ∉ s := fun h => lt_irrefl a (ha a h)
    have hr : ∀ x ∈ s, rank (insert a s) x = rank s x := by
      intro x hx
      unfold rank
      rw [Finset.filter_insert, if_neg (not_lt.2 (ha x hx).le)]
    have hra : rank (insert a s) a = s.card := by
      unfold rank
      rw [Finset.filter_insert, if_neg (lt_irrefl a), Finset.filter_true_of_mem ha]
    have hfilt : s.filter (fun x => rank (insert a s) x < k) = bottomK s k :=
      Finset.filter_congr (fun x hx => by rw [hr x hx])
    unfold bottomK
    rw [Finset.filter_insert, hra, hfilt, Finset.card_insert_of_notMem haS]
    have hnot : a ∉ bottomK s k := fun h => haS (Finset.filter_subset _ _ h)
    split_ifs with hk
    · rw [Finset.card_insert_of_notMem hnot, ih]; omega
    · rw [ih]; omega

/-- Exact block merge. Blocks need only cover `S` (overlap allowed, short blocks allowed). -/
theorem bottomK_merge {κ : Type*} (I : Finset κ) (B : κ → Finset α) (S : Finset α) (k : ℕ)
    (hsub : ∀ i ∈ I, B i ⊆ S) (hcov : ∀ x ∈ S, ∃ i ∈ I, x ∈ B i) :
    bottomK (I.biUnion (fun i => bottomK (B i) k)) k = bottomK S k := by
  set W := I.biUnion (fun i => bottomK (B i) k) with hW
  have hWS : W ⊆ S := by
    intro x hx
    rw [hW, Finset.mem_biUnion] at hx
    obtain ⟨i, hi, hx⟩ := hx
    exact hsub i hi (Finset.filter_subset _ _ hx)
  have hTW : bottomK S k ⊆ W := by
    intro x hx
    obtain ⟨i, hi, hxi⟩ := hcov x (Finset.filter_subset _ _ hx)
    rw [hW, Finset.mem_biUnion]
    exact ⟨i, hi, mem_bottomK_of_subset (hsub i hi) hxi hx⟩
  apply Finset.Subset.antisymm
  · intro x hx
    by_contra hnot
    have hxS : x ∈ S := hWS (Finset.filter_subset _ _ hx)
    have hsubW : bottomK S k ⊆ W.filter (· < x) := fun y hy =>
      Finset.mem_filter.2 ⟨hTW hy, lt_of_mem_bottomK_of_notMem hy hxS hnot⟩
    have h1 : (bottomK S k).card ≤ rank W x := Finset.card_le_card hsubW
    have h2 := rank_lt_card hxS
    rw [card_bottomK] at h1
    rw [mem_bottomK] at hx hnot
    have h3 : k ≤ rank S x := le_of_not_gt (fun h => hnot ⟨hxS, h⟩)
    omega
  · intro x hx
    exact mem_bottomK_of_subset hWS (hTW hx) hx

end Generic

/-! ## Layer 2: score-descending / ordinal-ascending ranking of a gallery. -/
section Scored
variable {ι β : Type*} [LinearOrder ι] [LinearOrder β]

/-- Sort key: larger score first, then smaller ordinal. Injective via the ordinal. -/
def keyFun (f : ι → β) (x : ι) : Lex (βᵒᵈ × ι) := toLex (OrderDual.toDual (f x), x)

omit [LinearOrder ι] [LinearOrder β] in
theorem keyFun_injective (f : ι → β) : Function.Injective (keyFun f) := by
  intro x y h
  exact congrArg (fun p => (ofLex p).2) h

def key (f : ι → β) : ι ↪ Lex (βᵒᵈ × ι) := ⟨keyFun f, keyFun_injective f⟩

/-- `beats f x y`: `x` is ranked strictly before `y`. -/
def beats (f : ι → β) (x y : ι) : Prop := key f x < key f y

instance (f : ι → β) : DecidableRel (beats f) :=
  fun x y => inferInstanceAs (Decidable (key f x < key f y))

theorem beats_iff (f : ι → β) (x y : ι) :
    beats f x y ↔ f y < f x ∨ (f x = f y ∧ x < y) := by
  show keyFun f x < keyFun f y ↔ _
  unfold keyFun
  rw [Prod.Lex.toLex_lt_toLex]
  simp [OrderDual.toDual_lt_toDual]

theorem beats_irrefl (f : ι → β) (x : ι) : ¬ beats f x x := lt_irrefl _

/-- Top-`k` of `S` under `f`. -/
def topK (f : ι → β) (S : Finset ι) (k : ℕ) : Finset ι :=
  S.filter (fun x => key f x ∈ bottomK (S.map (key f)) k)

theorem map_topK (f : ι → β) (S : Finset ι) (k : ℕ) :
    (topK f S k).map (key f) = bottomK (S.map (key f)) k := by
  ext z
  simp only [topK, Finset.mem_map, Finset.mem_filter]
  constructor
  · rintro ⟨x, ⟨_, hx⟩, rfl⟩; exact hx
  · intro hz
    obtain ⟨x, hxS, rfl⟩ := Finset.mem_map.1 (Finset.filter_subset _ _ hz)
    exact ⟨x, ⟨hxS, hz⟩, rfl⟩

theorem mem_topK {f : ι → β} {S : Finset ι} {k : ℕ} {x : ι} :
    x ∈ topK f S k ↔ x ∈ S ∧ (S.filter (fun y => beats f y x)).card < k := by
  simp only [topK, mem_bottomK, rank, Finset.filter_map, Finset.card_map, Finset.mem_filter,
    Finset.mem_map', Function.comp_def, and_self_left]
  exact Iff.rfl

theorem map_biUnion' {γ κ : Type*} [DecidableEq γ] (e : ι ↪ γ) (I : Finset κ)
    (g : κ → Finset ι) : (I.biUnion g).map e = I.biUnion (fun i => (g i).map e) := by
  ext z
  simp only [Finset.mem_map, Finset.mem_biUnion]
  constructor
  · rintro ⟨x, ⟨i, hi, hx⟩, rfl⟩; exact ⟨i, hi, x, hx, rfl⟩
  · rintro ⟨i, hi, x, hx, rfl⟩; exact ⟨x, ⟨i, hi, hx⟩, rfl⟩

theorem topK_subset (f : ι → β) (S : Finset ι) (k : ℕ) : topK f S k ⊆ S :=
  Finset.filter_subset _ _

theorem card_topK (f : ι → β) (S : Finset ι) (k : ℕ) : (topK f S k).card = min k S.card := by
  rw [← Finset.card_map (key f), map_topK, card_bottomK, Finset.card_map]

theorem mem_topK_of_subset {f : ι → β} {T S : Finset ι} {k : ℕ} {x : ι} (h : T ⊆ S)
    (hx : x ∈ T) (hS : x ∈ topK f S k) : x ∈ topK f T k := by
  rw [mem_topK] at *
  exact ⟨hx, lt_of_le_of_lt (Finset.card_le_card (Finset.filter_subset_filter _ h)) hS.2⟩

theorem topK_merge {κ : Type*} (f : ι → β) (I : Finset κ) (B : κ → Finset ι) (S : Finset ι)
    (k : ℕ) (hsub : ∀ i ∈ I, B i ⊆ S) (hcov : ∀ x ∈ S, ∃ i ∈ I, x ∈ B i) :
    topK f (I.biUnion (fun i => topK f (B i) k)) k = topK f S k := by
  apply Finset.map_injective (key f)
  rw [map_topK, map_topK, map_biUnion']
  simp_rw [map_topK]
  apply bottomK_merge
  · intro i hi; exact Finset.map_subset_map.2 (hsub i hi)
  · intro z hz
    obtain ⟨x, hxS, rfl⟩ := Finset.mem_map.1 hz
    obtain ⟨i, hi, hxi⟩ := hcov x hxS
    exact ⟨i, hi, Finset.mem_map_of_mem _ hxi⟩

/-- Sort an already selected set by score and ordinal. -/
def orderSelected (f : ι → β) (T : Finset ι) : List ι :=
  (T.map (key f)).sort.map (fun z => (ofLex z).2)

/-- The keys projected back to ordinals by `orderSelected` are strictly sorted. -/
theorem selectedKeys_sorted (f : ι → β) (T : Finset ι) :
    ((T.map (key f)).sort).SortedLT :=
  Finset.sortedLT_sort _

theorem selectedKeys_nodup (f : ι → β) (T : Finset ι) :
    ((T.map (key f)).sort).Nodup :=
  Finset.sort_nodup _ _

private theorem key_roundtrip_of_mem (f : ι → β) (T : Finset ι)
    {z : Lex (βᵒᵈ × ι)} (hz : z ∈ (T.map (key f)).sort) :
    key f (ofLex z).2 = z := by
  obtain ⟨x, _, rfl⟩ := Finset.mem_map.mp ((Finset.mem_sort (· ≤ ·)).mp hz)
  rfl

/-- The returned ordinals are strictly ordered by score, then ordinal. -/
theorem orderSelected_pairwise (f : ι → β) (T : Finset ι) :
    (orderSelected f T).Pairwise (beats f) := by
  let l := (T.map (key f)).sort
  have hsorted : l.SortedLT := selectedKeys_sorted f T
  have h : List.Pairwise (fun z w => z ∈ l ∧ w ∈ l ∧ z < w) l :=
    List.Pairwise.and_mem.mp hsorted.pairwise
  unfold orderSelected
  exact List.Pairwise.map (S := beats f) (fun z => (ofLex z).2)
    (fun z w ⟨hz, hw, hlt⟩ => by
      change key f (ofLex z).2 < key f (ofLex w).2
      rw [key_roundtrip_of_mem f T hz, key_roundtrip_of_mem f T hw]
      exact hlt) h

theorem orderSelected_nodup (f : ι → β) (T : Finset ι) :
    (orderSelected f T).Nodup := by
  apply List.Pairwise.imp ?_ (orderSelected_pairwise f T)
  intro a b hab heq
  subst b
  exact beats_irrefl f a hab

/-- The selected ordinals in their deterministic score/ordinal order. -/
def orderedTopK (f : ι → β) (S : Finset ι) (k : ℕ) : List ι :=
  orderSelected f (topK f S k)

/-- Exact ordered output after block selection and merge. -/
theorem orderedTopK_merge {κ : Type*} (f : ι → β) (I : Finset κ)
    (B : κ → Finset ι) (S : Finset ι) (k : ℕ)
    (hsub : ∀ i ∈ I, B i ⊆ S)
    (hcov : ∀ x ∈ S, ∃ i ∈ I, x ∈ B i) :
    orderedTopK f (I.biUnion (fun i => topK f (B i) k)) k =
      orderedTopK f S k :=
  congrArg (orderSelected f) (topK_merge f I B S k hsub hcov)

/-- One merge level, with natural-number group indices. -/
def MergeStep (f : ι → β) (k : ℕ) (S W : Finset ι) : Prop :=
  ∃ (I : Finset ℕ) (B : ℕ → Finset ι),
    (∀ i ∈ I, B i ⊆ S) ∧
    (∀ x ∈ S, ∃ i ∈ I, x ∈ B i) ∧
    W = I.biUnion (fun i => topK f (B i) k)

theorem mergeStep_exact {f : ι → β} {k : ℕ} {S W : Finset ι}
    (h : MergeStep f k S W) : topK f W k = topK f S k := by
  obtain ⟨I, B, hsub, hcov, rfl⟩ := h
  exact topK_merge f I B S k hsub hcov

/-- Repeated exact merge levels preserve the same global top-k. -/
theorem iterated_merge_exact {f : ι → β} {k : ℕ} {S W : Finset ι}
    (h : Relation.ReflTransGen (MergeStep f k) S W) :
    topK f W k = topK f S k := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => exact (mergeStep_exact hstep).trans ih

theorem iterated_ordered_merge_exact {f : ι → β} {k : ℕ}
    {S W : Finset ι} (h : Relation.ReflTransGen (MergeStep f k) S W) :
    orderedTopK f W k = orderedTopK f S k :=
  congrArg (orderSelected f) (iterated_merge_exact h)

theorem card_candidates_le {κ : Type*} (f : ι → β) (I : Finset κ) (B : κ → Finset ι) (k : ℕ) :
    (I.biUnion (fun i => topK f (B i) k)).card ≤ I.card * k := by
  calc (I.biUnion (fun i => topK f (B i) k)).card
      ≤ ∑ i ∈ I, (topK f (B i) k).card := Finset.card_biUnion_le
    _ ≤ ∑ _i ∈ I, k := Finset.sum_le_sum (fun i _ => by rw [card_topK]; exact min_le_left _ _)
    _ = I.card * k := by simp

end Scored

/-! ## Layer 3: conditional robust recall. -/
section Recall
variable {ι β : Type*} [LinearOrder ι] [AddCommGroup β] [LinearOrder β] [IsOrderedAddMonoid β]

/-- Pointwise score error at most `ε` on `S` (stated without `|·|`). -/
def ErrWithin (f g : ι → β) (ε : β) (S : Finset ι) : Prop :=
  0 ≤ ε ∧ ∀ y ∈ S, f y - ε ≤ g y ∧ g y ≤ f y + ε

/-- True winners whose exact score beats every true outsider by more than `2ε`. -/
def robust (f : ι → β) (ε : β) (S : Finset ι) (k : ℕ) : Finset ι :=
  (topK f S k).filter (fun x => ∀ y ∈ S, y ∉ topK f S k → f y + ε + ε < f x)

theorem robust_mem_topK_approx {f g : ι → β} {ε : β} {S : Finset ι} {k : ℕ} {x : ι}
    (hε : ErrWithin f g ε S) (hx : x ∈ robust f ε S k) : x ∈ topK g S k := by
  rw [robust, Finset.mem_filter] at hx
  obtain ⟨hxT, hmargin⟩ := hx
  have hxS : x ∈ S := topK_subset f S k hxT
  rw [mem_topK]
  refine ⟨hxS, ?_⟩
  have hsub : S.filter (fun y => beats g y x) ⊆ (topK f S k).erase x := by
    intro y hy
    rw [Finset.mem_filter] at hy
    obtain ⟨hyS, hyx⟩ := hy
    rw [Finset.mem_erase]
    refine ⟨fun h => beats_irrefl g x (h ▸ hyx), ?_⟩
    by_contra hyT
    have hgxy : g x ≤ g y := by
      rw [beats_iff] at hyx
      rcases hyx with h | ⟨h, _⟩
      · exact h.le
      · exact h.ge
    have h1 : f y + ε < f x - ε := lt_sub_iff_add_lt.2 (hmargin y hyS hyT)
    have h2 := (hε.2 y hyS).2
    have h3 := (hε.2 x hxS).1
    exact absurd (lt_of_le_of_lt h2 (lt_of_lt_of_le h1 h3)) (not_lt.2 hgxy)
  calc (S.filter (fun y => beats g y x)).card
      ≤ ((topK f S k).erase x).card := Finset.card_le_card hsub
    _ < (topK f S k).card := Finset.card_erase_lt_of_mem hxT
    _ ≤ k := by rw [card_topK]; exact min_le_left _ _

/-- If an ideal positive scores more than `2ε` above every negative, every
    observed top-1 result is positive. Other positives may outrank the witness.
    This is a label-recall guarantee, rather than exact-neighbor stability. -/
theorem top1_label_correct_of_positive_margin {ι : Type*} [LinearOrder ι]
    (S : Finset ι) (ideal observed : ι → ℝ) (positive : ι → Prop)
    (ε : ℝ) (hε : ErrWithin ideal observed ε S)
    (x : ι) (hx : x ∈ S) (hxPositive : positive x)
    (hmargin : ∀ y ∈ S, ¬ positive y → ideal y + ε + ε < ideal x) :
    ∀ z ∈ topK observed S 1, positive z := by
  intro z hz
  by_cases hzx : z = x
  · simpa [hzx] using hxPositive
  by_contra hzNegative
  have hzS : z ∈ S := topK_subset observed S 1 hz
  have hobs : observed z < observed x := by
    have hxLower := (hε.2 x hx).1
    have hzUpper := (hε.2 z hzS).2
    linarith [hmargin z hzS hzNegative]
  have hbeats : beats observed x z := (beats_iff observed x z).2 (Or.inl hobs)
  have hfilter : x ∈ S.filter (fun y => beats observed y z) :=
    Finset.mem_filter.mpr ⟨hx, hbeats⟩
  have hcard : 0 < (S.filter (fun y => beats observed y z)).card :=
    Finset.card_pos.mpr ⟨x, hfilter⟩
  have htop := (mem_topK.mp hz).2
  omega

noncomputable section
open Classical
/-- On a finite query panel, count queries with a positive witness whose ideal
    score clears every negative by `2ε`. Pointwise score-error certificates
    make this a lower bound on observed top-1 label hits. The premise must be
    checked for the deployed scorer and the actual query/gallery rows. -/
theorem certified_queries_le_top1_hits {Q ι : Type*} [DecidableEq Q] [LinearOrder ι]
    (queries : Finset Q) (gallery : Q → Finset ι)
    (ideal observed : Q → ι → ℝ) (positive : Q → ι → Prop) (ε : Q → ℝ)
    (herror : ∀ q ∈ queries, ErrWithin (ideal q) (observed q) (ε q) (gallery q)) :
    (queries.filter (fun q => ∃ x ∈ gallery q, positive q x ∧
      ∀ y ∈ gallery q, ¬ positive q y → ideal q y + ε q + ε q < ideal q x)).card ≤
    (queries.filter (fun q => ∃ z ∈ topK (observed q) (gallery q) 1,
      positive q z)).card := by
  classical
  apply Finset.card_le_card
  intro q hq
  obtain ⟨hqSet, x, hx, hxPositive, hmargin⟩ := Finset.mem_filter.mp hq
  have hcorrect := top1_label_correct_of_positive_margin
    (gallery q) (ideal q) (observed q) (positive q) (ε q)
    (herror q hqSet) x hx hxPositive hmargin
  have hcard : 0 < (topK (observed q) (gallery q) 1).card := by
    rw [card_topK]
    exact lt_min (by omega) (Finset.card_pos.mpr ⟨x, hx⟩)
  obtain ⟨z, hz⟩ := Finset.card_pos.mp hcard
  exact Finset.mem_filter.mpr ⟨hqSet, z, hz, hcorrect z hz⟩

/-- The certified-query fraction is a conditional lower bound on label
    Recall@1 over this exact, nonempty query panel. -/
theorem certified_fraction_le_top1_recall {Q ι : Type*} [DecidableEq Q] [LinearOrder ι]
    (queries : Finset Q) (gallery : Q → Finset ι)
    (ideal observed : Q → ι → ℝ) (positive : Q → ι → Prop) (ε : Q → ℝ)
    (hqueries : queries.Nonempty)
    (herror : ∀ q ∈ queries, ErrWithin (ideal q) (observed q) (ε q) (gallery q)) :
    ((queries.filter (fun q => ∃ x ∈ gallery q, positive q x ∧
      ∀ y ∈ gallery q, ¬ positive q y → ideal q y + ε q + ε q < ideal q x)).card : ℚ)
        / queries.card ≤
    ((queries.filter (fun q => ∃ z ∈ topK (observed q) (gallery q) 1,
      positive q z)).card : ℚ) / queries.card := by
  have hcount := certified_queries_le_top1_hits
    queries gallery ideal observed positive ε herror
  apply div_le_div_of_nonneg_right
  · exact_mod_cast hcount
  · exact_mod_cast (Finset.card_pos.mpr hqueries).le
end

/-- Recall (hits) is at least the number of robust winners. -/
theorem card_robust_le_hits {f g : ι → β} {ε : β} {S : Finset ι} {k : ℕ}
    (hε : ErrWithin f g ε S) :
    (robust f ε S k).card ≤ (topK f S k ∩ topK g S k).card :=
  Finset.card_le_card (fun _x hx =>
    Finset.mem_inter.2 ⟨Finset.filter_subset _ _ hx, robust_mem_topK_approx hε hx⟩)

/-- The same bound as a recall fraction, using the actual number of returned
true neighbors as the denominator. A positive denominator is required. -/
theorem robust_fraction_le_recall {f g : ι → β} {ε : β} {S : Finset ι} {k : ℕ}
    (hε : ErrWithin f g ε S) (hpositive : 0 < (topK f S k).card) :
    ((robust f ε S k).card : ℚ) / (topK f S k).card ≤
      ((topK f S k ∩ topK g S k).card : ℚ) / (topK f S k).card := by
  have hcount : (robust f ε S k).card ≤
      (topK f S k ∩ topK g S k).card := card_robust_le_hits hε
  apply div_le_div_of_nonneg_right
  · exact_mod_cast hcount
  · exact_mod_cast hpositive.le

theorem robust_fraction_le_recall_of_nonempty {f g : ι → β}
    {ε : β} {S : Finset ι} {k : ℕ}
    (hε : ErrWithin f g ε S) (hk : 0 < k) (hS : S.Nonempty) :
    ((robust f ε S k).card : ℚ) / (topK f S k).card ≤
      ((topK f S k ∩ topK g S k).card : ℚ) / (topK f S k).card := by
  have hpositive : 0 < (topK f S k).card := by
    rw [card_topK]
    exact lt_min hk (Finset.card_pos.mpr hS)
  exact robust_fraction_le_recall hε hpositive

/-- Full margin: approximate top-`k` equals exact top-`k` (recall one). -/
theorem topK_approx_eq_of_full_margin {f g : ι → β} {ε : β} {S : Finset ι} {k : ℕ}
    (hε : ErrWithin f g ε S)
    (hall : ∀ x ∈ topK f S k, ∀ y ∈ S, y ∉ topK f S k → f y + ε + ε < f x) :
    topK g S k = topK f S k := by
  symm
  apply Finset.eq_of_subset_of_card_le
  · intro x hx
    exact robust_mem_topK_approx hε (Finset.mem_filter.2 ⟨hx, hall x hx⟩)
  · rw [card_topK, card_topK]

end Recall

/-! ## Boundary examples checked by kernel reduction. -/
section Examples

def sc (l : List ℤ) : ℕ → ℤ := fun i => l.getD i 0

-- equal scores at ordinals 1 and 2: lower ordinal wins
example : topK (sc [5, 7, 7, 1]) {0, 1, 2, 3} 1 = {1} := by decide
example : topK (sc [5, 7, 7, 1]) {0, 1, 2, 3} 2 = {1, 2} := by decide
-- padded tail: ordinal 4 has a high score in the map but is absent from the gallery
example : topK (sc [5, 7, 7, 1, 9]) {0, 1, 2, 3} 2 = {1, 2} := by decide
-- short block {3} contributes all of itself; merge still exact
example : topK (sc [5, 7, 7, 1]) (({0, 1, 2} : Finset ℕ).biUnion
    (fun i => topK (sc [5, 7, 7, 1]) (([{0, 1}, {2}, {3}] : List (Finset ℕ)).getD i ∅) 2)) 2
    = {1, 2} := by
  decide
-- k larger than the gallery: everything is returned
example : (topK (sc [5, 7, 7, 1]) {0, 1, 2, 3} 9).card = 4 := by decide
-- non-robust margin: ε = 1, gap 1 ≤ 2ε, approximate top-1 differs; robust set is empty
example : topK (sc [3, 2]) {0, 1} 1 = {0} := by decide
example : topK (sc [2, 3]) {0, 1} 1 = {1} := by decide
example : robust (sc [3, 2]) 1 {0, 1} 1 = ∅ := by decide
example : robust (sc [3, 0]) 1 {0, 1} 1 = {0} := by decide
-- At the exact 2ε boundary, score error can create a tie and flip the winner.
example : ErrWithin (sc [0, 2]) (sc [1, 1]) 1 {0, 1} := by
  constructor
  · decide
  intro y hy
  simp only [Finset.mem_insert, Finset.mem_singleton] at hy
  rcases hy with rfl | rfl
  · constructor <;> decide
  · constructor <;> decide
example : topK (sc [0, 2]) {0, 1} 1 = {1} := by decide
example : topK (sc [1, 1]) {0, 1} 1 = {0} := by decide
example : robust (sc [0, 2]) 1 {0, 1} 1 = ∅ := by decide
-- The shared ordinal is deduplicated by the finite-set merge.
example : topK (sc [5, 7, 6]) (({0, 1} : Finset ℕ).biUnion
    (fun i => topK (sc [5, 7, 6])
      (([{0, 1}, {1, 2}] : List (Finset ℕ)).getD i ∅) 2)) 2 = {1, 2} := by
  decide

end Examples

end SforaProofs
