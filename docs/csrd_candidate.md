# Candidate 50: consensus-stable relational distillation (CSRD)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

Across five aligned CUB HERD training packs, different-class pair similarities
have mean cross-seed Pearson correlation 0.8127 and Spearman correlation 0.7936;
same-class relations reach 0.9098/0.9083. CSRD would train several teachers,
retain or upweight only pair relations on which they agree, and distil that
consensus geometry into one student. This treats replica agreement as a
measurement-reliability estimate rather than averaging all relations equally.

## Gate 2: prior art and cost

The operation is directly occupied. Multi-teacher agreement knowledge
distillation already treats agreement as knowledge reliability. Its relational
agreement component weights teacher distance and angular relations by agreement:

- *MTAKD: Multi-Teacher Agreement Knowledge Distillation for Edge AI Skin
  Disease Diagnosis*, 2025: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12722719/>

More generally, ensemble KD aggregates multiple teachers, and adaptive ensemble
KD explicitly handles teacher disagreement:

- Wu et al., *Unified and Effective Ensemble Knowledge Distillation*, 2022:
  <https://arxiv.org/abs/2204.00548>
- Du et al., *Agree to Disagree: Adaptive Ensemble Knowledge Distillation in
  Gradient Space*, NeurIPS 2020:
  <https://proceedings.neurips.cc/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html>

CSRD additionally violates the standing roughly-1x training-cost constraint by
requiring multiple complete teachers. Candidate 50 is **DEAD at Gate 2**.

