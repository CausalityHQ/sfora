# Low-rank learning research boundary

The new question is whether a fixed low-rank encoder-adaptation method can
improve TRAIN-held quality at bounded cost. It is not approximation of the
previously trained dense delta. No low-rank method, training run, rank sweep
or production loader has been selected yet. Fable consultation
`05f2cad4e3ad478f` is the sole active research job; its
[start receipt](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/start.json)
records the evidence, question and1,800s/$6 caps.

The previous rank32 SVD captured40.6174% of the first inspected Large dense
matrix change and stopped its approximation arm. Its
[decision](inshop_native_delta_rank32_result_2026-09-28.md) explicitly leaves
learning a different low-rank solution untested. The
[saved-state attribution](inshop_pe_learning_attribution_result_2026-09-28.md)
localizes the PE/Large reversal to encoder adaptation under that recipe;
it does not establish PE overfitting, an LR defect or a capacity ceiling.

An authenticated CPU load of the saved PE checkpoint measured24 upper-block
matrix shapes: per block, fused QKV2304×768, output768×768, MLP3072×768 and
768×3072, across blocks6–11. These contain42,467,328 dense elements; fixed
rank32 factors would contain2,359,296 elements, an18× reduction for those
matrices only. Independent stdlib arithmetic verified every shape/product
and total. This is a conditional parameter-count calculation, not measured
training speed, peak VRAM, inference speed or retrieval quality. Dense source
weights remain resident; pooling/projection, norms, biases, head and proxies
are outside this calculation. No factors were constructed or fitted.
Original CPU cost was0.96s and1,024,652KiB maximum RSS, exit0, no CUDA or
model forward. [Inventory and authority](evidence/compact_metric/sop-siglip2-substrate-v1/lowrank-learning-research-v1/inventory.json).

The review must resolve these boundaries before a candidate is frozen:

- Factorization changes optimizer coordinates, initialization/scaling,
  gradient clipping and weight decay. Equal nominal LR/decay coefficients
  do not isolate rank or preserve the original update. In particular,
  decaying both factors multiplies their product by the square of the
  per-factor decay, while freezing the original matrix removes its dense
  AdamW decay. Any experiment tests a declared complete adaptation method.
- Fixed-rank count and raw spectrum cannot forecast retrieval improvement.
  A CPU mechanics check can qualify inventory, zero-update behavior and
  merging, rather than masquerade as a causal quality diagnostic.
- Equal image exposure and equal compute are separate controls. Keep both
  descriptions explicit; fewer optimizer states do not prove lower wall
  time. No extension of the stopped PE100 recipe is authorized by this note.
- Merged dense serving preserves architecture shape, but separate low-rank
  matmuls need not be bitwise identical to a merged multiply. Final exported
  vectors and packed/native serving output must qualify the actual chosen
  merged checkpoint; no inference parity claim from real-arithmetic algebra.

[LoRA's original paper](https://arxiv.org/abs/2106.09685) supplies known
method motivation, not a new PE/In-Shop measurement or guarantee. Collect and
independently assess the single consultation before selecting one bounded
TRAIN-only candidate or a STOP. Consequential design requires Opus/Astra
critique. The complete SOP/In-Shop quality-and-speed goal remains active;
no official quality or serving result changes here.
