"""RS@k on one existing CONTROL score row; no embeddings, sampling or SiMix.

Patel--Tolias--Matas, arXiv:2108.11179; yashvarpatel/RecallatK_surrogate
src/losses.py, commit ed052029d258555df2f94dd82d6f7df60ef7cc6f.
rank(p)=1+sum(sigmoid((s[j]-s[p])/.01), j in eligible, j!=p);
loss(k)=1-min(sum(sigmoid(k-rank(p)), p in positive),k)/min(k,|positive|).
Rank temperature .01, membership temperature 1, source exponent clip +/-50.
The caller supplies the CONTROL eligible columns, including its self exclusion.
"""

K_VALUES = (1, 2, 4, 8, 16)


def official_sigmoid(value, temperature=1.):
    # sigmoid() loses positive-tail FP32 gradients that the official expression retains.
    return 1.0 / (1.0 + (-value / temperature).clamp(min=-50., max=50.).exp())


def rsk_terms(scores, positive, eligible, *, k_values=K_VALUES):
    """Return one loss per k; their mean replaces one anchor's SmoothAP mean.

    Only a positives-by-eligible matrix is built. Other positives remain rank
    competitors. Empty positives return a graph-connected zero, without a divisor.
    """
    import torch

    if (len(scores.shape) != 1 or
            any(type(j) is not int or not 0 <= j < scores.shape[0] for j in (*positive, *eligible)) or
            len(set(positive)) != len(positive) or len(set(eligible)) != len(eligible) or
            not set(positive) <= set(eligible)):
        raise ValueError('unique in-range positive subset of eligible columns required')
    if not k_values or any(type(k) is not int or k <= 0 for k in k_values) or len(set(k_values)) != len(k_values):
        raise ValueError('distinct positive integer k values required')
    if not torch.isfinite(scores[eligible]).all().item():
        raise ValueError('finite eligible scores required')
    if not positive:
        return torch.stack([scores[:0].sum()])
    comparisons = official_sigmoid(scores[eligible][None, :] - scores[positive][:, None], .01)
    columns = torch.tensor(eligible, device=scores.device)
    positives = torch.tensor(positive, device=scores.device)
    ranks = 1 + (comparisons * (columns[None, :] != positives[:, None])).sum(1)
    terms = []
    for k in k_values:
        retrieved = official_sigmoid(k - ranks).sum()
        capped = torch.minimum(retrieved, torch.full_like(retrieved, float(k)))
        terms.append(1 - capped / min(k, len(positive)))
    return torch.stack(terms)
