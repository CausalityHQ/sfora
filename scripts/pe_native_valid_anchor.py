"""Opt-in native F5 boundary; preserve the original batch rank denominator."""
import pe_large_optimization as driver

original_terms = driver.coverage.terms


def valid_rank(raw, bank, head, positives, ordinals):
    # Same mask/mean as train_inshop_siglip2_unseen_gallery.valid_anchor_rank_loss.
    valid = (positives >= 0).any(dim=1)
    if not bool(valid.any()):
        return raw.sum() * 0  # All-invalid microbatch retains a backward graph.
    return driver.pair.smoke.member_bank_rank_loss(
        raw[valid], bank, head, positives[valid], ordinals[valid], live_head=False
    ) * (valid.sum() / len(valid))


def terms(source, head, classifier, bank, target, positive, index, rank_active):
    ce, rank, raw = original_terms(source, head, classifier, bank, target, positive, index, rank_active)
    if not rank_active:
        rank = valid_rank(raw, bank, head, positive[index], index)
    return ce, rank, raw
