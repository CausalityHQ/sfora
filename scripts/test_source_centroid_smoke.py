from run_inshop_source_centroid_smoke import probe_failures


def test_smoke_requires_both_weak_probes_but_stops_either_dominant_probe():
    def probe(ratio, saturated=0):
        return {
            "weighted_auxiliary_to_main_gradient_ratio": ratio,
            "saturated_target_fraction": saturated,
        }

    assert probe_failures([probe(0.005), probe(0.005)]) == ["weak_auxiliary_encoder_gradient"]
    assert probe_failures([probe(0.005), probe(0.02)]) == []
    assert probe_failures([probe(0.02), probe(0.26)]) == ["dominant_auxiliary_encoder_gradient"]
    assert probe_failures([probe(0.02, 0.6), probe(0.02, 0.6)]) == ["saturated_auxiliary_targets"]
