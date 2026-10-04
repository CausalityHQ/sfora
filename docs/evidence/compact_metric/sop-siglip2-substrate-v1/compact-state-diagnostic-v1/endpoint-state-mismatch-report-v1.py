def endpoint_state_mismatch_report(facts, model_facts, state):
    # Root-only diagnostic: use the already computed complete typed hashes.
    # Read no checkpoint, image, cache, teacher, quality or new tensor bytes.
    values = {
        'vision': lambda: state['model'].state_dict(),
        'config': lambda: state['model'].config.to_dict(),
        'buffers': lambda: dict(state['model'].named_buffers()),
        'head': lambda: dict(state['head_object'].state_dict()),
        'A': lambda: state['A'],
        'means': lambda: state['means'],
    }
    mismatches = {}
    for field in values:
        expected = facts['vision_sha256'] if field == 'vision' else facts['members'][field]
        actual = model_facts['vision_sha256'] if field == 'vision' else model_facts['members'][field]
        if actual != expected:
            value = values[field]()
            info = {'expected_typed_sha256': expected, 'actual_typed_sha256': actual,
                    'actual_python_type': type(value).__module__ + '.' + type(value).__qualname__}
            if field == 'A':
                info.update(dtype=str(value.dtype), shape=list(value.shape), device=str(value.device))
            mismatches[field] = info
    return {'event': 'COMPACT_ENDPOINT_STATE_MISMATCH', 'fields': mismatches,
            'tensor_device_and_subclass_are_not_hash_identity': True}
