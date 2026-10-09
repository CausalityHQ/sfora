"""Serving lifecycle candidate; production/public bridge/native integration unrun."""

def _serving_full_exit(context, primary=None):
    checks = (
        ('runtime', lambda: _check_runtime()),
        ('helpers', lambda: _serving_helper_binding((context['helpers'], context['helper_guards'], context['checker']))),
        ('artifact', lambda: context['helpers'][2].admit_serving_artifact(context['directory'], trusted_serving_sha256=context['serving_sha256'], trusted_fragment_sha256=context['fragment_sha256'])),
        ('installed environment', lambda: _serving_exit_environment(context)),
        ('external anchors', lambda: batch_bound_files({}, context['anchors'].items())),
    )
    for label, check in checks:
        try:
            check()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note('serving exit '+label+' failed: '+repr(error))
    if primary is not None:
        raise primary


def _serving_exit_environment(context):
    raw = context['installed_raw']
    require(hashlib.sha256(raw).hexdigest() == context['installed_sha256'], 'installed authority changed before exit')
    expected = _serving_json(raw)
    actual = context['helpers'][4].verify_installed_environment(context['origin_raw'], context['owners_raw'], trusted_bundle_sha256=context['fragment_sha256']['origin.json'], trusted_ownership_audit_sha256=context['fragment_sha256']['origin-owners.json'], site_packages=expected['site_packages'])
    require(json.dumps(actual, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False), 'installed environment changed before exit')


def _serving_release(endpoint):
    context = endpoint.pop('_serving')
    inventory = tuple(context['owned_registry'])
    modules = tuple(endpoint['modules'].values())
    refs = [weakref.ref(endpoint[name]) for name in ('model', 'processor_object', 'head_object', 'A', 'C', 'mu_train')]
    refs += [weakref.ref(value) for value in (*endpoint['model'].parameters(), *endpoint['model'].buffers(), *endpoint['head_object'].parameters(), *endpoint['head_object'].buffers())]
    for name in ('means', 'common_statistics'):
        _serving_tensor_refs(endpoint.get(name), refs)
    primary = None
    try:
        _serving_helper_binding((context['helpers'], context['helper_guards'], context['checker']))
        require(all(sys.modules.get(name) is module for name, module in inventory), 'serving pre-release registry changed')
        require(all(any(module is owned for _, owned in inventory) for module in modules), 'serving release module is not owned')
    except BaseException as error:
        primary = error
    endpoint['modules'] = {}
    try:
        release_inference(endpoint)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note('serving original release failed: '+repr(error))
        # Only finished runtime frames; foreign frames and references remain intact.
        trace = error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_globals is globals():
                try:
                    trace.tb_frame.clear()
                except RuntimeError:
                    pass
            trace = trace.tb_next
    finally:
        endpoint.clear()
    try:
        gc.collect()
        require(all(ref() is None for ref in refs), 'serving lifetime survived artifact release')
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note('serving lifetime check failed: '+repr(error))
    try:
        _serving_full_exit(context)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note('serving full exit failed: '+repr(error))
    try:
        require(all(sys.modules.get(name) is module for name, module in inventory), 'serving post-release registry changed')
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            primary.add_note('serving registry postcheck failed: '+repr(error))
    if primary is not None:
        raise primary


def _serving_exact_json(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _serving_live_identity(endpoint):
    packages = endpoint['manifest']['environment']['packages']
    require(_serving_exact_json(model_structure(endpoint['model'], packages), endpoint['encoder_identity']['runtime']), 'serving typed installed model identity differs')
    processor = endpoint['processor_object']
    observed = {'config': json.loads(processor.to_json_string()), 'backend': processor.backend, 'origin': module_origin(type(processor), packages)}
    require(_serving_exact_json(observed, endpoint['processor']), 'serving typed installed processor identity differs')
