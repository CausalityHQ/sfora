"""Owned failed-load cleanup candidate; no foreign-frame or registry removal."""

def _serving_tensor_refs(value, refs):
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        refs.append(weakref.ref(value))
    elif type(value) is dict:
        for item in value.values():
            _serving_tensor_refs(item, refs)
    elif type(value) in (tuple, list):
        for item in value:
            _serving_tensor_refs(item, refs)


def _serving_drop_partial(partial, guards):
    processor = partial.get('processor')
    if processor is not None:
        cache = _processor_cache(processor, guards)
        cache.cache_clear()
        require(cache.cache_info().currsize == 0, 'partial serving processor cache survived')
    partial.clear()


def _serving_failed_load(context, primary):
    frames, pending, seen = [], [primary], set()
    endpoint, partial, refs = None, {}, []
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend(value for value in (error.__cause__, error.__context__) if value is not None)
        trace = error.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            if frame.f_globals is globals():
                frames.append(frame)
                if frame.f_code is _serving_load_payload.__code__:
                    candidate = frame.f_locals.get('endpoint')
                    if type(candidate) is dict:
                        endpoint = candidate
                if frame.f_code in (_serving_load_payload.__code__, construct_encoder.__code__):
                    for name in ('disk', 'copied'):
                        _serving_tensor_refs(frame.f_locals.get(name), refs)
                    for name in ('model', 'processor', 'head'):
                        value = frame.f_locals.get(name)
                        if value is not None:
                            partial[name] = value
            trace = trace.tb_next
    for value in partial.values():
        try:
            refs.append(weakref.ref(value))
        except TypeError as error:
            primary.add_note('partial serving owner is not weak-referenceable: '+repr(error))
    value = candidate = frame = trace = error = None
    for owned_frame in frames:
        try:
            owned_frame.clear()
        except RuntimeError:
            pass
    owned_frame = None
    if endpoint is not None:
        partial.clear()
        try:
            _serving_release(endpoint)
        except BaseException as error:
            primary.add_note('failed serving endpoint release failed: '+repr(error))
    else:
        try:
            _serving_drop_partial(partial, context['guards'])
        except BaseException as error:
            primary.add_note('partial serving release failed: '+repr(error))
        finally:
            partial.clear()
    endpoint = None
    try:
        gc.collect()
        require(all(ref() is None for ref in refs), 'failed serving owner survived cleanup')
    except BaseException as error:
        primary.add_note('failed serving lifetime check failed: '+repr(error))
    for filename in ('endpoint.pt', 'vision.pt'):
        try:
            mapping_absent(Path(context['directory']) / filename)
        except BaseException as error:
            primary.add_note('failed serving mapping check '+filename+' failed: '+repr(error))
    try:
        _serving_full_exit(context)
    except BaseException as error:
        primary.add_note('failed serving full exit failed: '+repr(error))
    raise primary


def load_serving_inference(directory, *, trusted_serving_sha256, trusted_fragment_sha256, installed_environment, trusted_installed_environment_sha256, serving_helpers):
    prepared = _serving_prepare(directory, trusted_serving_sha256=trusted_serving_sha256, trusted_fragment_sha256=trusted_fragment_sha256, installed_environment=installed_environment, trusted_installed_environment_sha256=trusted_installed_environment_sha256, serving_helpers=serving_helpers)
    endpoint = None
    try:
        endpoint = _serving_load_payload(prepared)
        _serving_helper_binding(serving_helpers)
        return endpoint
    except BaseException as primary:
        # A post-load guard can fail after the payload helper has returned.
        if endpoint is not None:
            try:
                _serving_release(endpoint)
            except BaseException as error:
                primary.add_note('post-load serving release failed: '+repr(error))
            endpoint = None
        _serving_failed_load(prepared, primary)
