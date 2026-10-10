    exit_check = prepared['exit_check']
    afn, efn = runtime._serving_exit_callables(prepared)
    old_a, old_e = helpers[2].admit_serving_artifact, helpers[4].verify_installed_environment
    executed = []
    helpers[2].admit_serving_artifact = lambda *a, **k: executed.append('forged artifact')
    helpers[4].verify_installed_environment = lambda *a, **k: executed.append('forged environment')
    saved_payload = (directory/'endpoint.pt').read_bytes()
    wheel_path = environment.target/'torch/__init__.py'
    saved_wheel = wheel_path.read_bytes()
    (directory/'endpoint.pt').write_bytes(saved_payload+b'changed')
    wheel_path.write_bytes(saved_wheel+b'changed')
    try:
        try: runtime._serving_full_exit(prepared)
        except ValueError as failure:
            notes = getattr(failure, '__notes__', ())
            assert any('artifact' in note for note in notes), notes
            assert any('installed environment' in note for note in notes), notes
        else: raise AssertionError('helper and payload/wheel mutation accepted')
        assert not executed, executed
        saved_code = afn.__code__;afn.__code__ = (lambda *a, **k: None).__code__
        try:
            try: runtime._serving_exit_callables(prepared)
            except ValueError: pass
            else: raise AssertionError('retained graph code mutation accepted')
        finally: afn.__code__ = saved_code
    finally:
        helpers[2].admit_serving_artifact, helpers[4].verify_installed_environment = old_a, old_e
        (directory/'endpoint.pt').write_bytes(saved_payload)
        wheel_path.write_bytes(saved_wheel)
    runtime._serving_full_exit(prepared)
    print('PASS genuine integrated retained graph: changed live helpers never execute; fresh artifact/wheel mutations reject independently; restored exit passes')
    old_package = helpers[4].__package__
    helpers[4].__package__ = '_foreign'
    try:
        try: runtime._serving_helper_binding(binding)
        except ValueError as error: assert 'private helper package' in str(error)
        else: raise AssertionError('foreign helper package accepted')
    finally: helpers[4].__package__ = old_package
    imported_require = helpers[2]._require
    helpers[2]._require = lambda *args: None
    try:
        try: runtime._serving_helper_binding(binding)
        except ValueError as error: assert 'relative dependency binding' in str(error)
        else: raise AssertionError('foreign imported helper callable accepted')
    finally: helpers[2]._require = imported_require
