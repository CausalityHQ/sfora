def _serving_exit_functions(binding):
    modules, guards, checker = _serving_helper_binding(binding)
    names = tuple(module.__name__.rsplit('.', 1)[-1] for module in modules)
    sources = tuple(read_serving_metadata(path, sha, require) for path, sha in guards)
    graph = {name: ModuleType(module.__name__) for name, module in zip(names, modules, strict=True)}
    builtins = __builtins__ if type(__builtins__) is dict else vars(__builtins__)
    original_import = builtins['__import__']

    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            require(level == 1 and name in graph, 'finite exit helper relative import required')
            return graph[name]
        return original_import(name, globals, locals, fromlist, level)

    guardian = type(checker.__self__)()
    for name, module, raw, (path, sha) in zip(names, modules, sources, guards, strict=True):
        namespace = vars(graph[name])
        namespace.update(__file__=path, __package__=module.__package__, __builtins__=MappingProxyType({**builtins, '__import__':imports}))
        exec(compile(raw, path, 'exec', dont_inherit=True), namespace)
        guardian._snapshot(graph[name], raw)
    namespaces, callables = tuple(guardian._namespaces), tuple(guardian._callables)
    bridge = sys.modules['sfora.connected_compact_serving']
    state_globals = bridge._literal_state.__globals__.copy()
    literal_state = FunctionType(bridge._literal_state.__code__, state_globals)
    state_globals['_literal_state'] = literal_state
    state_code = literal_state.__code__
    functions = graph[names[2]].admit_serving_artifact, graph[names[4]].verify_installed_environment

    def check():
        require(literal_state.__code__ is state_code and literal_state.__globals__ is state_globals and state_globals['_literal_state'] is literal_state, 'retained literal checker changed')
        for fn, code, defaults, kwdefaults, closure, default_state, kwdefault_state in callables:
            require(fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kwdefaults and fn.__closure__ is closure and literal_state(fn.__defaults__) == default_state and literal_state(fn.__kwdefaults__) == kwdefault_state, 'retained exit callable changed')
        for module, values, literals in namespaces:
            namespace = vars(module)
            require(namespace.keys() == values.keys() and all(namespace[key] is value for key, value in values.items()) and all(literal_state(namespace[key]) == state for key, state in literals.items()), 'retained exit namespace changed')
        return functions
    return check


def _serving_exit_callables(context):
    check = context['exit_check']
    expected = next(code for code in _serving_exit_functions.__code__.co_consts if isinstance(code, CodeType) and code.co_name == 'check')
    require(type(check) is FunctionType and check.__code__ is expected and check.__globals__ is globals() and check.__defaults__ is None and check.__kwdefaults__ is None, 'authenticated retained exit checker required')
    return check()
