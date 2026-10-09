"""Candidate payload predicate and expected metadata; not an integrated loader."""


def _serving_validate_payload(disk, manifest):
    require(
        disk.keys() == INFERENCE_KEYS
        and disk["schema"] == INFERENCE_SCHEMA
        and disk["arm"] in ARMS
        and fingerprint(disk) == manifest["endpoint_state_sha256"]
        and fingerprint({k: v for k, v in disk.items() if k != "fixed_sha256"})
        == disk["fixed_sha256"]
        and numerical_flags() == disk["numerical_flags"]
        and disk["vision_sha256"] == manifest["vision_sha256"]
        and disk["encoder_identity"] == manifest["encoder_identity"]
        and disk["base_vision"]["sha256"] == manifest["base_vision_sha256"]
        and disk["scope"]["arm"] == "control"
        and disk["scope"]["payload"]["scope_sha256"] == CONTROL_SHA256
        and len(disk["scope"]["payload"]["class_names"]) == 1008,
        "complete original-scope/updated inference identity differs",
    )


def _serving_expected_identities(prepared, disk):
    manifest = prepared['origin']
    _serving_validate_payload(disk, manifest)
    identity = prepared['helpers'][3]
    original_env = manifest['environment']
    original_roots = {name: row['root'] for name, row in original_env['packages'].items()}
    installed_roots = {name: row['root'] for name, row in prepared['installed']['expected_environment']['packages'].items()}
    result = {}
    for kind, original in (('model', disk['encoder_identity']), ('processor', disk['processor'])):
        projected = identity.project_identity(original, kind=kind, package_roots=original_roots, source_files=original_env['files'])
        result[kind] = identity.materialize_identity(projected, original, kind=kind, package_roots=original_roots, source_files=original_env['files'], installed_roots=installed_roots)
    _serving_helper_binding((prepared['helpers'], prepared['helper_guards'], prepared['checker']))
    return result
