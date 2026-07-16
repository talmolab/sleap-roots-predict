## REMOVED Requirements

### Requirement: Scan Metadata Resolution To Params

**Reason**: The `resolve_params` implementation is promoted into `sleap-roots-contracts`
(contracts PR #16, released `0.1.0a4`) as the single source of truth across producers;
predict's local copy is deleted in favor of consuming it.

**Migration**: Callers continue `from sleap_roots_predict import resolve_params` (predict
re-exports it from `sleap_roots_contracts`, unchanged public surface) or may import
`from sleap_roots_contracts import resolve_params` directly.

### Requirement: Species Name Normalization

**Reason**: Implementation detail of `resolve_params`, now owned by
`sleap-roots-contracts`.

**Migration**: No caller-visible change; `sleap_roots_contracts.params._normalize_species`
is the new home for anyone reaching into the internals.

### Requirement: Imaging Mode Resolution Seam

**Reason**: Implementation detail of `resolve_params`, now owned by
`sleap-roots-contracts`.

**Migration**: No caller-visible change; `sleap_roots_contracts.params._mode_for_scan` is
the new home for anyone reaching into the internals.

### Requirement: Age Resolution In Days

**Reason**: Implementation detail of `resolve_params`, now owned by
`sleap-roots-contracts`. Contracts additionally hardens this coercion against pandas/numpy
sentinels (`np.bool_`, `Decimal`, non-finite floats) that predict's copy silently
mis-coerced.

**Migration**: No caller-visible change for well-formed inputs; malformed inputs that used
to silently coerce now raise `ValueError` (documented in `CHANGELOG.md`).

### Requirement: Override Merge And Strict Post-Override Validation

**Reason**: Implementation detail of `resolve_params`, now owned by
`sleap-roots-contracts`.

**Migration**: No caller-visible change; override semantics are identical.

### Requirement: Public API Export

**Reason**: predict no longer implements `resolve_params`, so this is no longer a
`param-resolution` capability requirement.

**Migration**: predict continues to re-export `resolve_params` from
`sleap_roots_contracts` for backward compatibility (`from sleap_roots_predict import
resolve_params` still works); this is now covered by the general public-API test
(`tests/test_public_api.py`), not a dedicated requirement in this capability.

### Requirement: Interoperability With Model Selection

**Reason**: This capability (`param-resolution`) is removed wholesale; the round-trip
contract this requirement described belongs to the consumer side, not the resolver.

**Migration**: Migrated into the `model-management` capability's `Model Selection From Scan
Params` requirement (see that capability's spec delta in this same change), so the
metadata → params → model contract stays documented where `choose_models` lives.
