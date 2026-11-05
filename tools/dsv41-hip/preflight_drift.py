"""preflight_drift.py: drift between two preflight JSONs (preflight.sh --baseline / --diff).

Compares what a toolchain or OS update would silently change under the port: GPU stack, kernel and firmware package
versions, apt holds, the hash of the whole installed package set, kernel release, BIOS, toolchain versions, compiler
and runtime binary hashes, Zig, GPU ISA and the amdgpu/ttm parameters. Reads schema 1 (M0 r1) and schema 2 files; a
field recorded on one side only is listed apart (not comparable), never as equal or as drift.
"""


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return "<absent>"
        d = d[k]
    return d


def _packages(r):
    p = (r.get("packages") or {}).get("gpu_kernel_firmware")
    return p if p is not None else (r.get("toolchain") or {}).get("packages") or {}


def _out(v):
    return v.get("out") if isinstance(v, dict) and "out" in v else v


FIELDS = ["kernel.release", "distro.PRETTY_NAME", "machine.dmi.bios_version", "machine.dmi.bios_date", "gpu.isa",
          "gpu.amdgpu_params", "gpu.ttm_pages_limit_bytes", "toolchain.rocm_path", "toolchain.rocm_version_files",
          "toolchain.hipconfig_version", "toolchain.hipcc_version", "toolchain.amd_clang", "toolchain.python3",
          "toolchain.device_lib_bitcode_dirs", "toolchain.hip_include_dirs", "toolchain.binaries_sha256",
          "packages.apt_holds", "packages.installed_set_sha256"]


def drift(old, new):
    fields, one_sided = {}, {}
    for f in FIELDS:
        a, b = _out(_get(old, f)), _out(_get(new, f))
        if a != b:
            (one_sided if "<absent>" in (a, b) else fields)[f] = {"old": a, "new": b}
    za = sorted((z.get("realpath"), z.get("version"), z.get("sha256")) for z in (_get(old, "toolchain.zig_all") if isinstance(_get(old, "toolchain.zig_all"), list) else []))
    zb = sorted((z.get("realpath"), z.get("version"), z.get("sha256")) for z in (_get(new, "toolchain.zig_all") if isinstance(_get(new, "toolchain.zig_all"), list) else []))
    if za != zb:
        fields["toolchain.zig_all"] = {"old": za or _out(_get(old, "toolchain.zig")), "new": zb or _out(_get(new, "toolchain.zig"))}
    pa, pb = _packages(old), _packages(new)
    common = set(pa) & set(pb)
    pk = {"changed": {k: {"old": pa[k], "new": pb[k]} for k in sorted(common) if pa[k] != pb[k]},
          "added": {k: pb[k] for k in sorted(set(pb) - set(pa))}, "removed": {k: pa[k] for k in sorted(set(pa) - set(pb))}}
    schemas = (old.get("schema"), new.get("schema"))
    note = None
    if schemas[0] != schemas[1]:
        note = (f"schemas differ {schemas}: schema 1 recorded a narrower package set, so 'added' lists packages it did not "
                "look at; only 'changed' is a like-for-like comparison")
    return {"old": {"host": old.get("host"), "generated_utc": old.get("generated_utc"), "schema": schemas[0]},
            "new": {"host": new.get("host"), "generated_utc": new.get("generated_utc"), "schema": schemas[1]},
            "changed": bool(fields or pk["changed"] or (schemas[0] == schemas[1] and (pk["added"] or pk["removed"]))),
            "fields": fields, "packages": pk, "recorded_on_one_side_only": one_sided, "note": note}
