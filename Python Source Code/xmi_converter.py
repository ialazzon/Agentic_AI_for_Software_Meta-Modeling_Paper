"""
xmi_converter.py
================
Serialize an XMI .model file to a concise YAML "conceptual form",
and deserialize a (possibly modified) YAML back to a valid XMI .model file.

Usage
-----
    # Serialize
    python xmi_converter.py serialize --ecore path/to/meta.ecore \
                                      --model  path/to/instance.model \
                                      --out    path/to/output.yaml

    # Deserialize (YAML → XMI)
    python xmi_converter.py deserialize --ecore path/to/meta.ecore \
                                        --yaml  path/to/modified.yaml \
                                        --out   path/to/rebuilt.model

Programmatic API
----------------
    from xmi_converter import serialize_model, deserialize_model

    yaml_text = serialize_model("meta.ecore", "instance.model")
    deserialize_model("meta.ecore", yaml_text, "rebuilt.model")

    # Or work with Python dicts directly:
    from xmi_converter import model_to_dict, dict_to_model, save_model
    data = model_to_dict("meta.ecore", "instance.model")
    # … edit data …
    rset, root = dict_to_model("meta.ecore", data)
    save_model(rset, root, "rebuilt.model")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pyecore.ecore import EReference, EAttribute, EObject, EOrderedSet, EList
from pyecore.resources import ResourceSet, URI
from pyecore.resources.xmi import XMIResource

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_metamodel(rset: ResourceSet, ecore_path: str) -> None:
    """Register the .ecore metamodel into *rset* so XMI proxies resolve."""
    mm_uri = URI(str(Path(ecore_path).resolve()))
    mm_resource = rset.get_resource(mm_uri)
    mm_root = mm_resource.contents[0]          # EPackage
    rset.metamodel_registry[mm_root.nsURI] = mm_root


def _load_model(rset: ResourceSet, model_path: str) -> EObject:
    """Load the XMI .model file and return its root EObject."""
    m_uri = URI(str(Path(model_path).resolve()))
    resource = rset.get_resource(m_uri)
    return resource.contents[0]


def _eobj_to_dict(obj: EObject, visited: set | None = None) -> dict[str, Any]:
    """
    Recursively convert an EObject to a plain Python dict.

    Keys
    ----
    _type       : simple EClass name (e.g. "Package", "Class")
    _id         : xmi:id when available (omitted otherwise)
    <attr_name> : primitive / enum attribute value (omitted when unset)
    <ref_name>  : for containment references – nested dict or list of dicts
                  for non-containment references – fragment URI string(s)
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        # Return a back-reference placeholder
        return {"_ref": _fragment(obj)}
    visited = visited | {obj_id}

    result: dict[str, Any] = {"_type": obj.eClass.name}

    # xmi:id if the resource assigned one
    try:
        frag = obj.eURIFragment()
        if frag and frag != "/":
            result["_id"] = frag
    except Exception:
        pass

    eclass = obj.eClass
    for feature in eclass.eAllStructuralFeatures():
        if feature.derived or feature.transient:
            continue

        name = feature.name
        value = obj.eGet(feature)

        # Skip unset / empty / None
        if value is None:
            continue
        if isinstance(value, (EOrderedSet, EList)) and len(value) == 0:
            continue

        if isinstance(feature, EAttribute):
            # Primitive / enum attribute
            if isinstance(value, (EOrderedSet, EList)):
                serialised = [_attr_value(v) for v in value]
            else:
                serialised = _attr_value(value)
            result[name] = serialised

        elif isinstance(feature, EReference):
            if feature.containment:
                # Owned children → recurse
                if isinstance(value, (EOrderedSet, EList)):
                    result[name] = [_eobj_to_dict(v, visited) for v in value]
                else:
                    result[name] = _eobj_to_dict(value, visited)
            else:
                # Cross-reference → store URI fragment(s) as strings
                if isinstance(value, (EOrderedSet, EList)):
                    result[name] = [_fragment(v) for v in value]
                else:
                    result[name] = _fragment(value)

    return result


def _attr_value(v: Any) -> Any:
    """Convert an attribute value to a YAML-friendly scalar."""
    # EEnumLiteral → its name string; everything else → its native Python type
    if hasattr(v, "name"):          # EEnumLiteral
        return v.name
    return v


def _fragment(obj: EObject) -> str:
    """Return the xmi:id / URI fragment for a cross-reference."""
    try:
        return obj.eURIFragment()
    except Exception:
        return repr(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Deserialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_eclass(mm_root, class_name: str):
    """Search the metamodel package (and sub-packages) for an EClass by name."""
    for cls in mm_root.eAllContents():
        if cls.eClass.name == "EClass" and cls.name == class_name:
            return cls
    raise KeyError(f"EClass '{class_name}' not found in metamodel.")


def _dict_to_eobj(
    data: dict[str, Any],
    mm_root,
    id_map: dict[str, EObject],
    deferred_refs: list[tuple[EObject, str, Any]],
) -> EObject:
    """
    Recursively reconstruct an EObject from a dict produced by _eobj_to_dict.

    Cross-references are recorded in *deferred_refs* and resolved in a second
    pass after the whole tree is built (so forward references work).
    """
    type_name = data.get("_type")
    if not type_name:
        raise ValueError(f"Dict is missing '_type' key: {data}")

    eclass = _find_eclass(mm_root, type_name)
    obj = eclass().__class__()          # allocate via dynamic instance

    # Use the dynamic/meta approach that works with both static and dynamic models
    obj = eclass.eType() if hasattr(eclass, "eType") else _new_instance(eclass)

    # Register by _id for later cross-reference resolution
    node_id = data.get("_id")
    if node_id:
        id_map[node_id] = obj

    for feature in eclass.eAllStructuralFeatures():
        if feature.derived or feature.transient:
            continue
        name = feature.name
        if name not in data:
            continue
        value = data[name]

        if isinstance(feature, EAttribute):
            _set_attribute(obj, feature, value)

        elif isinstance(feature, EReference):
            if feature.containment:
                if feature.many:
                    items = value if isinstance(value, list) else [value]
                    # An inline object is a dict carrying "_type"; a dict with
                    # "_ref" (or a bare string) is a cross-reference.
                    dict_items = [
                        v for v in items
                        if isinstance(v, dict) and "_ref" not in v
                    ]
                    ref_items = [
                        v for v in items
                        if not (isinstance(v, dict) and "_ref" not in v)
                    ]
                    children = [
                        _dict_to_eobj(v, mm_root, id_map, deferred_refs)
                        for v in dict_items
                    ]
                    getattr(obj, name).extend(children)
                    # Cross-reference entries are resolved in the second pass.
                    if ref_items:
                        deferred_refs.append((obj, name, ref_items))
                else:
                    v = value[0] if isinstance(value, list) else value
                    if isinstance(v, dict) and "_ref" not in v:
                        child = _dict_to_eobj(v, mm_root, id_map, deferred_refs)
                        setattr(obj, name, child)
                    else:
                        deferred_refs.append((obj, name, v))
            else:
                # Non-containment reference. Values are normally
                # cross-references (resolved in the second pass), but an inline
                # object (dict with "_type") can legally appear on the opposite
                # side of a bidirectional reference whose containment lives on
                # the other end. Build such objects here so they get registered
                # in id_map and can be resolved by the containing feature.
                items = value if isinstance(value, list) else [value]
                built = []
                pending = []
                for v in items:
                    if isinstance(v, dict) and "_ref" not in v and "_type" in v:
                        built.append(
                            _dict_to_eobj(v, mm_root, id_map, deferred_refs)
                        )
                    else:
                        pending.append(v)
                if built:
                    if feature.many:
                        getattr(obj, name).extend(built)
                    else:
                        setattr(obj, name, built[0])
                if pending:
                    deferred_refs.append(
                        (obj, name, pending if feature.many else pending[0])
                    )

    return obj


def _new_instance(eclass) -> EObject:
    """Create a new dynamic EObject instance for the given EClass."""
    # pyecore dynamic instances
    from pyecore.ecore import EObject as _EObj
    obj = eclass.python_class()
    return obj


def _set_attribute(obj: EObject, feature, value: Any) -> None:
    """Set an EAttribute, handling many-valued and enum cases."""
    etype = feature.eType

    def _coerce(v):
        # If the attribute type is an EEnum, look up the literal by name.
        # Guard on the classifier actually being an EEnum: hasattr(...,
        # "getEEnumLiteral") is also true for non-enum datatypes, and calling
        # getEEnumLiteral with a non-string value (e.g. an int ID) raises
        # "argument of type 'int' is not iterable".
        if etype is not None and getattr(etype.eClass, "name", None) == "EEnum":
            lit = etype.getEEnumLiteral(name=str(v))
            return lit if lit else v
        # EDate → datetime
        if getattr(etype, "name", None) == "EDate" and isinstance(v, str):
            from datetime import datetime
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            return v
        # Otherwise cast to the Python type declared on the EDataType
        if hasattr(etype, "eType") and etype.eType is not None:
            try:
                return etype.eType(v)
            except Exception:
                return v
        return v

    if isinstance(value, list):
        col = getattr(obj, feature.name)
        col.extend([_coerce(v) for v in value])
    else:
        setattr(obj, feature.name, _coerce(value))


def _resolve_deferred(
    deferred_refs: list[tuple[EObject, str, Any]],
    id_map: dict[str, EObject],
) -> None:
    """Second-pass: resolve recorded cross-reference strings → EObjects."""

    def _key(v):
        # A deferred entry may be a cross-reference string (e.g. "_3"), a
        # reference dict {"_ref": "_3"}, or an inline dict carrying its own
        # "_id". Reduce all forms to the id_map lookup key.
        if isinstance(v, dict):
            return v.get("_ref") or v.get("_id")
        return v

    for obj, name, value in deferred_refs:
        if isinstance(value, list):
            keys = [_key(v) for v in value]
            resolved = [id_map[k] for k in keys if k is not None and k in id_map]
            if resolved:
                col = getattr(obj, name)
                col.extend(resolved)
        else:
            key = _key(value)
            if key is not None and key in id_map:
                setattr(obj, name, id_map[key])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def model_to_dict(ecore_path: str, model_path: str) -> dict[str, Any]:
    """
    Load *ecore_path* + *model_path* and return the model as a plain dict.
    """
    rset = ResourceSet()
    _load_metamodel(rset, ecore_path)
    root = _load_model(rset, model_path)
    return _eobj_to_dict(root)


def serialize_model(ecore_path: str, model_path: str) -> str:
    """
    Return a YAML string representing the model in concise conceptual form.
    """
    data = model_to_dict(ecore_path, model_path)
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)


def dict_to_model(ecore_path: str, data: dict[str, Any]):
    """
    Reconstruct an (rset, root_EObject) from a dict produced by model_to_dict.
    Returns the ResourceSet and the root EObject so the caller can save it.
    """
    rset = ResourceSet()
    _load_metamodel(rset, ecore_path)

    mm_resource = rset.resources[list(rset.resources.keys())[0]]
    mm_root = mm_resource.contents[0]

    id_map: dict[str, EObject] = {}
    deferred_refs: list[tuple[EObject, str, Any]] = []

    root = _dict_to_eobj(data, mm_root, id_map, deferred_refs)
    _resolve_deferred(deferred_refs, id_map)

    return rset, root


def save_model(rset: ResourceSet, root: EObject, output_path: str) -> None:
    """Save *root* to *output_path* as an XMI .model file."""
    out_uri = URI(str(Path(output_path).resolve()))
    resource = rset.create_resource(out_uri)
    resource.append(root)

    # If any EClass in the model declares an iD attribute (iD=true) whose value
    # is non-string (e.g. ELong/EInt), pyecore's _build_path_from runs
    # `' ' not in id_att_value` against that numeric value and raises
    # "argument of type 'int' is not iterable". Switching the resource to
    # xmi:id (uuid) mode makes pyecore serialize objects and cross-references
    # via generated ids and skip the numeric-ID branch entirely, while the ID
    # attribute keeps its correct declared (numeric) type.
    if _has_nonstring_id_attribute(root):
        resource.use_uuid = True

    resource.save()


def _has_nonstring_id_attribute(root: EObject) -> bool:
    """True if any object in the tree has an iD=true attribute typed non-string."""
    def _check(o: EObject) -> bool:
        for feat in o.eClass.eAllStructuralFeatures():
            if getattr(feat, "iD", False):
                etype = getattr(feat, "eType", None)
                tname = getattr(etype, "name", None)
                if tname != "EString":
                    return True
        return False

    if _check(root):
        return True
    for child in root.eAllContents():
        if _check(child):
            return True
    return False


def deserialize_model(ecore_path: str, yaml_text: str, output_path: str) -> None:
    """
    Parse *yaml_text* (produced by serialize_model / hand-edited) and write
    a valid XMI .model file to *output_path*.
    """
    # print(yaml_text)
    data = yaml.safe_load(yaml_text)
    rset, root = dict_to_model(ecore_path, data)
    save_model(rset, root, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli_serialize(args: argparse.Namespace) -> None:
    yaml_text = serialize_model(args.ecore, args.model)
    if args.out:
        Path(args.out).write_text(yaml_text, encoding="utf-8")
        print(f"[serialize] Written to {args.out}")
    else:
        sys.stdout.write(yaml_text)


def _cli_deserialize(args: argparse.Namespace) -> None:
    yaml_text = Path(args.yaml).read_text(encoding="utf-8")
    deserialize_model(args.ecore, yaml_text, args.out)
    print(f"[deserialize] Written to {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert XMI .model ↔ concise YAML using pyecore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serialize
    ser = sub.add_parser("serialize", help="XMI .model → YAML")
    ser.add_argument("--ecore", required=True, metavar="FILE",
                     help="Path to the .ecore metamodel file")
    ser.add_argument("--model", required=True, metavar="FILE",
                     help="Path to the XMI .model instance file")
    ser.add_argument("--out",   default=None,  metavar="FILE",
                     help="Output .yaml file (default: print to stdout)")
    ser.set_defaults(func=_cli_serialize)

    # deserialize
    deser = sub.add_parser("deserialize", help="YAML → XMI .model")
    deser.add_argument("--ecore", required=True, metavar="FILE",
                       help="Path to the .ecore metamodel file")
    deser.add_argument("--yaml",  required=True, metavar="FILE",
                       help="Path to the YAML conceptual-form file")
    deser.add_argument("--out",   required=True, metavar="FILE",
                       help="Output .model file")
    deser.set_defaults(func=_cli_deserialize)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
