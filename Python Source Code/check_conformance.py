#!/usr/bin/env python3
"""
XMI Model Conformance Checker
Usage: python check_conformance.py <metamodel.ecore> <model.xmi>

Validates that an XMI model instance conforms to its Ecore metamodel using
PyEcore only. Checks performed:
  - Model loads without structural errors
  - Every object's EClass belongs to the metamodel
  - Required (lowerBound >= 1) single-valued features are set
  - Required many-valued features satisfy minimum multiplicity
  - Maximum multiplicity (upperBound) is not exceeded
  - EAttribute values match their declared EDataType / EEnum literals
  - EReference targets are of the correct EClass (or subtype)
  - Containment integrity (an object must not be contained twice)
"""

import sys
import argparse
from collections import defaultdict

from pyecore.resources import ResourceSet, URI
from pyecore.ecore import (
    EClass, EObject, EAttribute, EReference,
    EEnum, EDataType, EcoreUtils,
)
from langchain_core.tools import tool
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

class Issue:
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"

    def __init__(self, severity, obj_repr, feature, message):
        self.severity  = severity
        self.obj_repr  = obj_repr
        self.feature   = feature
        self.message   = message

    def __str__(self):
        loc = f"{self.obj_repr}"
        if self.feature:
            loc += f".{self.feature}"
        return f"[{self.severity}] {loc}: {self.message}"


def _repr(obj):
    """Human-readable object identity."""
    try:
        name = getattr(obj, "name", None)
        if name:
            return f"{obj.eClass.name}(name={name!r})"
        frag = obj.eURIFragment()
        return f"{obj.eClass.name}(@{frag})"
    except Exception:
        return repr(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_metamodel(rset: ResourceSet, ecore_path: str):
    """Load an .ecore file and register its package(s) in the ResourceSet."""
    mm_resource = rset.get_resource(URI(ecore_path))
    mm_root = mm_resource.contents[0]          # EPackage
    rset.metamodel_registry[mm_root.nsURI] = mm_root
    return mm_root


def load_model(rset: ResourceSet, xmi_path: str):
    """Load the XMI model after the metamodel is already registered."""
    resource = rset.get_resource(URI(xmi_path))
    return resource


# ─────────────────────────────────────────────────────────────────────────────
# Conformance checks
# ─────────────────────────────────────────────────────────────────────────────

def _all_eclasses_in_package(pkg, result=None):
    """Recursively collect all EClass objects from an EPackage."""
    if result is None:
        result = set()
    for classifier in pkg.eClassifiers:
        if isinstance(classifier, EClass):
            result.add(classifier)
    for sub_pkg in pkg.eSubpackages:
        _all_eclasses_in_package(sub_pkg, result)
    return result


def _is_subtype(eclass, target_eclass):
    """Return True if eclass is target_eclass or a subtype of it."""
    if eclass is target_eclass:
        return True
    for super_type in eclass.eAllSuperTypes():
        if super_type is target_eclass:
            return True
    return False


def check_eclass_belongs_to_metamodel(obj, mm_eclasses, issues):
    ec = obj.eClass
    if ec not in mm_eclasses:
        issues.append(Issue(
            Issue.ERROR, _repr(obj), None,
            f"EClass '{ec.name}' is not defined in the metamodel."
        ))
        return False
    return True


def check_multiplicity(obj, issues):
    """Check lower/upper bounds for every structural feature."""
    for sf in obj.eClass.eAllStructuralFeatures():
        try:
            value = obj.eGet(sf)
        except Exception as exc:
            issues.append(Issue(Issue.WARNING, _repr(obj), sf.name,
                                f"Could not read feature: {exc}"))
            continue

        lower = sf.lowerBound  # int
        upper = sf.upperBound  # int; -1 means unlimited

        if sf.many:
            count = len(value) if value is not None else 0
            if count < lower:
                issues.append(Issue(
                    Issue.ERROR, _repr(obj), sf.name,
                    f"Multiplicity violation: has {count} value(s), "
                    f"minimum required is {lower}."
                ))
            if upper != -1 and count > upper:
                issues.append(Issue(
                    Issue.ERROR, _repr(obj), sf.name,
                    f"Multiplicity violation: has {count} value(s), "
                    f"maximum allowed is {upper}."
                ))
        else:
            is_set = value is not None
            if not is_set and lower >= 1:
                issues.append(Issue(
                    Issue.ERROR, _repr(obj), sf.name,
                    f"Required feature is not set (lowerBound={lower})."
                ))


def check_attribute_types(obj, issues):
    """Verify EAttribute values against their EDataType / EEnum."""
    for attr in obj.eClass.eAllStructuralFeatures():
        if not isinstance(attr, EAttribute):
            continue
        etype = attr.eType
        if etype is None:
            continue

        try:
            value = obj.eGet(attr)
        except Exception:
            continue

        values = list(value) if attr.many else ([value] if value is not None else [])

        for v in values:
            if isinstance(etype, EEnum):
                # Value must be one of the enum's literals
                literal_names = {lit.name for lit in etype.eLiterals}
                literal_values = {lit.value for lit in etype.eLiterals}
                v_check = v.name if hasattr(v, "name") else v
                if v_check not in literal_names and v not in literal_values:
                    issues.append(Issue(
                        Issue.ERROR, _repr(obj), attr.name,
                        f"Value {v!r} is not a valid literal of enum "
                        f"'{etype.name}' (literals: {sorted(literal_names)})."
                    ))
            else:
                # Basic Python type check for well-known EDataTypes
                type_name = etype.name if hasattr(etype, "name") else ""
                expected = {
                    "EString":  str,
                    "EInt":     int,
                    "EInteger": int,
                    "ELong":    int,
                    "EShort":   int,
                    "EByte":    int,
                    "EFloat":   float,
                    "EDouble":  float,
                    "EBoolean": bool,
                }.get(type_name)
                if expected and not isinstance(v, expected):
                    issues.append(Issue(
                        Issue.WARNING, _repr(obj), attr.name,
                        f"Value {v!r} has Python type '{type(v).__name__}', "
                        f"expected '{expected.__name__}' for EDataType '{type_name}'."
                    ))


def check_reference_types(obj, issues):
    """Verify EReference targets are instances of the declared EClass."""
    for ref in obj.eClass.eAllStructuralFeatures():
        if not isinstance(ref, EReference):
            continue
        etype = ref.eType
        if etype is None:
            continue

        try:
            value = obj.eGet(ref)
        except Exception:
            continue

        targets = list(value) if ref.many else ([value] if value is not None else [])

        for target in targets:
            if not isinstance(target, EObject):
                continue
            if not _is_subtype(target.eClass, etype):
                issues.append(Issue(
                    Issue.ERROR, _repr(obj), ref.name,
                    f"Referenced object is of EClass '{target.eClass.name}', "
                    f"but reference expects '{etype.name}' (or subtype)."
                ))


def check_containment_integrity(all_objects, issues):
    """Each EObject should appear in at most one containment reference."""
    seen = {}
    for obj in all_objects:
        for ref in obj.eClass.eAllStructuralFeatures():
            if not isinstance(ref, EReference) or not ref.containment:
                continue
            try:
                value = obj.eGet(ref)
            except Exception:
                continue
            children = list(value) if ref.many else ([value] if value is not None else [])
            for child in children:
                if not isinstance(child, EObject):
                    continue
                oid = id(child)
                if oid in seen:
                    prev_owner, prev_ref = seen[oid]
                    issues.append(Issue(
                        Issue.ERROR, _repr(child), None,
                        f"Object is contained in two places: "
                        f"{_repr(prev_owner)}.{prev_ref} AND "
                        f"{_repr(obj)}.{ref.name}."
                    ))
                else:
                    seen[oid] = (obj, ref.name)


# ─────────────────────────────────────────────────────────────────────────────
# Main conformance runner
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_objects(resource):
    """Iterate over every EObject reachable from the resource roots."""
    all_objs = []
    for root in resource.contents:
        all_objs.append(root)
        for child in root.eAllContents():
            all_objs.append(child)
    return all_objs


def check_conformance(ecore_path: str, xmi_path: str) -> list[Issue]:
    issues: list[Issue] = []
    rset = ResourceSet()

    # ── 1. Load metamodel ────────────────────────────────────────────────────
    try:
        mm_pkg = load_metamodel(rset, ecore_path)
        issues.append(Issue(Issue.INFO, ecore_path, None,
                            f"Metamodel loaded. nsURI='{mm_pkg.nsURI}', "
                            f"package='{mm_pkg.name}'."))
    except Exception as exc:
        issues.append(Issue(Issue.ERROR, ecore_path, None,
                            f"Failed to load metamodel: {exc}"))
        return issues

    mm_eclasses = _all_eclasses_in_package(mm_pkg)

    # ── 2. Load model instance ───────────────────────────────────────────────
    try:
        resource = load_model(rset, xmi_path)
        issues.append(Issue(Issue.INFO, xmi_path, None,
                            f"Model loaded. Root objects: {len(resource.contents)}."))
    except Exception as exc:
        issues.append(Issue(Issue.ERROR, xmi_path, None,
                            f"Failed to load model: {exc}"))
        return issues

    all_objects = collect_all_objects(resource)
    issues.append(Issue(Issue.INFO, xmi_path, None,
                        f"Total EObjects found: {len(all_objects)}."))

    # ── 3. Per-object checks ─────────────────────────────────────────────────
    for obj in all_objects:
        if not isinstance(obj, EObject):
            continue
        if not check_eclass_belongs_to_metamodel(obj, mm_eclasses, issues):
            continue   # skip further checks; EClass is unknown
        check_multiplicity(obj, issues)
        check_attribute_types(obj, issues)
        check_reference_types(obj, issues)

    # ── 4. Global checks ─────────────────────────────────────────────────────
    check_containment_integrity(all_objects, issues)

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check whether an XMI model conforms to an Ecore metamodel."
    )
    parser.add_argument("metamodel", help="Path to the .ecore metamodel file")
    parser.add_argument("model",     help="Path to the .xmi model instance file")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any WARNINGs are present (default: only ERRORs)."
    )
    args = parser.parse_args()

    issues = check_conformance(args.metamodel, args.model)

    # ── Print results ─────────────────────────────────────────────────────────
    counts = defaultdict(int)
    for issue in issues:
        print(issue)
        counts[issue.severity] += 1

    # print()
    # print("─" * 60)
    # print(f"Summary: {counts[Issue.ERROR]} error(s), "
    #       f"{counts[Issue.WARNING]} warning(s), "
    #       f"{counts[Issue.INFO]} info message(s).")

    has_errors   = counts[Issue.ERROR] > 0
    has_warnings = counts[Issue.WARNING] > 0

    if has_errors:
        print("Model does NOT conform to the metamodel.")
        sys.exit(1)
    elif args.strict and has_warnings:
        print("\n✗  Model has warnings (--strict mode enabled).")
        sys.exit(1)
    else:
        print("Model conforms to the metamodel.")
        sys.exit(0)


if __name__ == "__main__":
    main()
 

def check_conformance_model_to_metamodel(metamodel_path: str, input_model_path: str)->tuple[bool,str]:
    """Checks if the input model conforms with the source metamodel

    Args:
        metamodel_path (str): path to the source metamodel (ecore)
        input_model_path (str): path to the input model (.model file)

    Returns:
        bool: True if the input model conforms to the source metamodel, and False otherwise
    """
    # print("REAL TOOL INPUT")
    # print("input_model_path:", input_model_path)
    # print("metamodel_path:", metamodel_path)
    # print("---------")
    issues = check_conformance(metamodel_path, input_model_path)
    counts = defaultdict(int)
    errs = ""
    for issue in issues:
        # print(issue)
        counts[issue.severity] += 1
        if issue.severity == Issue.ERROR:
            errs = errs + str(issue) +"\n"

    has_errors   = counts[Issue.ERROR] > 0
    has_warnings = counts[Issue.WARNING] > 0
    if has_errors:
        return False,errs
    else:
        return True,""