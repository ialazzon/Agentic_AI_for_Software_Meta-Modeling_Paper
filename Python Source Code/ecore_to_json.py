#!/usr/bin/env python3
"""
ecore_to_json.py
================
Reads an EMF/Ecore metamodel (.ecore) file and emits a focused JSON
representation containing only:

  - Entities  (EClass) with their attributes
  - Associations between entities (EReference)
  - Inheritance relationships (eSuperTypes)

Usage
-----
    python ecore_to_json.py <path-to-metamodel.ecore> [-o output.json] [-p]

Arguments
---------
    ecore_file          Path to the .ecore metamodel file (required)
    -o / --output       Write JSON to this file instead of stdout
    -p / --pretty       Pretty-print the JSON output

Output shape
------------
{
  "entities": [
    {
      "name": "Professor",
      "abstract": false,
      "attributes": [
        { "name": "rank", "type": "ProfessorRank", "multiplicity": "0..1" }
      ],
      "inheritance": {
        "extends": ["Person"]          // direct super-types
      },
      "associations": [
        {
          "name": "teaches",
          "target": "Course",
          "multiplicity": "0..*",
          "containment": false,
          "opposite": null            // or "Course.instructor"
        }
      ]
    }
  ],
  "enums": [
    {
      "name": "ProfessorRank",
      "literals": ["ASSISTANT", "ASSOCIATE", "FULL", "EMERITUS"]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from pyecore.ecore import EAttribute, EClass, EEnum, EPackage, EProxy, EReference
    from pyecore.resources import ResourceSet, URI
except ImportError:
    sys.exit("PyEcore is not installed.  Run:  pip install pyecore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multiplicity(lower: int, upper: int) -> str:
    lo, hi = str(lower), "*" if upper == -1 else str(upper)
    return lo if lo == hi else f"{lo}..{hi}"


def _type_name(feature) -> str:
    et = feature.eType
    if et is None:
        return "EJavaObject"
    if isinstance(et, EProxy):
        try:
            et.force_resolve()
        except Exception:
            pass
    try:
        return et.name or repr(et)
    except Exception:
        return repr(et)


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

def _serialise_package(pkg: EPackage) -> tuple[list[dict], list[dict]]:
    """Return (entities, enums) lists for *pkg* and all its sub-packages."""
    entities: list[dict] = []
    enums: list[dict] = []

    for classifier in pkg.eClassifiers:

        # ── EClass → entity ────────────────────────────────────────────────
        if isinstance(classifier, EClass):
            attributes = []
            associations = []

            for feat in classifier.eStructuralFeatures:

                if isinstance(feat, EAttribute):
                    attributes.append({
                        "name": feat.name,
                        "type": _type_name(feat),
                        "multiplicity": _multiplicity(feat.lowerBound, feat.upperBound),
                    })

                elif isinstance(feat, EReference):
                    opp = feat.eOpposite
                    if opp is not None:
                        opp_class = opp.eContainingClass
                        opp_str = (
                            f"{opp_class.name}.{opp.name}"
                            if opp_class else opp.name
                        )
                    else:
                        opp_str = None

                    associations.append({
                        "name": feat.name,
                        "target": _type_name(feat),
                        "multiplicity": _multiplicity(feat.lowerBound, feat.upperBound),
                        "containment": feat.containment,
                        "opposite": opp_str,
                    })

            entities.append({
                "name": classifier.name,
                "abstract": classifier.abstract,
                "attributes": attributes,
                "inheritance": {
                    "extends": [
                        (s.name if not isinstance(s, EProxy) else _type_name_from_proxy(s))
                        for s in classifier.eSuperTypes
                    ]
                },
                "associations": associations,
            })

        # ── EEnum → enum ───────────────────────────────────────────────────
        elif isinstance(classifier, EEnum):
            enums.append({
                "name": classifier.name,
                "literals": [lit.name for lit in classifier.eLiterals],
            })

    # Recurse into sub-packages
    for sub in pkg.eSubpackages:
        sub_entities, sub_enums = _serialise_package(sub)
        entities.extend(sub_entities)
        enums.extend(sub_enums)

    return entities, enums


def _type_name_from_proxy(proxy: EProxy) -> str:
    try:
        proxy.force_resolve()
        return proxy.name
    except Exception:
        return repr(proxy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ecore_to_dict(ecore_path: str | Path) -> dict:
    """Load *ecore_path* and return the focused metamodel dictionary."""
    ecore_path = Path(ecore_path).resolve()
    if not ecore_path.exists():
        raise FileNotFoundError(f"File not found: {ecore_path}")

    rset = ResourceSet()
    resource = rset.get_resource(URI(str(ecore_path)))

    all_entities: list[dict] = []
    all_enums: list[dict] = []

    for root in resource.contents:
        if isinstance(root, EPackage):
            ents, enums = _serialise_package(root)
            all_entities.extend(ents)
            all_enums.extend(enums)

    return {
        "entities": all_entities,
        "enums": all_enums,
    }


def ecore_to_json(ecore_path: str | Path, indent: int | None = None) -> str:
    return json.dumps(ecore_to_dict(ecore_path), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ecore_to_json",
        description="Convert a .ecore metamodel to a focused JSON (entities, attributes, associations, inheritance).",
    )
    p.add_argument("ecore_file", help="Path to the .ecore file")
    p.add_argument("-o", "--output", metavar="FILE", help="Write JSON to FILE instead of stdout")
    p.add_argument("-p", "--pretty", action="store_true", help="Pretty-print (2-space indent)")
    args = p.parse_args(argv)

    try:
        result = ecore_to_json(args.ecore_file, indent=2 if args.pretty else None)
    except FileNotFoundError as exc:
        p.error(str(exc))
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()