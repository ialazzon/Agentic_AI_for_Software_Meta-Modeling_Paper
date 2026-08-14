import sys
from pyecore.resources import ResourceSet, URI
from pyecore.ecore import EObject, EReference, EAttribute


def load_metamodel(ecore_path, rset):
    resource = rset.get_resource(URI(ecore_path))
    root = resource.contents[0]
    rset.metamodel_registry[root.nsURI] = root
    for child in root.eAllContents():
        if hasattr(child, 'nsURI') and child.nsURI:
            rset.metamodel_registry[child.nsURI] = child
    return root


def load_instance(xmi_path, rset):
    resource = rset.get_resource(URI(xmi_path))
    return resource


def get_name(obj):
    for attr in ('name', 'id', 'title', 'label'):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val:
                return str(val)
    return None


def describe_attributes(obj):
    parts = []
    for feature in obj.eClass.eAllStructuralFeatures():
        if isinstance(feature, EAttribute):
            val = getattr(obj, feature.name, None)
            if val is None or val == [] or val == '':
                continue
            if feature.many:
                vals = ", ".join(str(v) for v in val)
                parts.append(f"{feature.name} = [{vals}]")
            else:
                parts.append(f"{feature.name} = '{val}'")
    return parts


def describe_references(obj, obj_labels):
    parts = []
    for feature in obj.eClass.eAllStructuralFeatures():
        if isinstance(feature, EReference):
            val = getattr(obj, feature.name, None)
            if val is None or val == []:
                continue
            if feature.many:
                targets = [obj_labels.get(id(t), get_name(t) or t.eClass.name)
                           for t in val]
                if targets:
                    parts.append(
                        f"is linked via '{feature.name}' to "
                        + ", ".join(targets)
                    )
            else:
                target = obj_labels.get(id(val), get_name(val) or val.eClass.name)
                parts.append(f"has '{feature.name}' referencing {target}")
    return parts


def generate_description(resource):
    all_objects = []
    for root in resource.contents:
        all_objects.append(root)
        all_objects.extend(list(root.eAllContents()))

    # Build stable labels for each object
    obj_labels = {}
    type_counts = {}
    for obj in all_objects:
        tname = obj.eClass.name
        name = get_name(obj)
        if name:
            label = f"the {tname} '{name}'"
        else:
            type_counts[tname] = type_counts.get(tname, 0) + 1
            label = f"a {tname} (#{type_counts[tname]})"
        obj_labels[id(obj)] = label

    # Aggregate type counts for overview
    overview_counts = {}
    for obj in all_objects:
        overview_counts[obj.eClass.name] = overview_counts.get(obj.eClass.name, 0) + 1

    sentences = []

    # Overview sentence
    total = len(all_objects)
    type_summary = ", ".join(
        f"{count} {tname}{'s' if count != 1 else ''}"
        for tname, count in sorted(overview_counts.items())
    )
    sentences.append(
        f"This model instance contains {total} element"
        f"{'s' if total != 1 else ''} in total, comprising {type_summary}."
    )

    # Per-object descriptions
    for obj in all_objects:
        label = obj_labels[id(obj)]
        clause = label[0].upper() + label[1:]
        attrs = describe_attributes(obj)
        # Remove the name attribute already used in the label to avoid redundancy
        name = get_name(obj)
        attrs = [a for a in attrs if not (name and a.endswith(f"'{name}'")
                 and a.startswith(('name', 'id', 'title', 'label')))]
        refs = describe_references(obj, obj_labels)

        fragments = []
        if attrs:
            fragments.append("with " + ", ".join(attrs))
        if refs:
            fragments.append("and " + "; ".join(refs))

        if fragments:
            sentences.append(f"{clause} is defined " + " ".join(fragments) + ".")
        else:
            sentences.append(f"{clause} is present with no further details.")

    return " ".join(sentences)


def main():
    if len(sys.argv) != 3:
        print("Usage: python describe_model.py <metamodel.ecore> <instance.xmi>")
        sys.exit(1)

    ecore_path, xmi_path = sys.argv[1], sys.argv[2]

    rset = ResourceSet()
    load_metamodel(ecore_path, rset)
    resource = load_instance(xmi_path, rset)

    description = generate_description(resource)
    print(description)


if __name__ == "__main__":
    main()