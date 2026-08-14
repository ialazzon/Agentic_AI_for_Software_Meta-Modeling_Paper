from config import set_environment

set_environment()

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

from ecore_to_json import ecore_to_json
from xmi_converter import deserialize_model
from check_conformance import check_conformance_model_to_metamodel
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from ocl_runner import run_ocl_validation
from ocl_runner import ValidationReport
import re

# ----------------------------
# TOOL: External OCL Validator
# ----------------------------
def run_ocl_constraints(xmi_path: str, ocl_constraints: str) -> Dict:
    """
    External tool (assumed provided).
    Runs OCL constraints against XMI model.
    """
    raise NotImplementedError("Connect to your OCL engine here")


# ----------------------------
# STATE
# ----------------------------
class ModelState(TypedDict):
    ecore_path: str
    ecore_content: str
    user_spec: str

    xmi_model: str
    xmi_path: str

    ocl_constraints: list[str]
    ocl_errors: str
    verification_result: Dict
    ocl_iteration: int

    iteration: int
    max_iterations: int
    conformance_passed: bool
    conformance_errors: str  # feedback passed back to LLM in generate_xmi

    # --- multi-instance generation loop ---
    number_of_model_instances_to_generate: int
    current_instance_index: int
    generated_specs: List[str]
    generated_instances: List[Dict[str, Any]]
    additional_constraints: str  # optional extra constraints the user wants applied to every instance


# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(model="gpt-4", temperature=0)


# ----------------------------
# NODE 1: Load Ecore
# ----------------------------
def load_ecore(state: ModelState) -> ModelState:
    source_ecore = state["ecore_path"]
    source_metamodel_json = ecore_to_json(source_ecore, indent=2)
    # with open(state["ecore_path"], "r", encoding="utf-8") as f:
    #     ecore = f.read()

    return {
        **state,
        "ecore_content": source_metamodel_json
    }


# ----------------------------
# NODE 1.5: Generate Model Spec (NEW — replaces human natural-language input)
# ----------------------------
def generate_model_spec(state: ModelState) -> ModelState:
    """
    Agent that autonomously proposes a natural-language description of a
    valid model instance for the given Ecore metamodel. This replaces the
    human typing a spec at the console. It is called once per instance we
    want to generate, and must produce a DIFFERENT instance every time
    (tracked via state["generated_specs"]).
    """
    previous_specs = state.get("generated_specs", [])
    instance_number = state.get("current_instance_index", 0) + 1
    total = state.get("number_of_model_instances_to_generate", 1)
    additional_constraints = state.get("additional_constraints", "")

    previous_specs_block = (
        "\n\n".join(f"Instance {i + 1}:\n{s}" for i, s in enumerate(previous_specs))
        if previous_specs else "None yet — this is the first instance."
    )

    messages = [
        ("system",
         "You are an expert EMF/Ecore modeler acting as a requirements author. "
         "Given an Ecore metamodel (serialized as JSON), write a natural-language "
         "specification describing ONE concrete, valid instance of that metamodel. "
         "The specification must be detailed enough for another engineer to build "
         "a conforming model from it (mention concrete class instances, attribute "
         "values, and how references/containments connect them), while respecting "
         "the metamodel's structure (types, multiplicities, containment, abstract "
         "classes, enumerations — always describe concrete/instantiable classes, "
         "never abstract ones). "
         "Do not mention OCL, XMI, YAML, or any implementation detail — write "
         "plain natural language describing the model content itself. "
         "You are generating instance {instance_number} of {total} instances. "
         "Every instance must be MEANINGFULLY DIFFERENT from all previously "
         "generated instances (different values, different quantities, different "
         "structural choices) — do not repeat the same specification. "
         "Always respect any additional constraints the user has provided; those "
         "constraints apply to every single instance. "
         "Return ONLY the natural-language specification text, with no preamble, "
         "no headers, and no markdown formatting."),
        ("user",
         "ECORE METAMODEL:\n{ecore}\n\n"
         "ADDITIONAL USER CONSTRAINTS (apply to every instance, if any):\n{constraints}\n\n"
         "PREVIOUSLY GENERATED INSTANCE SPECIFICATIONS (must not be repeated):\n{previous}\n\n"
         "Write the specification for instance {instance_number} of {total}.")
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm

    response = chain.invoke({
        "ecore": state["ecore_content"],
        "constraints": additional_constraints or "None.",
        "previous": previous_specs_block,
        "instance_number": instance_number,
        "total": total
    })

    new_spec = response.content.strip()
    print(f"\n=== Generated spec for instance {instance_number}/{total} ===\n{new_spec}\n")

    return {
        **state,
        "user_spec": new_spec,
        "generated_specs": previous_specs + [new_spec],
        # reset per-instance transient fields so this instance starts clean
        "iteration": 0,
        "conformance_passed": False,
        "conformance_errors": "",
        "ocl_errors": "",
        "ocl_iteration": 0,
    }


# ----------------------------
# NODE 2: Generate XMI Model
# ----------------------------
def generate_xmi(state: ModelState) -> ModelState:
    conformance_errors = state.get("conformance_errors", "")
    ocl_errors = state.get("ocl_errors", "")
    iteration = state.get("iteration", 0)
    
    messages = [
("system",
 "You are an expert EMF/Ecore modeler. "
 "Generate a valid YAML representation of an XMI instance that conforms exactly to the given Ecore metamodel serialized json. "
 "Do not invent any new thing not stated or mentioned by the user. "
 "Do not invent or assign values for primitive-typed attributes (e.g. EString, EInt, EBoolean, EDouble) that are not explicitly provided by the user, even if the attribute has a lower bound of 1 or more. "
 "Only include a primitive attribute in the output if the user explicitly supplied its value; otherwise omit the attribute entirely rather than fabricating a default or placeholder value. "
 "For example, given:\n"
 "  <chapters title=\"Chapter 1\" nbPages=\"10\" author=\"Author 1\"/>\n"
 "if 'title' has a lower bound of 1 or more but no value was provided by the user, do not invent a value for it. "
 "NEVER instantiate an abstract EClass (one whose 'abstract' or 'eAbstract' property is true, or which is an EClass marked as an interface). "
 "Abstract classes cannot be instantiated directly and doing so causes a 'Can't instantiate abstract EClass' error. "
 "Whenever a value must be created for a reference whose declared type is an abstract EClass, choose a CONCRETE (non-abstract) subclass of that type from the metamodel and use that subclass name as the '_type'. "
 "If multiple concrete subclasses exist, pick the one whose attributes/references best match what the user provided; if the user gave no information to disambiguate, choose the most specific concrete subclass that requires no fabricated values, or omit the element if none fits without inventing data. "
 "The '_type' field must ALWAYS refer to a concrete instantiable EClass, never an abstract one. "
 "Add _type and _id inferring from the metamodel's YAML. "
 "=== ABSTRACT CLASS RULES (CRITICAL) ===\n"
 "NEVER set '_type' to an abstract EClass. An EClass is abstract when its metamodel definition has 'abstract': true (or 'eAbstract': true) or is an interface. "
 "Instantiating an abstract EClass causes a fatal 'Can't instantiate abstract EClass' error. "
 "Before emitting any element, locate its EClass in the metamodel and confirm 'abstract' is false. If it is true, you MUST instead pick a CONCRETE (non-abstract) subclass of that class and use the subclass name as '_type'. "
 "When a reference's declared type is an abstract EClass, the actual element you create for it must be one of its concrete subclasses — never the abstract type itself. "
 "Enumerate the concrete subclasses available in the metamodel and choose the one whose attributes/references best match the user's data. "
 "If the user's information does not indicate which concrete subclass to use, pick the most specific concrete subclass that requires no fabricated values; if none fits without inventing data, omit the element rather than emit an abstract '_type'. "
 "The '_type' field must ALWAYS name a concrete instantiable EClass. Never output a generic base type (such as a top-level 'Element' / 'RailwayElement' / 'AbstractX' style class) when concrete subclasses exist. "
 "\n"
 "=== ENUM (EEnum) RULES (CRITICAL) ===\n"
 "An attribute is enum-typed when its 'eType' in the metamodel refers to an EEnum (a classifier whose definition lists 'eLiterals'). Treat these attributes completely differently from primitive (EString/EInt/EBoolean/EDouble) attributes. "
 "For every enum-typed attribute you intend to emit, FIRST locate the corresponding EEnum in the metamodel and read its 'eLiterals' list. The value you output MUST be EXACTLY one of the literal NAMES defined there — copied verbatim, character-for-character, including case. "
 "Do NOT output an enum literal's integer 'value'/ordinal (e.g. 0, 1, 2); always output the literal NAME string. Outputting a number where an enum literal is expected causes a fatal 'argument of type \\'int\\' is not iterable' error during deserialization. "
 "Do NOT invent, translate, pluralize, abbreviate, re-case, or otherwise alter enum literal names. If the user's wording (e.g. 'street') does not exactly match a defined literal, map it to the closest defined literal NAME exactly as spelled in the metamodel (e.g. 'Street'); if no literal reasonably matches and the attribute is optional, OMIT the attribute rather than guessing. "
 "Never quote an enum value as if it were a free-text string value the user invented — it must be a real literal name from the EEnum. Never place a primitive/free-text value in an enum-typed attribute, and never place an enum literal name in a primitive attribute. "
 "If an enum-typed attribute's value was not supplied by the user, OMIT it entirely (do not fabricate a default literal), unless the metamodel marks it required AND defines a defaultValueLiteral, in which case you may use that exact default literal name. "
 "Enum values are single literal names, never lists, unless the attribute's upper bound is greater than 1; in that case output a YAML list where every element is a valid literal NAME from the same EEnum. "
 "Before finalizing, re-scan every attribute you emitted: for each one whose metamodel eType is an EEnum, confirm the emitted value is an exact member of that EEnum's literal names and is a string (not a number). "
 "\n"
 "YAML contents should not start with any special heading or symbols such as ` or ```. "
 "Emphasize: YAML contents should not start with any special heading or symbols such as ` or ```. "
 """ An example is:
 _type: AllocationProblem
 ID: 'Alloc1'
components:
- _type: Component
  _id: //@components.0
  compName: '0'
  resConsumptions:
  - //@resConsumptions.0
  - //@resConsumptions.1

- _type: Component
  _id: //@components.1
  compName: '1'
  resConsumptions:
  - //@resConsumptions.8
  - //@resConsumptions.9
 """),
("user",
 "ECORE METAMODEL:\n{ecore}\n\n"
 "USER SPECIFICATION:\n{spec}\n\n"
 "{error_section}"
 "Before emitting any element, verify its '_type' is a concrete (non-abstract) EClass in the metamodel. "
 "For every attribute, verify whether its eType is an EEnum; if so, ensure the value is an exact literal NAME from that EEnum (a string, never a number). "
 "Return ONLY the YAML content.")
    ]
    
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    
    # Build the error section — conformance errors take priority; otherwise show OCL errors
    if conformance_errors:
        error_section = (
            f"PREVIOUS ATTEMPT FAILED CONFORMANCE CHECK (iteration {iteration}).\n"
            f"Errors:\n{conformance_errors}\n\n"
            f"Fix the above issues and regenerate the YAML.\n\n"
        )
    elif ocl_errors:
        error_section = (
            f"PREVIOUS ATTEMPT FAILED OCL VALIDATION (iteration {iteration}).\n"
            f"OCL Violations:\n{ocl_errors}\n\n"
            f"Fix the model so it satisfies all OCL constraints and regenerate the YAML.\n\n"
        )
    else:
        error_section = ""
    
    response = chain.invoke({
        "ecore": state["ecore_content"],
        "spec": state["user_spec"],
        "error_section": error_section
    })


    yaml_content = response.content
    updated_yaml_content = re.sub(r'^.*?```yaml\s*\n?', '', yaml_content, flags=re.DOTALL)
    updated_yaml_content = re.sub(r'\n?```.*$', '', updated_yaml_content, flags=re.DOTALL)
    print('***',updated_yaml_content,'***')
    instance_idx = state.get("current_instance_index", 0) + 1
    xmi_path = rf".\models3\generated_model_{instance_idx}.model"
    deserialize_model(state["ecore_path"], updated_yaml_content, xmi_path)

    
    # with open(xmi_path, "w", encoding="utf-8") as f:
    #     f.write(xmi)

    return {
        **state,
        "xmi_path": xmi_path,
        "conformance_errors": "",        # reset until check_conformance repopulates
        "conformance_passed": False,     # reset, check_conformance will set it
        "ocl_errors": "",            # reset
        "iteration": iteration + 1
    }


# ----------------------------
# NODE 3: Check Conformance
# ----------------------------
def check_conformance(state: ModelState) -> ModelState:
    check_conf = check_conformance_model_to_metamodel(
        state["ecore_path"],
        state["xmi_path"]
    )
    passed = check_conf[0]
    if not passed:
        return {**state, "conformance_passed": False, "conformance_errors": check_conf[1]}
    return {**state, "conformance_passed": True, "conformance_errors": ""}


def route_after_conformance(state: ModelState) -> str:
    if state["conformance_passed"]:
        return "generate_ocl"
    elif state.get("iteration", 0) >= 5:# state.get("max_iterations", 1):
        return "human_feedback"   # instead of force-ending
    else:
        return "generate_xmi"
    

# ----------------------------
# NODE 4: Human Feedback
# ----------------------------    
def human_feedback(state: ModelState) -> ModelState:
    """Pauses the graph and waits for human input."""
    user_input = interrupt({
        "message": "The LLM could not fix the conformance issues automatically.",
        "errors": state["conformance_errors"],
        "iteration": state["iteration"],
        "prompt": "Please provide clarification or instructions to fix the issue:"
    })
    
    # user_input is whatever the human types — append it to the spec
    updated_spec = state["user_spec"] + f"\n\nUser clarification (iteration {state['iteration']}):\n{user_input}"
    
    return {
        **state,
        "user_spec": updated_spec,
        "iteration": 0,  # reset iteration counter after human intervenes
        "conformance_errors": ""
    }
    
# ----------------------------
# NODE 5: Generate OCL Constraints
# ----------------------------
def generate_ocl(state: ModelState) -> ModelState:
    """
    Called after conformance passes. Uses all accumulated user specifications
    to generate OCL constraints — one per line, no extras.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
"You are a formal verification expert in OCL (Object Constraint Language). "
"Generate OCL self-contained invariant expressions for an EMF model instance.\n"

"=== OUTPUT FORMAT ===\n"
"- One constraint per line, no blank lines\n"
"- Each line must be a complete, standalone OCL invariant expression\n"
"- Do NOT include 'context', 'inv:', or any OCL file header — just the body expression\n"
"- All expressions are evaluated with 'self' bound to the root_class (first class in metamodel) instance\n"
"- Do NOT number lines or add bullet points\n"
"- Do NOT add any explanation, comments, or markdown\n"

"=== FEATURE RESOLUTION (Unresolved Property errors) ===\n"
"- Only navigate via association/reference/attribute names that are ACTUALLY DECLARED on the element's current static type. Never invent or assume a property name — if you need persons, projects, assignedTo, name, firstname, from, straight, divergent, etc., that exact role/attribute name must exist on the source class in the metamodel. Referencing a non-existent feature causes an 'Unresolved Property' parse error.\n"
"- Before referencing ANY feature, identify the current static type of the expression to its left and confirm the feature is declared on that type or one of its supertypes. If it is declared only on a subtype, you MUST oclAsType to that subtype first.\n"
"- Only reference a named classifier (in oclIsKindOf, oclIsTypeOf, oclAsType) if that classifier is actually declared in the metamodel. Referencing an undeclared type (e.g. oclIsKindOf(Signal) when no Signal class exists) is itself a parse error. Do not introduce types the metamodel does not define.\n"

"=== STATIC TYPE vs RUNTIME FILTER (the most common error) ===\n"
"- oclIsKindOf inside a select/forAll/exists predicate is ONLY a runtime filter — it does NOT change the static type of the iterator variable. The variable keeps its declared (super)type, so any subtype-only property referenced inside that same predicate will still fail with 'Unresolved Property'. You must oclAsType the collection BEFORE the predicate that uses subtype features.\n"
"- Properties only resolve against the STATIC declared type. select() does not narrow the static type — only oclAsType does. Never rely on an oclIsKindOf guard to make a subtype property resolve.\n"
"- Separate type-narrowing from the value test into two stages, for counting and quantifying alike:\n"
"- WRONG: coll->select(n | n.oclIsKindOf(Sub) and n.subProp = X)  — n is still the supertype, n.subProp is unresolved\n"
"- RIGHT: coll->select(n | n.oclIsKindOf(Sub)).oclAsType(Sub)->select(t | t.subProp = X)\n"
"- RIGHT: coll->select(n | n.oclIsKindOf(Sub)).oclAsType(Sub)->forAll(t | t.subProp = X)\n"
"- RIGHT: coll->select(n | n.oclIsKindOf(Sub)).oclAsType(Sub)->exists(t | t.subProp = X)\n"

"=== SUBTYPE FEATURES ON A SINGLE OBJECT INSIDE ONE PREDICATE ===\n"
"- The same static-type rule applies when, INSIDE a single select/exists/forAll predicate, you guard one object with oclIsKindOf(Sub) and then want several subtype-only features of THAT SAME object. Each such access must be individually cast with e.oclAsType(Sub).feature — the guard does not retype e for the rest of the predicate.\n"
"- WRONG: elements->select(e | e.oclIsKindOf(Turnout) and e.from.oclIsKindOf(X) and e.straight.oclIsKindOf(Y))  — 'from' and 'straight' are declared on Turnout, but e is still the supertype here\n"
"- RIGHT: elements->select(e | e.oclIsKindOf(Turnout) and e.oclAsType(Turnout).from.oclIsKindOf(X) and e.oclAsType(Turnout).straight.oclIsKindOf(Y))\n"
"- Equivalent and cleaner when ALL features in the predicate are subtype-only: narrow the collection first, then test — elements->select(e | e.oclIsKindOf(Turnout)).oclAsType(Turnout)->select(t | t.from.oclIsKindOf(X) and t.straight.oclIsKindOf(Y))\n"

"=== COLLECTION vs OBJECT NAVIGATION (arrow vs dot) ===\n"
"- A collection-valued navigation step like '->select(c | c.name = 'X')' returns a COLLECTION, NOT a single object. You CANNOT navigate a further reference directly off a collection with '->' as if it were an object's feature. Navigating a reference off each element of a collection requires the dot operator, which auto-flattens: 'coll->select(...).referenceName'. Using 'collection->referenceName' (arrow before a reference name) is WRONG and yields 'Unresolved Property' because the reference is declared on the element type, not on the Collection type.\n"
"- WRONG: self.company->select(c | c.name = 'X')->persons->size() = 2  — 'persons' is on Company, not on the OrderedSet\n"
"- RIGHT: self.company->select(c | c.name = 'X').persons->size() = 2  — dot navigates 'persons' on each Company and flattens, then '->size()' operates on the collection\n"
"- RIGHT: self.company->select(c | c.name = 'X').persons->exists(p | p.firstname = 'Max' and p.lastname = 'Trump')\n"
"- Use '->' ONLY for collection operations (select, exists, forAll, size, notEmpty, isEmpty, oclAsType on a collection, etc.). Use '.' to access an attribute or navigate a reference declared on the element/object type.\n"

"=== CASTING: COLLECTION vs SINGLE OBJECT ===\n"
"- When casting a COLLECTION to a subtype, use '.oclAsType(Sub)' (dot) applied to the collection so each element is cast: coll->select(...).oclAsType(Sub)->forAll(...). This is the one collection-to-subtype idiom.\n"
"- When casting a SINGLE OBJECT, apply oclAsType directly to that object: obj.oclAsType(Sub).subProperty, e.g. cu.oclAsType(ControlUnit).processor.clockSpeed.\n"
"- Do NOT use '->oclAsType(Sub)' on a single object, and do NOT use '.oclAsType(Sub)' expecting collection semantics on a single object. Match the cast form to whether the left side is a collection or an object.\n"

"=== TYPE-CHECK OPERATOR CHOICE ===\n"
"- When checking an object's type, prefer oclIsKindOf over oclIsTypeOf so that subtypes are included.\n"
"- For abstract classes, NEVER use oclIsTypeOf (no instance has an abstract type exactly); always use oclIsKindOf.\n"
"- Only use oclIsTypeOf when the user explicitly requires an exact, subtype-excluding type match.\n"
"- Note that x.oclIsKindOf(T) is always true when T is x's own declared static type or any supertype of it; such a guard filters nothing. Only guard against an actual subtype if you intend to narrow.\n"

"=== NULL / UNDEFINED SAFETY ===\n"
"- Navigating a reference that may be unset can yield null/invalid and break downstream operations. When a single-valued reference may be absent before you cast or navigate it, guard with '<> null' (or check ->notEmpty() for collection-valued ends) before accessing its features.\n"
"- A subtype guard does not imply non-null: e.oclIsKindOf(Turnout) tells you the type if e is set, but combine type and null reasoning where the model permits unset references.\n"

"=== DATES ===\n"
"- Date attributes do NOT coerce from string literals; p.start = '<any string>' is always false. Equality and ISO-prefix matching both fail.\n"
"- The engine serializes dates as Java Date.toString(): 'Fri Jan 01 00:00:00 GST 2021' (weekday, 3-letter month, 2-digit day, time, tz, 4-digit year-last).\n"
"- Match dates via fixed-position substrings on toString(): month = substring(5,7) (e.g. 'Jan'), day = substring(9,10) (e.g. '01'), year via endsWith (e.g. endsWith('2021')). Verify 1-based inclusive indexing in the target dialect before relying on it.\n"
"- WRONG: p.start = '2021-01-01T00:00:00.000000'\n"
"- WRONG: p.start.toString().startsWith('2021-01-01')\n"
"- RIGHT: p.start.toString().substring(5,7) = 'Jan' and p.start.toString().substring(9,10) = '01' and p.start.toString().endsWith('2021')\n"

"=== SELF-CONTAINMENT ===\n"
"- Every expression must be evaluable on its own with only 'self' in scope. Do not reference iterator variables from other lines, and do not assume helper definitions, let-bindings, or 'def:' clauses exist.\n"

"=== PRE-EMISSION CHECKLIST (apply silently to every line) ===\n"
"1. Does every named type actually exist in the metamodel?\n"
"2. For every feature access, is the feature declared on the current static type (post-cast), not merely on a subtype guarded by oclIsKindOf?\n"
"3. Is every reference navigated off a collection done with '.', and every collection operation done with '->'?\n"
"4. Is each oclAsType form (collection-dot vs object-dot) matched to the left side?\n"
"5. Are possibly-unset references guarded before navigation?\n"
"6. Are dates matched via toString() substrings, never literal equality?\n"

"=== EXAMPLE OUTPUT ===\n"
"self.components->notEmpty()\n"
"self.components->forAll(c | c.compName <> '')\n"
"self.resConsumptions->size() >= self.components->size()\n"
"self.elements->forAll(e | e.oclIsKindOf(AbstractElement))\n"
"self.company->select(c | c.name = 'Datastar').persons->size() = 2\n"
"self.company->select(c | c.name = 'Datastar').persons->exists(p | p.firstname = 'Max' and p.lastname = 'Trump')\n"
"self.company->select(c | c.name = 'Datastar').projects->exists(p | p.name = 'Concert')\n"
"self.company->select(c | c.name = 'Datastar').persons->select(p | p.firstname = 'Max' and p.lastname = 'Trump').assignedTo->exists(a | a.name = 'Concert')\n"
"self.company->select(c | c.name = 'Datastar').projects->select(p | p.oclIsKindOf(National)).oclAsType(National)->exists(n | n.name = 'Concert' and n.budget = 10000)\n"
"self.children->select(c | c.oclIsKindOf(ControlUnit)).oclAsType(ControlUnit)->forAll(cu | cu.processor.clockSpeed = 3200)\n"
"self.elements->select(e | e.oclIsKindOf(Turnout)).oclAsType(Turnout)->exists(t | t.from.oclIsKindOf(Segment) and t.straight.oclIsKindOf(Turnout) and t.divergent.oclIsKindOf(Segment))\n"
"self.elements->select(e | e.oclIsKindOf(Turnout) and e.oclAsType(Turnout).from <> null and e.oclAsType(Turnout).from.oclIsKindOf(Segment))->size() = 1\n"
"self.workflows->forAll(w | w.nodes->select(n | n.oclIsKindOf(AutomaticTask)).oclAsType(AutomaticTask)->select(t | t.name = 'GrindBeans')->size() = 1)\n"
"self.workflows->forAll(w | w.nodes->select(n | n.oclIsKindOf(AutomaticTask)).oclAsType(AutomaticTask)->exists(t | t.name = 'Brewing' and t.duration = 10))\n"
"self.projects->exists(p | p.shortname = 'AIResearch' and p.start.toString().substring(5,7) = 'Jan' and p.start.toString().substring(9,10) = '01' and p.start.toString().endsWith('2021') and p.end.toString().substring(5,7) = 'Dec' and p.end.toString().substring(9,10) = '31' and p.end.toString().endsWith('2025') and p.devmail = 'ai-research@foundation.org' and p.homepage = 'www.ai-research.com')"
),
        ("user",
         "ECORE METAMODEL:\n{ecore}\n\n"
         "USER SPECIFICATION (all requirements):\n{spec}\n\n"
         "Generate OCL constraints that fully capture the user's intent. "
         "Return only constraint expressions, one per line.")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "ecore": state["ecore_content"],
        "spec":  state["user_spec"],
    })

    # Parse into a clean list[str] — strip blanks, comments, accidental backticks
    raw_lines = response.content.splitlines()
    constraints = [
        line.strip()
        for line in raw_lines
        if line.strip() and not line.strip().startswith("--") and not line.strip().startswith("`")
    ]

    print("\n".join(constraints))
    return {
        **state,
        "ocl_constraints": constraints,   # list[str], one constraint per element
        "ocl_iteration": 0,        # reset every time we regenerate constraints
    }


# ----------------------------
# NODE 6: Run OCL Validation
# ----------------------------
def run_ocl_validation_node(state: ModelState) -> ModelState:
    """
    Runs each OCL constraint via the Java validator and collects violations.
    Populates ocl_errors with a structured summary if any constraints fail.
    """
    report: ValidationReport = run_ocl_validation(
        ecore_path=state["ecore_path"],
        xmi_path=state["xmi_path"],
        ocl_expressions=state["ocl_constraints"],  # list[str]
    )

    # Java/process-level failure — treat as a hard error, surface it clearly
    if not report.success:
        error_msg = (
            f"OCL validation could not run:\n{report.raw_error}\n\n"
            f"Review the constraints for syntax errors or unsupported operations."
        )
        return {**state, "ocl_errors": error_msg, "ocl_iteration": state.get("ocl_iteration", 0) + 1}

    # Collect every result that is not cleanly satisfied
    failures = [
        r for r in report.results
        if not r.satisfied or r.severity in ("ERROR", "PARSE_ERROR", "RUNTIME_ERROR")
    ]

    if not failures:
        return {**state, "ocl_errors": "", "ocl_iteration": state.get("ocl_iteration", 0) + 1}  # all constraints passed

    # Build a structured, LLM-readable error summary
    lines = [f"OCL validation failed — {len(failures)} constraint(s) violated:\n"]
    for i, r in enumerate(failures, 1):
        lines.append(f"  [{i}] Constraint : {r.constraint}")
        lines.append(f"      Context    : {r.context}")
        lines.append(f"      Severity   : {r.severity}")
        if r.message:
            lines.append(f"      Message    : {r.message}")
        lines.append("")   # blank line between entries

    print("\n".join(lines))
    return {**state, "ocl_errors": "\n".join(lines),"ocl_iteration": state.get("ocl_iteration", 0) + 1}

def route_after_ocl(state: ModelState) -> str:
    """
    If OCL validation still has errors and we have retries left, go fix the
    XMI. Otherwise (passed, or gave up), the current instance is DONE —
    hand off to finalize_instance to record it and decide whether to
    generate another instance or stop.
    """
    if state.get("ocl_errors", ""):
        if state.get("ocl_iteration", 0) < state.get("max_iterations", 5):
            return "generate_xmi"
        print(
            f"\n[WARN] OCL fix limit reached after {state['ocl_iteration']} attempts. "
            f"Moving on with unresolved violations:\n{state['ocl_errors']}"
        )
    return "finalize_instance"


# ----------------------------
# NODE 7: Finalize Instance (NEW — bookkeeping + loop control)
# ----------------------------
def finalize_instance(state: ModelState) -> ModelState:
    """
    Runs once a model instance's conformance/OCL loop is done (either it
    fully passed, or the retry budget was exhausted). Records the result
    and advances the instance counter so route_after_instance can decide
    whether to generate another instance or stop.
    """
    record = {
        "instance_index": state.get("current_instance_index", 0) + 1,
        "user_spec": state.get("user_spec", ""),
        "xmi_path": state.get("xmi_path", ""),
        "conformance_passed": state.get("conformance_passed", False),
        "ocl_errors": state.get("ocl_errors", ""),
    }
    generated_instances = state.get("generated_instances", []) + [record]

    print(
        f"\n=== Instance {record['instance_index']}/"
        f"{state.get('number_of_model_instances_to_generate', 1)} finalized "
        f"(conformance_passed={record['conformance_passed']}) ===\n"
    )

    return {
        **state,
        "generated_instances": generated_instances,
        "current_instance_index": state.get("current_instance_index", 0) + 1,
    }


def route_after_instance(state: ModelState) -> str:
    """Loop back to generate another instance's spec, or stop once we've
    generated the requested number of instances."""
    if state.get("current_instance_index", 0) < state.get("number_of_model_instances_to_generate", 1):
        return "generate_model_spec"
    return "end"


# ----------------------------
# BUILD LANGGRAPH
# ----------------------------
graph = StateGraph(ModelState)

graph.add_node("load_ecore", load_ecore)
graph.add_node("generate_model_spec", generate_model_spec)
graph.add_node("generate_xmi", generate_xmi)
graph.add_node("check_conformance", check_conformance)
graph.add_node("human_feedback", human_feedback)
graph.add_node("generate_ocl", generate_ocl)
graph.add_node("run_ocl_validation", run_ocl_validation_node)
graph.add_node("finalize_instance", finalize_instance)

graph.set_entry_point("load_ecore")
graph.add_edge("load_ecore", "generate_model_spec")
graph.add_edge("generate_model_spec", "generate_xmi")
graph.add_edge("generate_xmi", "check_conformance")
graph.add_edge("human_feedback", "generate_xmi")
graph.add_edge("generate_ocl", "run_ocl_validation")
 
graph.add_conditional_edges(
    "check_conformance",
    route_after_conformance,
    {
        "generate_ocl": "generate_ocl",
        "generate_xmi": "generate_xmi",
        "human_feedback": "human_feedback"
    }
)

graph.add_conditional_edges(
    "run_ocl_validation",
    route_after_ocl,
    {
        "finalize_instance": "finalize_instance",
        "generate_xmi": "generate_xmi"
    }
)

graph.add_conditional_edges(
    "finalize_instance",
    route_after_instance,
    {
        "generate_model_spec": "generate_model_spec",
        "end": END
    }
)

app = graph.compile(
    checkpointer=MemorySaver()  # required for interrupt to work
)


# ----------------------------
# RUN FUNCTION
# ----------------------------
def run_agent(ecore_path: str, number_of_model_instances_to_generate: int = 1, additional_constraints: str = ""):
    """
    Runs the graph to generate `number_of_model_instances_to_generate`
    distinct, conformant model instances for the given Ecore metamodel.
    `additional_constraints` is optional free-text guidance (from the user)
    that the spec-generating agent will apply to every instance it proposes.
    """
    initial_state: ModelState = {
        "ecore_path": ecore_path,
        "ecore_content": "",
        "user_spec": "",
        "xmi_model": "",
        "xmi_path": "",
        "ocl_constraints": [],
        "ocl_errors": "",
        "verification_result": {},
        "iteration": 0,
        "ocl_iteration": 0,
        "max_iterations": 5,
        "conformance_passed": False,
        "conformance_errors": "",
        "number_of_model_instances_to_generate": number_of_model_instances_to_generate,
        "current_instance_index": 0,
        "generated_specs": [],
        "generated_instances": [],
        "additional_constraints": additional_constraints,
    }

    thread = {"configurable": {"thread_id": "1"}, "recursion_limit": 200}

    for event in app.stream(initial_state, config=thread):
        pass

    state = app.get_state(thread)

    while state.next:
        # This can still happen if a single instance repeatedly fails
        # conformance and needs human clarification.
        print("\n--- HUMAN FEEDBACK REQUIRED (conformance could not be auto-fixed) ---")
        print("Conformance errors:", state.values.get("conformance_errors"))
        user_input = input("Provide clarification: ")

        for event in app.stream(Command(resume=user_input), config=thread):
            pass

        state = app.get_state(thread)

    return state.values["generated_instances"]  # <-- list of per-instance results

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    ecore_path = input("Enter path to .ecore file: ")
    n = int(input("How many model instances do you want to generate? "))
    extra_constraints = input(
        "Any additional constraints to apply to every instance (optional, press Enter to skip): "
    )

    results = run_agent(
        ecore_path,
        number_of_model_instances_to_generate=n,
        additional_constraints=extra_constraints
    )

    print(f"\nGenerated {len(results)} model instance(s).")
    for r in results:
        print(r)