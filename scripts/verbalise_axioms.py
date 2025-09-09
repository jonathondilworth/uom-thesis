import sys
import argparse
import collections
import json
from rdflib import Graph, URIRef, Literal, BNode, Namespace, RDF
from rdflib.namespace import RDFS, OWL
from tqdm import tqdm


def format(graph, node):
    """get a text literal for any node $n$ in graph $g$"""
    if isinstance(node, Literal):
        return str(node)
    if isinstance(node, (URIRef, BNode)):
        label = next(graph.objects(node, RDFS.label), None)
        if (isinstance(label, Literal)):
            return str(label)
    # else, fallback:
    return str(node)


def is_resource(node) -> bool:
    """functional identifier for resource (URIRef, BNode) nodes"""
    return isinstance(node, (URIRef, BNode))


def is_named(node) -> bool:
    """functional identifier for named (URIRef) ndes"""
    return isinstance(node, URIRef)


def classify_nodes(graph):
    """find all C, D in O s.t. C sqsubseteq D, or C equiv D"""
    subclass_edges = set()
    for itr_idx, (subj, pred, ob) in enumerate(tqdm(graph.triples((None, RDFS.subClassOf, None)))):
        subclass_edges.add((subj, ob)) # path : subj -> obj | subClassOf(path) holds

    equiv_edges = set()
    for itr_idx, (subj, pred, ob) in enumerate(tqdm(graph.triples((None, OWL.equivalentClass, None)))):
        equiv_edges.add((subj, ob)) # path : subj -> obj | equivalence(path) holds
    
    resource_nodes = set()
    for itr_idx, (subject, object) in enumerate(tqdm(subclass_edges | equiv_edges)):
        # unionise participating nodes s.t. n != literal forall n in sqsubseteq UNION equiv
        if is_resource(subject):
            resource_nodes.add(subject)
        if is_resource(object):
            resource_nodes.add(object)

    return resource_nodes, subclass_edges, equiv_edges


def map_antecedents(subclass_relationships):
    """maps all child to parent relationships"""
    parents = collections.defaultdict(set)
    for subject, objt in subclass_relationships:
        if not isinstance(objt, BNode):
            parents[subject].add(objt)
    return parents


def map_descendants(subclass_relationships):
    """maps all parent to child relationships"""
    children = collections.defaultdict(set)
    for subject, objt in subclass_relationships:
        if not isinstance(objt, BNode):
            children[objt].add(subject)
    return children


def get_snomed_top_level_concept(rdf_xml_or_ttl_graph):
    """searches a graph for SNOMED CONCEPT, i.e. http://snomed.info/id/138875005"""
    snomed_tld = URIRef("http://snomed.info/id/138875005")
    if (snomed_tld, None, None) in rdf_xml_or_ttl_graph:
        return snomed_tld
    # else, exhausted the graph searching for SNOMED CONCEPT:
    raise KeyError("Cannot find SNOMED CONCEPT in the graph.")


def identify_root_nodes(nodes, parent_mappings):
    """finds nodes with no parents"""
    root_nodes = []
    for node in nodes:
        if len(parent_mappings.get(node, set())) == 0:
            root_nodes.append(node)
    return root_nodes


def identify_named_nodes(nodes):
    """returns the set of named classes given a node set/list"""
    named_cls_nodes = []
    for node in nodes:
        if is_named(node):
            named_cls_nodes.append(node)
    return named_cls_nodes


def map_equivalence(equivalence_relationships):
    """maps all C equiv D"""
    equivalence_map = collections.defaultdict(set)
    for C, D in equivalence_relationships:
        if is_resource(C) and is_resource(D):
            equivalence_map[C].add(D)
            equivalence_map[D].add(C)
    # symmetric, reflexive, transitive
    return equivalence_map


def bfs_and_verbalise(graph, top_level_nodes, parents, children, equiv_mappings, max_depth):
    seen = set()
    node_queue = collections.deque()
    # queue up TLD/Cs
    for node in top_level_nodes:
        node_queue.append((node, 0))
    # begin processing the queue (grows as while loops)
    while node_queue:
        current_node, depth = node_queue.popleft()
        if current_node in seen:
            continue
        # else:
        seen.add(current_node)
        # verbalise equivalent classes
        for eq_cls in sorted(
            equiv_mappings.get(current_node, ()), key=lambda x: format(graph, x)
        ):
            print(f"{format(graph, current_node)} is equivalent to {format(graph, eq_cls)}")
        # ...
        # verbalise subclass axioms
        for parent in sorted(
            parents.get(current_node, ()), key=lambda x: format(graph,x)
        ):
            print(f"{format(graph, current_node)} is a subclass of {format(graph, parent)}")
        # ...
        # traverse to children unless depth cap hit
        if max_depth is None or depth < max_depth:
            for child in sorted(
                children.get(current_node, ()), key=lambda x: format(graph,x)
            ):
                node_queue.append((child, depth + 1))


def render_verbalisation(graph: Graph, axiom) -> str:
    """transform a subset of OWL axioms to a verbalised form"""
    if isinstance(axiom, URIRef) or not isinstance(axiom, BNode):
        return format(graph, axiom)
    
    # owl:Restriction
    if (axiom, RDF.type, OWL.Restriction) in graph:
        on_property = next(graph.objects(axiom, OWL.onProperty), None)
        some_values_from = next(graph.objects(axiom, OWL.someValuesFrom), None)
        constituents = []
        dl = []
        if on_property is not None and some_values_from is not None:
            constituents.append(f"{format(graph, on_property)} some {render_verbalisation(graph, some_values_from)}")
        if constituents:
            return " and ".join(constituents)
        # else:
        return format(graph, axiom)

    # owl:intersectionOf (RDF list)
    collection = next(graph.objects(axiom, OWL.intersectionOf), None)
    if collection is not None:
        items = []
        while collection and collection != RDF.nil:
            head = next(graph.objects(collection, RDF.first), None)
            if head is not None:
                items.append(render_verbalisation(graph, head))
            collection = next(graph.objects(collection, RDF.rest), None)
        return " and ".join(items)

    # fallback, anonymous expression:
    return format(graph, axiom)


def render_description_logic_string(graph: Graph, axiom) -> str:
    """transform a subset of OWL axioms to a basic description logic string"""
    if isinstance(axiom, URIRef) or not isinstance(axiom, BNode):
        return format(graph, axiom)
    
    # owl:Restriction
    if (axiom, RDF.type, OWL.Restriction) in graph:
        on_property = next(graph.objects(axiom, OWL.onProperty), None)
        some_values_from = next(graph.objects(axiom, OWL.someValuesFrom), None)
        dl = []
        if on_property is not None and some_values_from is not None:
            dl.append(f" ∃{format(graph, on_property)}.{render_description_logic_string(graph, some_values_from)} ")
        if dl:
            return f" ⊓ ".join(dl)
        # else:
        return format(graph, axiom)

    # owl:intersectionOf (RDF list)
    collection = next(graph.objects(axiom, OWL.intersectionOf), None)
    if collection is not None:
        items = []
        while collection and collection != RDF.nil:
            head = next(graph.objects(collection, RDF.first), None)
            if head is not None:
                items.append(render_description_logic_string(graph, head))
            collection = next(graph.objects(collection, RDF.rest), None)
        return " ⊓ ".join(items)

    # fallback, anonymous expression:
    return format(graph, axiom)


def build_axiom_verb_json(graph: Graph, nodes, subclass_edges, equivalence_edges):
    """builds a handy JSON data structure to hold axiom definitions/verbalisatins for (later) RAG"""
    parents_map = map_antecedents(subclass_edges)
    equivalents_map = map_equivalence(equivalence_edges)
    named_classes = identify_named_nodes(nodes)
    json_struct: dict[str, dict] = {}
    # populate json_struct
    for iri_node in named_classes:
        iri_str = str(iri_node)
        node_label = format(graph, iri_node)
        # get every named parent class (relative to this class) and materialise its label
        named_parents = identify_named_nodes([antecedent for antecedent in parents_map.get(iri_node, set())])
        subsumed_by_xs = [format(graph, antecedent) for antecedent in named_parents]
        # get every equiv class and materialise its verbalised class expression
        equivalents = [equiv for equiv in equivalents_map.get(iri_node, set()) if equiv != iri_node]
        equivalent_to_xs = [render_verbalisation(graph, equiv) for equiv in equivalents]
        dl_equivalent_to_xs = [render_description_logic_string(graph, equiv) for equiv in equivalents]
        # DL 
        # verbalisations
        verbal_subclass = [f"is a type of {name}" for name in subsumed_by_xs]
        verbal_equiv = [f"defined as {expr}" for expr in equivalent_to_xs]
        dl_subclass = [f"⊑ {name}" for name in subsumed_by_xs]
        dl_equiv = [f"≡ {expr}" for expr in dl_equivalent_to_xs]
        # build data structure ready for downstream RAG
        json_struct[iri_str] = {
            "label": node_label,
            "subclass_of": subsumed_by_xs,
            "equivalent_to": equivalent_to_xs,
            "verbalization": {
                "subclass_of": verbal_subclass,
                "equivalent_to": verbal_equiv,
            },
            "dl_subclass_of": dl_subclass,
            "dl_equivalent_to": dl_equiv
        }
    # finally:
    return json_struct


def main():
    parser=argparse.ArgumentParser(
        description="Run a BFS over O, verbalise 'C sqsubseteq D' and 'C equiv D' axioms"
    )
    parser.add_argument(
        "--input", 
        help="Path to RDF/OWL or TTL file"
    )
    parser.add_argument(
        "--output",
        help="Path to output JSON file"
    )
    args=parser.parse_args()

    print("Loading graph...")
    graph = (Graph()).parse(args.input)

    print("Walking graph to collect participant nodes...")
    nodes, subclass_axioms, equiv_axioms = classify_nodes(graph)

    print("Building verbalisation structure for export...")
    catalog = build_axiom_verb_json(graph, nodes, subclass_axioms, equiv_axioms)

    print(f"Writing verbalisation JSON to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # # #
    # OLD VERBALISATION CODE
    # # #

    # print("Building map of parent -> child, and child -> parent relationships...")
    # parents = map_descendants(subclass_axioms)
    # children = map_antecedents(subclass_axioms)

    # print("Mapping all participating equivalence relations...")
    # equivs  = map_equivalence(equiv_axioms)

    # print("Searching graph for SNOMED CT concept (top level concept, besides owl:Thing)...")
    # root_node, root_nodes, roots = None, None, None
    # try:
    #     root_node = [get_snomed_top_level_concept(graph)]
    # except KeyError:
    #     print("Unable to find SNOMED CT concept in graph, falling back to finding parents with an outdegree of zero...")
    #     root_nodes = identify_root_nodes(nodes, parents)
    # finally:
    #     roots = root_node if root_node is not None else root_nodes
    
    # print("Running BFS and verbalising...")
    # bfs_and_verbalise(graph, roots, parents, children, equivs, max_depth=None)

    print("DONE!")

if __name__=="__main__":
    main()