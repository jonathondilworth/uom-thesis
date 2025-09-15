import argparse
import json
import re
from collections import deque, defaultdict
from typing import Dict, Set, List, Tuple

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDFS, OWL, SKOS
from rdflib.query import ResultRow



class DisjointSetUnion:    
    
    def __init__(self):
        self.parent = {}

    def find(self, x: URIRef):
        parent = self.parent.get(x, x)
        if parent != x:
            parent = self.find(parent)
            self.parent[x] = parent
        else:
            self.parent.setdefault(x, x)
        return parent

    def union(self, a: URIRef, b: URIRef):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a



def build_label_xs(g: Graph, subj_node: URIRef, pred):
    xs = []
    for label_lit in g.objects(subj_node, pred):
        if isinstance(label_lit, Literal):
            xs.append(str(label_lit))
    return xs



def labels_dict(g: Graph, node: URIRef):
    labs = {}

    rdfs = build_label_xs(g, node, RDFS.label)
    if rdfs:
        labs['rdfs:label'] = rdfs[0]

    pref = build_label_xs(g, node, SKOS.prefLabel)
    if pref:
        labs['skos:prefLabel'] = pref[0]

    alt  = build_label_xs(g, node, SKOS.altLabel)
    if alt:
        labs['skos:altLabels'] = alt

    return labs



def snomed_numeric_key(iri: URIRef):
    larger_than_snomed_iri_id = 2**63 - 1
    iri_string = str(iri)
    matched = re.search(r'(\d+)$', iri_string)
    if matched:
        return (int(matched.group(1)), iri_string)
    # else:
    return (larger_than_snomed_iri_id, iri_string)



def build_equivalence_maps(g: Graph):
    """
    member2canon, map each named class IRI to its root 
    canon2members, map each root to the set of member IRIs in its equivalence class
    """
    dsu = DisjointSetUnion()

    for s, _, o in g.triples((None, OWL.equivalentClass, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            dsu.union(s, o)

    # collect all named rdfs:subClassOf nodes
    subclass_subjects = set(s for s, _, _ in g.triples((None, RDFS.subClassOf, None)) if isinstance(s, URIRef))
    subclass_objects  = set(o for _, _, o in g.triples((None, RDFS.subClassOf, None)) if isinstance(o, URIRef))
    named_taxonomy_nodes = subclass_subjects | subclass_objects

    # group by root
    groups = defaultdict(set)
    for node in named_taxonomy_nodes:
        groups[dsu.find(node)].add(node)

    member2canon  = {}
    canon2members = {}

    for root, members in groups.items():
        canon = min(members, key=snomed_numeric_key)  # smallest SNOMED ID
        canon2members[canon] = set(members)
        for member in members:
            member2canon[member] = canon

    # ensure identity mapping for uri refs seen elsewhere in the graph
    all_urirefs = {t for t in g.subjects()} | {t for t in g.objects()}
    for node in all_urirefs:
        if isinstance(node, URIRef) and node not in member2canon:
            member2canon[node] = node
            canon2members.setdefault(node, {node})

    return member2canon, canon2members



def direct_parents_simple(g: Graph, eq_members: Set[URIRef], member2canon: Dict[URIRef, URIRef]):
    """assumes the input file already has only direct edges"""
    parents = set()
    for child in eq_members:
        for parent in g.objects(child, RDFS.subClassOf):
            if isinstance(parent, URIRef):
                parents.add(member2canon.get(parent, parent))
    return parents



def direct_parents_sparql(g: Graph, eq_members: Set[URIRef], member2canon: Dict[URIRef, URIRef]):
    """SPARQL filter to ensure direct parents (even if redundant edges exist)"""
    parents = set()
    values = " ".join(f"<{m}>" for m in eq_members)
    query = f"""
    PREFIX rdfs: <{RDFS}>
    SELECT DISTINCT ?p WHERE {{
      VALUES ?c {{ {values} }}
      ?c rdfs:subClassOf ?p .
      FILTER(isIRI(?p))
      FILTER NOT EXISTS {{
        ?c rdfs:subClassOf ?m .
        FILTER(isIRI(?m) && ?m != ?p)
        ?m rdfs:subClassOf+ ?p .
      }}
    }}
    """
    for row in g.query(query):
        parent: ResultRow = row[0] # type: ignore
        if isinstance(parent, URIRef):
            parents.add(member2canon.get(parent, parent))

    return parents



# BFS over Hasse edges / enforced-direct

def bfs_ancestors(g: Graph, target: URIRef, member2canon: Dict[URIRef, URIRef], canon2members: Dict[URIRef, Set[URIRef]], enforce_direct: bool, max_depth: int | None = None):
    """find depth map over canonical IRIs (dist[canon] = hop distance, 0 at target); depth is defined on the Hasse diagram"""
    target_canon = member2canon.get(target, target)
    dist: Dict[URIRef, int] = {target_canon: 0}
    Q = deque([target_canon])

    while Q:
        u = Q.popleft()
        if max_depth is not None and dist[u] >= max_depth:
            continue

        # choose parent retriever
        if enforce_direct:
            parents = direct_parents_sparql(g, canon2members[u], member2canon)
        else:
            parents = direct_parents_simple(g, canon2members[u], member2canon)

        for v in parents:
            if v not in dist:
                dist[v] = dist[u] + 1
                Q.append(v)
    
    return dist



def anonymous_restrictions(g: Graph, eq_members: Set[URIRef]) -> List[Dict[str, object]]:
    out = []
    for c in eq_members:
        for obj in g.objects(c, RDFS.subClassOf):
            if isinstance(obj, BNode):
                onp = g.value(obj, OWL.onProperty)
                fillr = g.value(obj, OWL.someValuesFrom)
                if isinstance(onp, URIRef) and isinstance(fillr, URIRef):
                    entry = {
                        "onProperty": str(onp),
                        "filler": str(fillr),
                        "labels": {
                            "property": labels_dict(g, onp),
                            "filler":   labels_dict(g, fillr),
                        },
                    }
                    out.append(entry)
    return out



def named_equivalents(g: Graph, target: URIRef, member2canon: Dict[URIRef, URIRef], canon2members: Dict[URIRef, Set[URIRef]]):
    target_canon = member2canon.get(target, target)
    
    eqs = []
    for member in sorted(canon2members[target_canon], key=lambda u: snomed_numeric_key(u)):
        if member != target_canon:
            eqs.append(member)
    
    result_eqs_xs = []
    for member in eqs:
        result_eqs_xs.append({
            "iri": str(member),
            **labels_dict(g, member)
        })

    return result_eqs_xs



def assemble_json(g: Graph, target: URIRef, dist: Dict[URIRef, int], member2canon: Dict[URIRef, URIRef], canon2members: Dict[URIRef, Set[URIRef]], include_anonymous: bool, include_equivalents: bool):

    target_canon = member2canon.get(target, target)

    target_block = {"iri": str(target_canon), "depth": 0}
    target_block.update(labels_dict(g, target_canon))

    parents = []
    ancestors = []
    for v, d in sorted(dist.items(), key=lambda kv: (kv[1], snomed_numeric_key(kv[0]))):
        if v == target_canon:
            continue
        entry = {"iri": str(v), "depth": d}
        entry.update(labels_dict(g, v))
        if d == 1:
            parents.append(entry)
        elif d >= 2:
            ancestors.append(entry)

    out = {
        "target_entity": target_block,
        "parent_entities": parents,
        "ancestors": ancestors,
    }

    if include_anonymous:
        out["anonymous_axioms"] = anonymous_restrictions(g, canon2members[target_canon])
    else:
        out["anonymous_axioms"] = []

    if include_equivalents:
        out["equivalent_classes"] = named_equivalents(g, target_canon, member2canon, canon2members)
    else:
        out["equivalent_classes"] = []

    return out



def main():
    parser = argparse.ArgumentParser(
        description="SNOMED CT: entailed parents/ancestors with Hasse distances."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="path to OWL or TTL inferred view"
    )
    parser.add_argument(
        "--iri",
        required=True,
        help="full IRI of the SNOMED concept"
    )
    parser.add_argument(
        "--json-out",
        required=False,
        help="path to write JSON"
    )
    parser.add_argument(
        "--enforce-direct",
        action="store_true",
        help="derive direct parents by SPARQL (slower)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="optional max ancestor depth."
    )
    parser.add_argument(
        "--include-anonymous",
        action="store_true",
        help="include anonymous restrictions (annotations only)"
    )
    parser.add_argument(
        "--include-equivalents",
        action="store_true",
        help="include named equivalent classes of the target"
    )
    args = parser.parse_args()

    g = Graph()
    print(f"loading {args.input} ... ")
    g.parse(args.input)
    print(f"loaded graph with triples: {len(g)}")

    target = URIRef(args.iri)

    member2canon, canon2members = build_equivalence_maps(g)

    # compute Hasse-hop depths
    dist = bfs_ancestors(
        g=g,
        target=target,
        member2canon=member2canon,
        canon2members=canon2members,
        enforce_direct=args.enforce_direct,
        max_depth=args.max_depth
    )

    # prepare JSON for output
    output = assemble_json(
        g=g,
        target=target,
        dist=dist,
        member2canon=member2canon,
        canon2members=canon2members,
        include_anonymous=args.include_anonymous,
        include_equivalents=args.include_equivalents
    )

    # write results
    json_dump = json.dumps(
        output,
        ensure_ascii=False,
        indent=2
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(json_dump + "\n")
        print(f"written {args.json_out} to disk")
    else:
        print(json_dump)


if __name__ == "__main__":
    main()