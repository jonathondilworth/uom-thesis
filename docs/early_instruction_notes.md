# PROCESSING SNOMED DATA: AXIOMS & EL ONTOLOGY

**prerequisites**

1. JDK: `sudo apt install openjdk-17-jdk`
2. Maven: `sudo apt install maven`
3. ROBOT: `git clone git@github.com:ontodev/robot.git` *see: (https://robot.obolibrary.org/)[https://robot.obolibrary.org/]*
4. snomed-owl-toolkit: `git clone git@github.com:IHTSDO/snomed-owl-toolkit.git` and:
    * `cd snomed-owl-toolkit`
    * `wget https://github.com/IHTSDO/snomed-owl-toolkit/releases/download/5.3.0/snomed-owl-toolkit-5.3.0-executable.jar`
5. HierarchyTransformer: `git clone git@github.com:KRR-Oxford/HierarchyTransformers.git`
6. OntologyTransformer: `git clone <URL>`

**steps: AXIOM VERBALISATION**

1. download SNOMED CT latest release

2. place SNOMED CT zip file in data directory
3. `cd robot`
4. `mvn clean package` or `mvn clean test` or `mvn clean verify`
5. Run (replacing the directory structure, etc with your specific requirements):

*(replace with your IRI and VERSION)*

```
java -Xms4g --add-opens java.base/java.lang=ALL-UNNAMED -jar snomed-owl-toolkit/snomed-owl-toolkit-5.3.0-executable.jar \
  -rf2-to-owl \
  -rf2-snapshot-archives ./data/SnomedCT_InternationalRF2_PRODUCTION_20250701T120000Z.zip \
  -iri http://snomed.info/sct/900000000000207008 \
  -version 20250701 
```

For AUG release:

```
java -Xms4g --add-opens java.base/java.lang=ALL-UNNAMED -jar snomed-owl-toolkit/snomed-owl-toolkit-5.3.0-executable.jar \
  -rf2-to-owl \
  -rf2-snapshot-archives ./data/SnomedCT_InternationalRF2_PRODUCTION_20250701T120000Z.zip \
  -iri http://snomed.info/sct/900000000000208008 \
  -version 20250801 
```

6. `mv ./ontology-*.owl ./data/snomedct-international.owl`

7. There is a bug in the `snomed-owl-toolkit` script that generates invalid ontology IRIs (version IRI and ontology IRI) when applying a reasoner over the resulting `.owl` file directly (this also occurs in protege when trying to export the inferred view), so we need to use a different format...

8. Run a reasoner over snomed (ELK for EL ontology):

```
# increase the amount of memory available to the JVM
export JAVA_TOOL_OPTIONS="-Xmx32g"

alias robot="./robot/bin/robot"

VERY_VERBOSE=1

# base ontology: asserted UNION inferred
#   => arbitrary length assertions & inference
robot reason \
  --input ./data/snomedct-international.owl \
  --reasoner ELK \
  --equivalent-classes-allowed all \
  --axiom-generators "SubClass EquivalentClass" \
  --exclude-duplicate-axioms true \
  --exclude-tautologies structural \
  --output ./data/base-snomed-reasoned.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# atomic taxonomy: asserted UNION inferred
#   => $A$, $A \equiv B$ and $A \sqsubseteq B$, where $\{A, B\}$ are atomic
# -- STEPS --
# 1. use base-snomed-reasoned
# 2. remove anonymous class expressions of the form:
#        $C_{complex} \leftarrow \{A \sqcap B$, $\exists r.C \sqsubseteq D\}$
./robot/bin/robot filter \
  --input  ./data/base-snomed-reasoned.ttl \
  --select "complement anonymous" \
  --axioms "subclass equivalent declaration annotation" \
  --output ./data/snomedct-atomic-tax.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# atomic taxonomy \wo transitive closure: 
#   => trim := $f : tax_{atomic} \rightarrow (\textit{asserted} \cup \textit{inferred})$
robot reduce \
  --reasoner ELK \
  --input ./data/snomedct-atomic-tax.ttl \
  --output ./data/snomedct-atomic-tax-wo-transitive-closure.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# inferred atomic taxonomy ONLY
robot reason \
  --input ./data/snomedct-international.owl \
  --reasoner ELK \
  --equivalent-classes-allowed all \
  --create-new-ontology true \
  --axiom-generators "SubClass EquivalentClass" \
  --exclude-tautologies structural \
  --output ./data/snomedct-atomic-tax-inferred.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# inferred atomic taxonomy ONLY \wo transitive closure:
robot reduce \
  --reasoner ELK \
  --input ./data/snomedct-atomic-tax-inferred.ttl \
  --output ./data/snomedct-atomic-tax-inferred-wo-transitive-closure.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# EL ontology:
robot reason \
  --input ./data/snomedct-international.owl \
  --reasoner ELK \
  --equivalent-classes-allowed all \
  --exclude-tautologies structural \
  --exclude-duplicate-axioms true \
  --annotate-inferred-axioms false \
  --remove-redundant-subclass-axioms true \
  --output ./data/snomedct-el-full.ttl \
  ${VERY_VERBOSE:+-vvv} \
  -stdout

# VERIFY EL ONTOLOGY
robot validate-profile --profile EL --input ./data/snomedct-el-full.ttl

# PREP FOR CONVERSION

mkdir ./data/ofn
mkdir ./data/owx
mkdir ./data/owl

# CONVERT ATOMIC TAX TO VARIOUS FORMATS

./robot/bin/robot convert --input ./data/snomedct-atomic-tax.ttl \
  --format ofn \
  --output ./data/ofn/snomedct-atomic-tax.ofn

./robot/bin/robot convert --input ./data/snomedct-atomic-tax.ttl \
  --format owx \
  --output ./data/owx/snomedct-atomic-tax.owx

# throws an exception and fails to write
./robot/bin/robot convert --input ./data/snomedct-atomic-tax.ttl \
  --format owx \
  --output ./data/owl/snomedct-atomic-tax.owl

# CONVERT ATOMIC TAX WITHOUT TRANSITIVE CLOSURE TO VARIOUS FORMATS

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-wo-transitive-closure.ttl \
  --format ofn \
  --output ./data/ofn/snomedct-atomic-tax-wo-transitive-closure.ofn

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-wo-transitive-closure.ttl \
  --format owx \
  --output ./data/owx/snomedct-atomic-tax-wo-transitive-closure.owx

# throws an exception and fails to write  <-- see 3
./robot/bin/robot convert --input ./data/snomedct-atomic-tax-wo-transitive-closure.ttl \
  --format owx \
  --output ./data/owl/snomedct-atomic-tax-wo-transitive-closure.owl

# CONVERT ATOMIC TAX INFERRED TO VARIOUS FORMATS

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred.ttl \
  --format ofn \
  --output ./data/ofn/snomedct-atomic-tax-inferred.ofn

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred.ttl \
  --format owx \
  --output ./data/owx/snomedct-atomic-tax-inferred.owx

# throws an exception and fails to write  <-- see 3
./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred.ttl \
  --format owx \
  --output ./data/owl/snomedct-atomic-tax-inferred.owl

# CONVERT ATOMIC TAX INFERRED WITHOUT TRANSITIVE CLOSURE TO VARIOUS FORMATS

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred-wo-transitive-closure.ttl \
  --format ofn \
  --output ./data/ofn/snomedct-atomic-tax-inferred-wo-transitive-closure.ofn

./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred-wo-transitive-closure.ttl \
  --format owx \
  --output ./data/owx/snomedct-atomic-tax-inferred-wo-transitive-closure.owx

# throws an exception and fails to write  <-- see 3
./robot/bin/robot convert --input ./data/snomedct-atomic-tax-inferred-wo-transitive-closure.ttl \
  --format owx \
  --output ./data/owl/snomedct-atomic-tax-inferred-wo-transitive-closure.owl

# CONVERT EL ONTOLOGY TO VARIOUS FORMATS

./robot/bin/robot convert --input ./data/snomedct-el-full.ttl \
  --format ofn \
  --output ./data/ofn/snomedct-el-full.ofn

./robot/bin/robot convert --input ./data/snomedct-el-full.ttl \
  --format owx \
  --output ./data/owx/snomedct-el-full.owx

# throws an exception and fails to write <-- see 3
./robot/bin/robot convert --input ./data/snomedct-el-full.ttl \
  --format owx \
  --output ./data/owl/snomedct-el-full.owl
```

9. If your enviornment doesn't already support `DeepOnto`, install it:

```
pip install deeponto
```

10. Run the test script under `./scripts/deeponto_test.py`:

```python
# imports
from deeponto.onto import Ontology
from deeponto.onto.taxonomy import OntologyTaxonomy

# onto = Ontology("../data/snomedct-classified.ttl",  reasoner_type="struct")   # ~1gb
onto = Ontology("../data/snomedct-classified.ofn", reasoner_type="struct")      # ~300mb
# onto = Ontology("../data/snomedct-classified.owx", reasoner_type="struct")    # ~700mb
# onto = Ontology("../data/snomedct-classified.owl", reasoner_type="struct")    # ~700mb

tax = OntologyTaxonomy(onto, reasoner_type="struct")

print(f"{len(tax.nodes):,} concepts, {len(tax.edges):,} IS-A edges loaded")

example_iri = # <include example IRI here>

print("Parents:", tax.get_parents(example_iri, apply_transitivity=False))
print("Ancestors:", tax.get_parents(example_iri, apply_transitivity=True))
print("Direct children:", tax.get_children(example_iri))
# networkx
g = tax.graph
print("Out-degree:", g.out_degree(example_iri))
```

11. Consider increasing the amount of available memory for the JVM:

`export JAVA_TOOL_OPTIONS="-Xmx32g"`

12. Produce Taxonomy Verbalisations:

```python
from deeponto.onto import Ontology, OntologyVerbaliser
from deeponto.onto.taxonomy import OntologyTaxonomy

from HierarchyTransformers.src.hierarchy_transformers.datasets.construct import HierarchyDatasetConstructor

onto = Ontology("./data/snomedct-classified.ofn", reasoner_type="struct")

# construct dataset example

snomed_tax = OntologyTaxonomy(onto, reasoner_type="struct")

hit_data_constructor = HierarchyDatasetConstructor(snomed_tax)

verbaliser = OntologyVerbaliser(onto, keep_iri=True, apply_lowercasing=True)

subsumption_axioms = onto.get_subsumption_axioms(entity_type="Classes")

verbalisation = verbaliser.verbalise_class_expression(subsumption_axioms)

print(type(verbalisation))

import json

with open('snomed_subsumption_axiom_verbalisations.json', 'w', encoding='utf-8') as f:
    json.dump(verbalisation, f, ensure_ascii=False, indent=4)
```
