from thesis.utils import data_utils

def test_strip_parens_basic():
    assert data_utils.strip_parens("example string (with parens)") == "example string "
    assert data_utils.strip_parens("Type 2 diabetes (disorder)") == "Type 2 diabetes "

def test_naive_tokenise():
    # naive_tokenise
    seqs = ["example list of sentences", "another sentence"]
    assert data_utils.naive_tokenise(seqs[0]) == ["example", "list", "of", "sentences"]
    # parallel_tokenise
    toks = data_utils.parallel_tokenise(seqs, workers=1, chunksize=1)
    assert toks == [["example","list","of","sentences"], ["another","sentence"]]

def test_data_mapping_fns():
    example_dataset = {
        "datasetA": {
            "q1": {
                "question": "Q1?", 
                "options": {
                    "A":"x",
                    "B":"y"
                }, 
                "answer":"A"
            },
            "q2": {
                "question": "Q2?", 
                "options": {
                    "A":"x2",
                    "B":"y2"
                }, 
                "answer":"B"
            },
        },
        "datasetB": {
            "x1": {
                "question": "QB1?", 
                "options": {
                    "A":"a",
                    "B":"b"
                }, 
                "answer":"B"
            },
        },
    }
    
    # get_dataset_question_mapping
    mappingA = data_utils.get_dataset_question_mapping("datasetA", example_dataset)
    assert mappingA == {0:"q1", 1:"q2"}
    
    # get_question_str
    assert data_utils.get_question_str("q2", example_dataset["datasetA"]) == "Q2?"
    
    # get_question_opts
    assert data_utils.get_question_opts("x1", example_dataset["datasetB"]) == {"A":"a","B":"b"}
    
    # get_question_ans
    assert data_utils.get_question_ans("x1", example_dataset["datasetB"]) == "B"
    
    # get_dataset_names
    assert sorted(data_utils.get_dataset_names(example_dataset)) == ["datasetA","datasetB"]
    
    # get_question_count
    assert data_utils.get_question_count("datasetA", example_dataset) == 2
