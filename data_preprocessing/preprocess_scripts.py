import os
import re
from numpy import character
import uuid
from regex import F
import json
from flair.data import Sentence
from flair.models import SequenceTagger
PER = "PER"
NUM_SENTENCES = 4




def write_N_to_file(input_path, output_path):
    
    res = []
    dir_list = os.listdir(input_path)
    for i,dir in enumerate(dir_list): 
        script_path = os.path.join(input_path, dir, 'ScriptsText')
        for script in os.listdir(script_path):

            script_file = os.path.join(script_path, script)
            parsed_file = os.path.join(input_path, dir, 'Parsed', script)
            output_n_path = os.path.join(output_path, dir, script)
            # print(parsed_path)
            try:
                with open(script_file) as f:
                    lines_script = f.read().splitlines()
                with open(parsed_file) as f:
                    lines_parsed = f.read().splitlines() #f.readlines()
                if len(lines_parsed) != len(lines_script):
                    with open("unsync_lines.txt", "a") as f:
                        f.write(script_file+"\n")
                for j,line in enumerate(lines_parsed):
                    if line == 'N' and j < len(lines_script):
                        res.append(re.sub("\s+", " ", lines_script[j])) 
                with open(output_n_path, "w") as f:
                    f.write(" ".join(res))

                res.clear()

            except:
                with open("not_found.txt", "a") as f:
                    f.write(parsed_file+"\n")

def get_names(file_path, tagger):
    with open(file_path) as f:
        s = f.read()
        # print(s)
        sentence = Sentence(s)

    tagger.predict(sentence)
    names = set()
    for entity in sentence.get_spans('ner'):
        if entity.get_labels()[0].value == PER and entity.get_labels()[0].score > 0.9:
            names.add(entity.text)

    sentences = s.split(".")
    result = {}
    for name in names:
        for idx, sentence_str in enumerate(sentences):
            if name in sentence_str:
                result[name] = ". ".join(sentences[idx:idx+NUM_SENTENCES])

    return names, result

def generate_csqa_json(name, context):
    yield {
    "id": str(uuid.uuid4()),
    "question": {
        "question_concept": "age",
        "choices": [
        {
            "label": "A",
            "text": "young"
        },
        {
            "label": "B",
            "text": "old"
        },
        {
            "label": "C",
            "text": "child"
        },
        {
            "label": "D",
            "text": "not specified"
        },
        {
            "label": "E",
            "text": "not specified"
        }
        ],
        "stem": context + f". What is the age of {name}"
    }
    }

    yield {
    "id": str(uuid.uuid4()),
    "question": {
        "question_concept": "gender",
        "choices": [
        {
            "label": "A",
            "text": "male"
        },
        {
            "label": "B",
            "text": "female"
        },
        {
            "label": "C",
            "text": "not specified"
        },
        {
            "label": "D",
            "text": "not specified"
        },
        {
            "label": "E",
            "text": "not specified"
        }
        ],
        "stem": context + f". What is the gender of {name}?"
        }
    }




if __name__ == '__main__':

    input_path = "./SAIL-team-spellcheck"
    output_path = "./script_scenes/lego-titan"
    # write_N_to_file(input_path, output_path)

    tagger = SequenceTagger.load("flair/ner-english")
    file_list = os.listdir(output_path)
    
    with open("/proj/vrundhas/qagnn/data/csqa/test_rand_split_no_answers.jsonl", "w") as f:
        for i,file in enumerate(file_list):
            names, contexts = get_names(os.path.join(output_path, file), tagger)
            for name in names:
                if name in contexts:
                    for generated in generate_csqa_json(name, contexts[name]):
                        f.write(json.dumps(generated))
                        f.write("\n")
