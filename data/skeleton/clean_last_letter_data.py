import os, sys
sys.path.append(os.getcwd())

import json

train_file = "data/skeleton/last_letter_concatenation_train.json"
test_file = "data/skeleton/last_letter_concatenation_test.json"

with open(train_file) as f_train, open(test_file) as f_test:
    train_data = json.load(f_train)
    test_data = json.load(f_test)

for dataset_type in ["train", "test"]:
    file_name = f"data/skeleton/last_letter_concatenation_{dataset_type}.json"
    with open(file_name) as f:
        data = json.load(f)

    incorrect_count = 0
    refined_data = []
    for s in data:
        instance = dict()
        q = s['input']
        answer_to_json = s['answer']

        words_to_concat = q.split('"')[1]
        words = words_to_concat.split(" ")
        words = [w.strip() for w in words if w]
        
        real_answer = "".join([c[-1] for c in words])
        
        if answer_to_json != real_answer:
            answer_to_json = real_answer
            incorrect_count += 1

        instance["input"] = q
        instance["answer"] = answer_to_json
        refined_data.append(instance)
    
    with open(file_name, "w") as f_refine:
        json.dump(refined_data, f_refine, indent=4)
    
    print(f"{dataset_type} || {incorrect_count}")
        
    