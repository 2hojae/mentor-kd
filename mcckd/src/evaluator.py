import re

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.generator import Generator
from src.model import DistributedModule
from src.tokenizer import Tokenizer
from src.utils import json_dump


class DistributedEvaluator:
    ANSWER_TYPE_NUMERIC = "NUMERIC"
    ANSWER_TYPE_CHOICE = "CHOICE"
    ANSWER_TYPE_OPEN_ENDED = "OPEN_ENDED"
    NUMERIC_TASKS = ['GSM8K', 'SVAMP', 'ASDiv']
    CHOICE_TASKS = ['CSQA', 'date_understanding', 'tracking_shuffled_objects']
    OPEN_ENDED_TASKS = ['last_letter_concatenation']
    YES_NO_TASKS = ['strategy_qa']

    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.generator = Generator(self.model, tokenizer)
        self.tokenizer = tokenizer

    def _get_answer_type(self, task: str):
        if task in self.NUMERIC_TASKS:
            return self.ANSWER_TYPE_NUMERIC
        elif task in self.CHOICE_TASKS:
            return self.ANSWER_TYPE_CHOICE
        elif task in self.OPEN_ENDED_TASKS:
            return self.ANSWER_TYPE_OPEN_ENDED
        elif task in self.YES_NO_TASKS:
            return self.YES_NO_TASKS
        else:
            raise ValueError(
                f"Unrecognized task `{task}`; "
                f"Currently supported {self.NUMERIC_TASKS+self.CHOICE_TASKS}")

    def evaluate(self,
                 task: str,
                 label_file,
                 output_file,
                 batch_size,
                 max_seq_len,
                 temperature: float,
                 top_p: float):
        print("Evaluating.........")
        answer_type = self._get_answer_type(task)

        def extract_predict(_output, _answer_type) -> list:
            _output = _output.strip()
            # only count for the last line
            # endline = _output.split('\n')[-1]
            endline = _output.split("Therefore, the answer is")[-1]
            if _answer_type == self.ANSWER_TYPE_NUMERIC:
                matches = re.findall(r'(-?\d+)(,?\d+)?(\.\d+)?', endline)
            elif _answer_type == self.ANSWER_TYPE_CHOICE:
                matches = re.findall(r'[A-G]', endline)
            elif _answer_type == self.ANSWER_TYPE_OPEN_ENDED:
                matches = re.sub("\"|\'|\n|\.|\s", "", endline)
                matches = [matches]
            elif _answer_type == self.YES_NO_TASKS:
                matches = endline.lower()
                matches = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", matches)
                matches = matches.split(" ")
                matches = [i for i in matches if i in ("yes", "no")]
                
            else:
                raise ValueError(
                    f"Unrecognized answer type: `{_answer_type}`; "
                    f"Currently supported [`{self.ANSWER_TYPE_NUMERIC}`|`{self.ANSWER_TYPE_CHOICE}`]")
            predict = []
            for match in matches:
                predict.append(''.join(match))
            return predict

        def evaluate_accuracy(_datalist, _answer_type):
            _hit = 0
            for _data in _datalist:
                if _answer_type in [self.ANSWER_TYPE_NUMERIC, self.ANSWER_TYPE_CHOICE, self.ANSWER_TYPE_OPEN_ENDED, self.YES_NO_TASKS]:
                    if _answer_type == self.YES_NO_TASKS:
                        _data['label'] = _data['label'].lower()
                    if _data['label'] in _data['predict'][-1:]:
                        _hit += 1
                else:
                    raise ValueError(f"Unknown answer type: {_answer_type}")
            _accuracy = _hit / (len(_datalist) + 1e-7)
            return _accuracy


        tokenizer_kwargs = {"padding": "longest", "max_length": 512, "truncation": True, "return_tensors": "pt"}

        dataset = JsonDataset(label_file)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        datalist = []
        generation_kwargs = {"max_length": 512}
        for data in tqdm(data_loader):
            #results = self.generator.generate(
            #    prompts=data['instruction'],
            #    max_gen_len=max_seq_len,
            #    temperature=temperature,
            #    top_p=top_p)
            #print(data['instruction'])
            input_ids = self.tokenizer(data['instruction'], **tokenizer_kwargs)['input_ids']
            #print(input_ids, input_ids.shape)
            #print(input_ids)
            #print(self.model)
            results = self.model.generate(input_ids = input_ids.cuda(), max_length = 512).detach()
            #results = self.model.generate(input_ids.cuda()).detach()
            
            results = self.tokenizer.batch_decode(results, skip_special_tokens=True)
            
            for i, result in enumerate(results):
                
                
                ret = {
                    'instruction' : data['instruction'][i],
                    'output' : result,
                    'predict' : extract_predict(result, answer_type),
                    'label' : data['label'][i]
                }
                #print(ret)
                datalist.append(ret)
                '''   
                datalist.append(dict(
                    instruction=result['instruction'],
                    output=result['output'],
                    predict=extract_predict(result['output'], answer_type),
                    label=data['label'][i]))
                '''    
        accuracy = evaluate_accuracy(datalist, answer_type)
        print(f"Accuracy: {accuracy}")
        if output_file is not None:
            json_dump(datalist, f"{output_file}-{round(float(accuracy), 4)}.log")
        return accuracy

    def load(self, ckpt_dir):
        self.model.load(ckpt_dir)