import re

from typing import List, Tuple, Dict

PREDICTION_PREFIXES = {
    None: None,
    "zs": None,
    "ft_natural": None,
    "ft_token": None,
    "fs_cot": "The answer is",
    "zs_cot": None,
    "ft_cot_natural": "Therefore, the answer is",
    "ft_cot_token": "-->",
}

class Evaluator:
    def __init__(self, dataset_key, task_type="ft_cot_token"):
        self.dataset_key = dataset_key
        self.prediction_prefix = PREDICTION_PREFIXES[task_type]

    def _extract_prediction_candidates(self, prediction: str) -> List[str]:
        """
        Extracts all potential answer predictions which satisfy the dataset's answer format from the
        prediction string
        """
        
        original_prediction = [prediction]
        
        if self.dataset_key in ("aqua", "commonsense_qa"):
            prediction = re.findall(r'[ABCDE]', prediction)
        elif self.dataset_key == "date_understanding":
            prediction = re.findall(r'[ABCDEF]', prediction)
        elif self.dataset_key in ("tracking_shuffled_objects"):
            prediction = re.findall(r'[ABC]', prediction)
        elif self.dataset_key in ("gsm8k", "addsub", "multiarith", "svamp", "single_eq", "asdiv"):
            prediction = prediction.replace(",", "")
            prediction = re.findall(r'-?\d+(?:\.\d+)?', prediction)
            if self.dataset_key in ("addsub", "svamp", "single_eq"):
                prediction = [float(s) for s in prediction]
        elif self.dataset_key in ("strategy_qa", "coin_flip"):
            prediction = prediction.lower()
            prediction = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", prediction)
            prediction = prediction.split(" ")
            prediction = [i for i in prediction if i in ("yes", "no")]
        elif self.dataset_key == "last_letter_concatenation":
            prediction = re.sub("\"|\'|\n|\.|\s", "", prediction)
            prediction = [prediction]
        else:
            raise ValueError("Invalid dataset: {}".format(self.dataset_key))
        
        if len(prediction) != 0:
            return prediction
        else:
            return original_prediction
    
    def cleanse_prediction(self, completion, return_all=False):
        if self.prediction_prefix is None:
            # If no prefix, use first candidate
            predictions = self._extract_prediction_candidates(completion)
            first = True
        else:
            index = completion.find(self.prediction_prefix)
            if index == -1:
                # If prefix not found, use *last* candidate
                predictions = self._extract_prediction_candidates(completion)
                first = False
            else:
                # If prefix found, use *first* candidate after prefix
                start_of_answer = index + len(self.prediction_prefix)
                predictions = self._extract_prediction_candidates(completion[start_of_answer:])
                first = True
        
        answer = None
        if predictions:
            answer = (predictions[0] if first else predictions[-1])
        
        return (answer, predictions) if return_all else answer
    
    def cleanse_answer(self, answer: str) -> str:
        if self.dataset_key in ["gsm8k", "addsub", "multiarith", "svamp", "single_eq", "asdiv"]:
            answer = answer.replace(",", "")
        if self.dataset_key == "strategy_qa":
            answer = answer.lower()
        if self.dataset_key in ["addsub", "svamp", "single_eq"]:
            answer = float(answer)
        
        # Added
        elif self.dataset_key == "commonsense_qa":
            answer = answer.split()[0][0]
        
        return answer
    
    
    def _compare_prediction_and_answer(self, prediction, answer) -> bool:
        if self.dataset_key in ("addsub", "svamp", "single_eq"):
            if type(prediction) is float or type(prediction) is int:
                return prediction is not None and abs(prediction - answer) <= 1e-6
            else:
                return False
        
        elif self.dataset_key in ("last_letter_concatenation"):
            return prediction is not None and prediction.lower() == answer.lower()
        
        else:
            return prediction is not None and prediction == answer
        
    def evaluate_single_instance(self, prediction, answer) -> bool:
        cleanse_prediction = self.cleanse_prediction(prediction)
        cleanse_answer = self.cleanse_answer(answer)
        evaluation = self._compare_prediction_and_answer(cleanse_prediction, cleanse_answer)
        return evaluation