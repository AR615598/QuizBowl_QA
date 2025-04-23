from typing import List, Tuple
import torch as th
import numpy as np
import pandas as pd
from proctor import Proctor
from buzzer import Buzzer
from guesser import Guesser
import json
import datasets

class QBAM:

    def __init__(self, model: str = "BERT", checkpoint: str = "distilbert-base-cased-distilled-squad"):
        self.model = Guesser(model, checkpoint)
        self.buzzer = Buzzer()    
        self.proctor = Proctor(self.model, self.buzzer)


    def __call__(self, question: dict) -> Tuple[int, str]:
        self.proctor.new_question(question)
        return self.proctor(question, 1, 300)


    def batch_guess(self, question_text: List[str]) -> List[Tuple[int, str]]:
        guesses = []
        for question in question_text:
            guess = self(question)
            guesses.append(guess)
        return guesses
 
     # allow for a json file to be passed in and so the model can be trained on that data and make predictions
    def json_to_question(self, json_file: str) -> list[dict["text":str, "answer": str]]:
        with open(json_file) as read:
            data = json.load(read)
            for question in data:
                answer = question['answer']
                text = question['text']
        pass
    # pkl files are used to save the model
    def save(self):
        pass
    # pkl files are used to load the model
    def load(self):
        pass
    def evaluate(self, json_file: str) -> dict:
        pass

if __name__ == "__main__":
        ds = datasets.load_from_disk('../res/data/QANTA-IgnoreIMP')

        question = ds['guesstrain'][0]
        answer = question['answer']
        model = QBAM(checkpoint = "../res/models/optuna_IgnoreIMP")

        score, guess = model(question, 1)
        print(f"Question: {question['text']}")
        print(f"Prediction: {guess}, Score: {score}") 
        print(f"Answer: {answer}") 

    
