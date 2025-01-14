from typing import List, Tuple
import torch as th
import nltk
import numpy as np
import pandas as pd
from proctor import Proctor
from buzzer import Buzzer
from guesser import Guesser
import json

class QBAM:

    def __init__(self, model: str = "BERT"):
        self.model = Guesser(model)
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
        question = "A speaker of one of this man's poems eats \"reality sandwiches\" and admits \"a naked lunch is natural to us.\" The speaker of another of this man's poems asserts that \"Death is that remedy all singers dream of\" as he walks in Greenwich Village thinking of his mother Naomi. The refrain \"I'm with you in Rockland\" appears in another poem by this author of \"Kaddish.\" That poem by this man begins, \"I saw the best (*) minds of my generation destroyed by madness.\" For 10 points, name this Beat poet who wrote \"Howl.\"" 
        answer= "Allen Ginsberg"
        question = {"text": question, "answer": answer}
        model = QBAM()

        score, guess = model(question)
        print(f"Prediction: {guess}, Score: {score}") 
        print(f"Answer: {answer}") 

    
