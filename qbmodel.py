from typing import List, Tuple
import torch as th
import nltk
import numpy as np
import pandas as pd
import contextGenerator
import guesser
import json

class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.

        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """


        # For pure retrieval
        # guesser = pyserini_guesser.pyserini_guesser('data/qanta.train.json', False)
        # self.guesser = guesser

        # For bert_guess
        model = guesser.BertGuess(True)
        self.model = model

        

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 

        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]

        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        # For pure retrieval
        # guesses = []
        # for question in question_text:
        #     guess = self.guesser(question, 1)[0]
        #     score = guess['confidence']
        #     pred = guess['id']
        #     guesses.append([pred, score])
        # return guesses
    
        # For bert_guess
        guesses = []
        for question in question_text:
            guess = self.model(question, 1, 'simple')[0]
            guesses.append([guess['answer'], guess['confidence']])
        return guesses
