from typing import Any
from logging import log
import guesser
import buzzer
import json


class Proctor: 
    def __init__ (self, model: guesser, buzz: buzzer.Buzzer):
        self.question = None   
        self.answer = None
        self.tot_len = None
        self.guesser = model
        self.buzzer = buzz
        self.cur_len = 0
        damn = []


    def score(self, confidence: float, guess: str) -> float:
        adj_conf = (1 - (self.cur_len + 1 / self.tot_len + 1)) * confidence
        
        if guess == self.answer:
            return adj_conf
        else:
            return -adj_conf
    
    def guess(self) -> tuple[str, float]:
        if self.cur_len > self.tot_len: 
            self.cur_len = self.tot_len
        cur = " ".join(self.question[:(self.cur_len + 1)])        
        ans = self.guesser(cur, 1)
        guess = ans[0]["answer"]
        conf = ans[0]["confidence"]
        return guess, conf


    def new_question(self, question: dict) -> None:
        self.question = question["full_question"].split(" ")
        self.answer = question["raw_answer"]
        self.tot_len = len(self.question)
        self.cur_len = 10
    

    def __call__(self, question: str, freq: int, thresh: int) -> tuple[int, str]:
        self.new_question(question)
        while self.cur_len < self.tot_len: 
            guess, conf = self.guess()            
            if conf >= thresh: 
                return self.score(conf, guess), guess
            self.cur_len += freq
        guess, conf = self.guess()            
        return self.score(conf, guess), guess



