from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from utils import clean_text
from contextGenerator import LuceneRetrieval
import torch
import torch.nn.functional as f
import spacy
import utils
import numpy as np
import itertools
class Guesser:
    def __init__(self, model: str, checkpoint: str = "distilbert-base-cased-distilled-squad"):
        if model == "BERT":
            self.model = BertGuess(checkpoint)
        elif model == "RET":
            self.model = RETGuess() 
        else: 
            raise ValueError("Invalid model type, model must be either BERT or RET")
            
    
    def  __call__(self,
            question: str,
            num_guesses: int):
        
        return self.model(
            question,
            num_guesses
        )
        
class RETGuess():
    def __init__(self):
        self.context_model = LuceneRetrieval()
        self.nlp = spacy.load("en_core_web_lg")

    def extract_title(self, context): 
        tokens = self.nlp(context)
        ents = [x.text for x in tokens.ents]
        if len(tokens.ents) > 0: 
            return ents[0]
        else:
            return "NAN"


    def  __call__(self,
                  question: str,
                  num_guesses: int
                  ) -> any:
        context = self.context_model(question, 1)[0]
        cont, conf = context['contents'], context['confidence']
        cont = (" ").join(cont.split(" ")[:10])
        y_hat = self.extract_title(cont)
        x = [{"answer": y_hat, "confidence": conf}]
        return [{"answer": y_hat, "confidence": conf}]
    
    
class BertGuess():
    def __init__(self, checkpoint: str = "distilbert-base-cased-distilled-squad"):
        self.checkpoint = checkpoint

        try:
            self.context_model = LuceneRetrieval()
        except Exception as e:
            print(f"Error loading Lucene: {e}")
            exit(1)
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.checkpoint)
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.checkpoint)
        except Exception as e:
            print(f"Error loading BERT: {e}")
            exit(1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
    def valid_spans(self, st_pos, ed_pos, k):
        top_k_idx_start = np.argpartition(st_pos, range(-k, 0, 1), None)[-k:]
        top_k_idx_end = np.argpartition(ed_pos, range(-k, 0, 1), None)[-k:]
        zeroes = None
        if 0 in top_k_idx_start or 0 in top_k_idx_end:
            top_k_idx_start = np.delete(top_k_idx_start, np.where(top_k_idx_start == 0))
            top_k_idx_end = np.delete(top_k_idx_end, np.where(top_k_idx_end == 0))
            zeroes = [(0,0)]
            
        try:
            pair_matrix = list(itertools.product(top_k_idx_start, top_k_idx_end)) + zeroes
        except:
             pair_matrix = list(itertools.product(top_k_idx_start, top_k_idx_end))
             
        for x in pair_matrix: 
            st, ed = x
            if st > ed: 
                pair_matrix.remove(x)
        score_matrix = np.full(len(pair_matrix), np.NINF)

        for i, pair in enumerate(pair_matrix):
            start, end = pair
            score_matrix[i] = st_pos[0,start] + ed_pos[0,end]
        
        lst = ([pair_matrix[x] for x in np.argpartition(score_matrix, range(-k, 0, 1), None)[-k:]])
        lst.reverse()
        
        return lst
    
    def answer_extraction(self, context, question):
        k = 5
        try:
            inputs = self.tokenizer(
                text = context, 
                text_pair=question, 
                padding = 'max_length', 
                truncation = 'only_first', 
                max_length = 512, 
                return_tensors = 'pt', 
                padding_side = 'right'
                )
        except:
            cleaned = utils.clean_text(question)
            inputs =  self.tokenizer(
                text = context,
                text_pair = cleaned, 
                padding = 'max_length', 
                truncation = 'only_first', 
                max_length = 512, 
                return_tensors = 'pt', 
                padding_side = 'right',
                return_length = True
                )
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = None
        answer_end_index = None

        top_k = self.valid_spans(outputs.start_logits.detach(), outputs.end_logits.detach(), k)
        answer_start_index, answer_end_index= top_k[0]

        if answer_start_index == None : 
            answer_start_index = 0
            answer_end_index = 0     
                           
        answer = self.tokenizer.decode(inputs['input_ids'][0,answer_start_index:answer_end_index+ 1])
        confidence = outputs.start_logits[0, answer_start_index]  + outputs.end_logits[0, answer_end_index]

        return {"answer": answer, "confidence": confidence} 
        

    def __call__(
        self,
        question: str,
        num_guesses: int
    ) -> any:
        guesses = []
        contexts = self.context_model(question, 1)
        for context in contexts:
            curr_context = context["contents"]
            guesses.append(self.answer_extraction(curr_context, question))
        guesses.sort(key=lambda x: x["confidence"], reverse=True)
        # returns the top num_guesses guesses
        return guesses[:num_guesses]

    # returns a list of guesses for a list of questions
    def batch_guess(
        self, questions: list[str], num_guesses: int, truncate: str
    ) -> list[any]:
        return [self(q, num_guesses, truncate) for q in questions]

    # truncates the context to 450 tokens
    # returns a string
    def simple_truncation(self, question: str, length: int):
        tokens = self.tokenizer.tokenize(question)
        tokens = tokens[:length]
        return self.tokenizer.convert_tokens_to_string(tokens)

    # truncates the context into chunks of 450 tokens
    # returns a list of strings
    def chunk_truncation(self, question):
        chunks = []
        tokens = self.tokenizer.tokenize(question)
        while len(tokens) > 0:
            chunk = tokens[:450]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
            tokens = tokens[450:]
        return chunks
    
    def save(self, dir_name: str):
        try: 
            self.model.save_pretrained(save_directory = dir_name) 
        except:
            exit

    def load(self, dir_name: str):
        self.checkpoint = dir_name
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.checkpoint)
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.checkpoint)
        except Exception as e:
            print(f"Error loading BERT: {e}")
            exit(1)
