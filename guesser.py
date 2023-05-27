from transformers import BertTokenizer, BertForQuestionAnswering
import transformers
import contextGenerator 
import torch
import regex as re
import math
from  utils import utils
import json
from transformers import TrainingArguments, Trainer


# generates guesses for a given question
class BertGuess:
    def __init__(self, bool: True):

        try:
            # initialize pyserini_guesser
            # used to get context for bert
            self.context_model = contextGenerator.pyserini_guesser('', bool)
        except Exception as e:
            print(f"Error loading Pyserini: {e}")
            exit(1)
        try:
            # initialize bert
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            
        except Exception as e:
            print(f"Error loading BERT: {e}")
            exit(1)
        # for training, but i couldent get it to work    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()





    # returns a list of guesses for a given question
    # truncate_type can be 'simple' or 'chunk'
    # simple truncation truncates the context to 450 tokens
    # chunk truncation splits the context into chunks of 450 tokens
    # and returns the highest confidence guess from each chunk
    # Simple truncation is faster, but chunk truncation can get more accurate guesses
    # chunk truncation takes way too long to run, only do for maybe 1 question at a time
    def __call__(self, question: str, num_guesses: int, truncate_type) -> any:
        # list of dicts
        guesses = []
        # gets the actual question from the question string
        match = re.match(r'.* For \d+ points, (.*).', question)
        if match is not None:
            actual_quest = (match.group(1))      
        else:
            actual_quest = question  

        
        # gets the context from the question
        contexts = self.context_model(question, 1)

        # loops through each context and gets the guess
        for context in contexts:
            # Decides whether to use simple or chunk truncation
            curr_context = context['contents']
            if truncate_type == 'simple':
                # truncates the context to 450 tokens
                curr_context = self.simple_truncation(curr_context)
                # gets the encoded dict from the tokenizer
                encoded_dict = self.tokenizer.encode_plus(text = actual_quest, text_pair=curr_context, add_special_tokens=True, return_tensors='pt')
                # gets the input ids and segment ids from the encoded dict
                input_ids = encoded_dict['input_ids']
                segment_ids = encoded_dict['token_type_ids']
                # gets the output from the model
                out = self.model(input_ids, token_type_ids=segment_ids)
                # gets the start and end indices of the answer
                answerStart = torch.argmax(out.start_logits)
                answerEnd = torch.argmax(out.end_logits)
                # gets the tokens from the input ids
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                # gets the answer from the tokens
                answer = tokens[answerStart]
                for i in range(answerStart + 1, answerEnd + 1):
                    if tokens[i][0:2] == '##':
                        answer += tokens[i][2:]
                    else:
                        answer += ' ' + tokens[i]
                # gets the confidence of the answer    
                confidence = math.exp(out.start_logits[0][answerStart].item()) + math.exp(out.end_logits[0][answerEnd].item())
                guesses.append({'answer': answer, 'confidence': confidence})
            elif truncate_type == 'chunk':
                # truncates the context into chunks of 450 tokens
                # takes way too long to run
                curr_context = self.chunk_truncation(curr_context)
                # gets the highest confidence guess from each chunk
                highest_confidence = -100
                highest_answer = None
                for chunk in curr_context:
                    encoded_dict = self.tokenizer.encode_plus(text = actual_quest, text_pair=chunk, add_special_tokens=True, return_tensors='pt')
                    input_ids = encoded_dict['input_ids']
                    segment_ids = encoded_dict['token_type_ids']
                    out = self.model(input_ids, token_type_ids=segment_ids)
                    answerStart = torch.argmax(out.start_logits)
                    answerEnd = torch.argmax(out.end_logits)
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                    answer = tokens[answerStart]
                    for i in range(answerStart + 1, answerEnd + 1):
                        if tokens[i][0:2] == '##':
                            answer += tokens[i][2:]
                        else:
                            answer += ' ' + tokens[i]
                    confidence = math.exp(out.start_logits[0][answerStart].item()) + math.exp(out.end_logits[0][answerEnd].item())
                    # gets the highest confidence guess
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        highest_answer = answer
                guesses.append({'answer': highest_answer, 'confidence': highest_confidence})
        guesses.sort(key=lambda x: x['confidence'], reverse=True)
        # returns the top num_guesses guesses
        return guesses[:num_guesses]

            
            
        
    # returns a list of guesses for a list of questions
    def batch_guess(self, questions: list[str], num_guesses: int, truncate: str) -> list[any]:
        return [self(q, num_guesses, truncate) for q in questions]
    

    # truncates the context to 450 tokens
    # returns a string 
    def simple_truncation(self, question: str):
        tokens = self.tokenizer.tokenize(question)
        tokens = tokens[:412]
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
    

