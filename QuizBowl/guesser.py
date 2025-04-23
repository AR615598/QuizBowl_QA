from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from utils import clean_text
from contextGenerator import LuceneRetrieval
import torch
import torch.nn.functional as f
import spacy
import utils
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
            num_guesses: int,
            truncate_type: str,
            preprocessing: bool = False):
        
        return self.model(
            question,
            num_guesses,
            truncate_type,
            preprocessing,
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
                  num_guesses: int,
                  truncate_type: str,
                  preprocessing: bool = True):
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
        
        
    
    def answer_extraction(self, context, question):
        try: 
            encoding =  self.tokenizer(
                text = context, 
                text_pair = question, 
                padding = 'max_length', 
                truncation = 'only_first', 
                max_length = 512, 
                return_tensors = 'pt', 
                padding_side = 'right',
            )
        except:
            cleaned = utils.clean_text(question)
            encoding =  self.tokenizer(
                text = context,
                text_pair = cleaned, 
                padding = 'max_length', 
                truncation = 'only_first', 
                max_length = 512, 
                return_tensors = 'pt', 
                padding_side = 'right',
                )
        with torch.no_grad():
            outputs = self.model(**encoding)

        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax()

        predict_answer_tokens = encoding.input_ids[
            0, answer_start : answer_end + 1
        ]
        
        answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )

        start_probs = f.softmax(outputs.start_logits, dim=-1)
        end_probs = f.softmax(outputs.end_logits, dim=-1)

        confidence = start_probs[0, answer_start] * end_probs[0, answer_end]

        return {"answer": answer, "confidence": confidence}
        

    def __call__(
        self,
        question: str,
        num_guesses: int,
        truncate_type: str,
        preprocessing: bool = False,
    ) -> any:
        guesses = []
        contexts = self.context_model(question, 1)
        for context in contexts:
            curr_context = context["contents"]
            if preprocessing:
                curr_context = clean_text(curr_context)
                question = clean_text(question)
            
            if truncate_type == "simple":
                guesses.append(self.answer_extraction(curr_context, question))
                
                
            elif truncate_type == "chunk":
                # truncates the context into chunks of 450 tokens
                # takes way too long to run
                curr_context = self.chunk_truncation(curr_context)
                # gets the highest confidence guess from each chunk
                highest_confidence = -100
                highest_answer = None
                for chunk in curr_context:
                    guesses.append(self.answer_extraction(chunk, question))
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
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.checkpoint)
            self.model = DistilBertModel.from_pretrained(self.checkpoint)
        except Exception as e:
            print(f"Error loading BERT: {e}")
            exit(1)
