from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, RobertaForQuestionAnswering
from utils import clean_text
from contextGenerator import LuceneRetrieval
import torch
import torch.nn.functional as f


class BertGuess:
    def __init__(self):
        checkpoint = "deepset/bert-base-cased-squad2"

        try:
            self.context_model = LuceneRetrieval()
        except Exception as e:
            print(f"Error loading Lucene: {e}")
            exit(1)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = BertForQuestionAnswering.from_pretrained(checkpoint)
        except Exception as e:
            print(f"Error loading BERT: {e}")
            exit(1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        
    
    def answer_extraction(self, context, question):
        encoded_dict = self.tokenizer(
            text=question,
            text_pair=context,
            return_tensors="pt",
            max_length=512,
            truncation="only_second",
        )
        with torch.no_grad():
            outputs = self.model(**encoded_dict)

        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax()

        predict_answer_tokens = encoded_dict.input_ids[
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
        preprocessing: bool = True,
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
