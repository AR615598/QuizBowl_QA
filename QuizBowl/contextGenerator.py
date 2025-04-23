from typing import Any
from pyserini.search.lucene import LuceneSearcher
import json
import spacy
import utils 
import re

# class to generate context for bert_guess
class LuceneRetrieval():
    def __init__(self):
        self.searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")
        self.nlp = spacy.load("en_core_web_lg")

        
    def ner_extraction(self, question: str):
        tok = self.nlp(question)
        ents = [x.text for x in tok.ents]
        for toks in tok:
            if toks.pos_ == 'NOUN':
                ents.append(toks.text)
        agg_ents = " ".join(ents)
        return agg_ents
        
    

    def __call__(self, question: str, num_guesses: int) -> list[Any]:
        ents = self.ner_extraction(question)
        hits = self.searcher.search(ents, k=num_guesses)
        return self.scored_to_dict(hits, num_guesses)

    def batch_guess(self, questions: list[str], num_guesses: int):
        qids = [str(x) for x in range(len(questions))]
        hits =  self.searcher.batch_search(questions, qids, k=num_guesses, threads=1)
        batch = []
        for qid in qids:
            match = hits[qid]
            batch.append(self.scored_to_dict(match, num_guesses))
        return batch
    
    
    def remove_adj_dup(self, text: str):
        sents = text.split(". ")
        sent = sents[0]
        i = 1
        if len(sent) < 5:
            i = 2
            sent = " ".join(sents[:2])
        words = sent.split(" ")
        diff = 2

        seen = {}
        
        for word in words: 
            word = word.lower()
            if word in seen:
                seen[word] += 1
            else:
                seen[word] = 1
        ngram = []
        final = []
        for word in words: 
            temp = word.lower()
            if temp in seen and seen[temp] > 1 and temp not in ngram:
                ngram.append(temp)
                final.append(word)
                seen[temp] -= 1
            elif temp in ngram and seen[temp] >= 1:
                pass
            else:
                final.append(word)
                if diff <= 0:
                    ngram = []
                    diff = 2
                else:
                    diff -= 1
                
        sents = " ".join(final + sents[i:])
        return sents
    
    def scored_to_dict(self, obj, num_guesses):
        match = obj
        docs = []
        for hit in match:
            dict = json.loads(hit.lucene_document.get("raw"))
            contents = utils.lazy_split(dict["contents"], " ", 400)
            contents = (" ").join(contents)
            contents = re.sub('\n', ' ',contents)
            contents = self.remove_adj_dup(contents)
            

            score = hit.score
            id = dict["id"]
            docs.append({"id": id, "contents": contents, "confidence": score})
        if len(docs) < num_guesses:
            lst = [({"id": -1, "contents": "", "confidence": 0}) for x in range(num_guesses - len(docs))]
            docs.extend(lst)
        return docs
    


