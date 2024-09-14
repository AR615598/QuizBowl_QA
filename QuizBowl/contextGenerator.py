from typing import Any
from pyserini.search.lucene import LuceneSearcher
import subprocess
import os
import json
# enwiki-paragraphs
# wikipeia-dpr
# wikipedia-dpr-multi-bf
# wikipedia-dpr-dkrr-tqa


# class to generate context for bert_guess
class PyseriniGuesser:
    def __init__(self, json_file, use_prebuilt=True):
        self.json_file = json_file
        if not use_prebuilt:
            dir_path = 'indexes/sample_collection_json'
            if not os.path.exists(dir_path):
                self.load_data()
            self.searcher = LuceneSearcher('indexes/sample_collection_json')
        else:  
            self.searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')


    def update_index(self, json_file):
        pass
    def save_index(self):
        pass
    def load_index(self):
        pass
    



    def load_data(self):
        with open(self.json_file) as read:
            mylist = []
            data = json.load(read)
            data = data['questions']
            for question in data:
                answer = question['answer']
                cat = question['category']
                dataset = question['dataset']
                difficulty = question['difficulty']
                subcategory = question['subcategory']
                query = question['text']
                dataset = "N/A" if dataset is None else dataset
                difficulty = "N/A" if difficulty is None else difficulty
                subcategory = "N/A" if subcategory is None else subcategory
                seperator = " "
                contents = seperator.join([answer, cat, dataset, difficulty, subcategory, query])
                mylist.append({"id": answer, 'contents': contents})
        new_data = json.dumps(mylist)
        with open('pydata/BM25_1.json', 'w') as write:
            write.write(new_data)
        cmd_str = f"python3 -m pyserini.index.lucene --collection JsonCollection --input pydata --index indexes/sample_collection_json --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw"
        subprocess.run(cmd_str, shell=True)



    def save(self): 
        pass

    def __call__(self, question: str, num_guesses: int) -> list[Any]:
        hits = self.searcher.search(question, k=num_guesses)
        docs = []
        for hit in hits:

            dict= json.loads(hit.lucene_document.get('raw'))            
            contents = dict['contents']
            score = hit.score
            id = dict['id']

            docs.append({"id": id, "contents": contents, "confidence": score})
        return docs
    
    def batch_guess(self, questions: list[str], num_guesses: int) -> list[list[Any]]:
        batch_guesses = []
        for question in questions:
            batch_guesses.append(self(question, num_guesses))
        return batch_guesses



