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
class pyserini_guesser:
    def __init__(self, json_file, use_prebuilt=True):
        self.json_file = json_file
        # You can either use a prebuilt index or build your own
        # If you want to build your own, set use_prebuilt to False
        # The prebuilt index is the wikipedia-kilt-doc index it is around 12gb 
        # if not found it will be downloaded automatically. If you choose to build your own
        # it will be much smaller, but should not be used with the bert_guess model since the 
        # Content is generally not imformative enough for bert_guess to make a good guess.
        # Though by using self built index you can get the guess from the id of the document


        # So if you use prebuilt you need to use bert_guess. 
        # If you use your own index you should use pure retrieval by calling it 
        # and passing in the id of the document you want to use as the guess
        # Prebuilt is much slower and does not always work as well as your own index.
        if not use_prebuilt:
            dir_path = 'indexes/sample_collection_json'
            if not os.path.exists(dir_path):
                # for pure retrieval 
                self.load_data()
            self.searcher = LuceneSearcher('indexes/sample_collection_json')
        else:  
            self.searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')


    # This function is used to build your own index
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

    # retuns a list of documents
    # each document is a dict with the following keys
    # id: the id of the document
    # contents: the contents of the document
    # confidence: the confidence of the document
    def __call__(self, question: str, num_guesses: int) -> list[Any]:
        hits = self.searcher.search(question, k=num_guesses)
        docs = []
        for hit in hits:
            dict= json.loads(hit.raw)            
            contents = dict['contents']
            score = hit.score
            id = dict['id']
            docs.append({"id": id, "contents": contents, "confidence": score})
        return docs
    

    # returns a list of guesses for a list of questions    
    def batch_guess(self, questions: list[str], num_guesses: int) -> list[list[Any]]:
        batch_guesses = []
        for question in questions:
            batch_guesses.append(self(question, num_guesses))
        return batch_guesses
