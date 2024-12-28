from typing import Any
from pyserini.search.lucene import LuceneSearcher
import json

# enwiki-paragraphs
# wikipeia-dpr
# wikipedia-dpr-multi-bf
# wikipedia-dpr-dkrr-tqa


# class to generate context for bert_guess
class LuceneRetrieval():
    def __init__(self):
        self.searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")

    def __call__(self, question: str, num_guesses: int) -> list[Any]:
        hits = self.searcher.search(question, k=num_guesses)
        docs = []
        for hit in hits:
            dict = json.loads(hit.lucene_document.get("raw"))
            contents = dict["contents"]
            score = hit.score
            id = dict["id"]
            docs.append({"id": id, "contents": contents, "confidence": score})
        if len(docs) < num_guesses:
            lst = [({"id": -1, "contents": "", "confidence": 0}) for x in range(num_guesses - len(docs))]
            docs.extend(lst)
        return docs

    def batch_guess(self, questions: list[str]):
        qids = [x for x in range(len(questions))]
        return self.searcher.batch_search(questions,qids=qids, k=1)
