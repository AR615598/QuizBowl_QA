import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation
# from transformers import BertTokenizer, BertModel
import torch
import spacy
## You may need to download stopwords and wordnet with the following: nltk.download('wordnet')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# nlp1 = spacy.load('en_trf_bertbaseuncased_lg')
nlp2 = spacy.load('en_core_web_lg')
nlp3 = spacy.load('en_core_web_sm')

class utils:
## Text Preprocessing
    def clean_text(text: str) -> str:
        """
        This will lowercase the text, remove punctuation, remove stopwords, and lemmatize.
        return: str
        """
        # Lowercase
        text = text.lower()
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        # Remove punctuation
        text = ''.join([char for char in text if char not in punctuation])
        # Lemmatize
        lemma = WordNetLemmatizer()
        text = ' '.join([lemma.lemmatize(word) for word in text.split()])
        return text


    def tokenize_text_words(text: str) -> list[str]:
        """
        This will tokenize the text into words.
        return: List[str]
        """
        return word_tokenize(text)

    def tokenize_text_sentences(text: str) -> list[str]:
        """
        This will tokenize the text into sentences.
        return: List[str]
        """
        return sent_tokenize(text)

    def word_embedding(text: str, contexual: bool = False) -> torch.tensor:
        """
        This will return a word embedding of the text.
        return: List[str]
        """
        if contexual:
            return torch.tensor(nlp2(text).vector)
        else:
            return torch.tensor(nlp3(text).vector)


## General Utils
    def get_top_n(lst: list, n: int) -> list:
        """
        This will sort the list then return the top n items from a list.
        return: List
        """
        lst.sort(reverse = True)
        return lst[:n]
    
        
