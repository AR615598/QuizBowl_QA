import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation
from transformers import BertTokenizer, BertModel
from transformers import BatchEncoding
import torch
import spacy
import re
## You may need to download stopwords and wordnet with the following: nltk.download('wordnet')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# nlp1 = spacy.load('en_trf_bertbaseuncased_lg')
nlp2 = spacy.load('en_core_web_lg')
nlp3 = spacy.load('en_core_web_sm')


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

## General Utils
def get_top_n(lst: list, n: int) -> list:
    """`
    This will sort the list then return the top n items from a list.
    return: List
    """
    lst.sort(reverse = True)
    return lst[:n]

def lazy_split(string, sep, n = 400):
    splits = []
    temp = ''
    i = 0
    for char in string:
        if i > n:
            break
        if char == sep:
            splits.append(temp)
            temp = ''
            i+=1
        else:
            temp += char    
    return splits

def term_char_index(ans: str, context: str):
    return [(m.start(), m.end()) for m in re.finditer(ans.replace("_"," "), context, flags = re.IGNORECASE)]

# char_to_token 
def tokenize_row(row: dict, tokenizer):
    encoding =  tokenizer(
        text = row['context']['contents'], 
        text_pair = row['full_question'], 
        padding = 'max_length', 
        truncation = 'only_first', 
        max_length = 512, 
        return_tensors = 'pt', 
        padding_side = 'right'
        )
    start_pos = []
    end_pos = []
    # Convert the dictionary to a BatchEncoding object
    for (x, y) in row['char_pos']:
        start_pos.append(encoding.char_to_token(x))
        end_pos.append(encoding.char_to_token(y - 1))
    return start_pos, end_pos, encoding
    

def token_positions(row: dict):
    start_pos = []
    end_pos = []
    # Convert the dictionary to a BatchEncoding object
    encodings = BatchEncoding(row['encodings'], tensor_type='pt')
    for (x, y) in row['char_pos']:
        start_pos.append(encodings.char_to_token(x))
        end_pos.append(encodings.char_to_token(y - 1))
    return start_pos, end_pos