import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK corpora are downloaded once
for corpus in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{corpus}')
    except LookupError:
        nltk.download(corpus, quiet=True)

# Initialize once at module load for fast execution
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Keep negations so phrases like "not good" retain their negative meaning
negations = {
    'no', 'nor', 'not', 'never', 'neither',
    'dont', 'didnt', 'wont', 'cant', 'isnt', 'arent', 'wasnt', 'werent'
}
stop_words = stop_words - negations

# Pre-compiled regex patterns
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
MENTION_REGEX = re.compile(r'@\w+')
NUMBER_REGEX = re.compile(r'\d+')
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def preprocess_text(text: str) -> str:
    """
    Cleans raw text for NLP model training and Flask inference:
    1. Lowercases text
    2. Removes URLs and @mentions
    3. Removes numbers and punctuation
    4. Removes stopwords (excluding negations)
    5. Applies WordNet lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and @mentions
    text = URL_REGEX.sub('', text)
    text = MENTION_REGEX.sub('', text)

    # 3. Remove numbers and punctuation
    text = NUMBER_REGEX.sub('', text)
    text = text.translate(PUNCT_TABLE)

    # 4. Tokenize & remove stopwords, 5. Lemmatize
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_words)
