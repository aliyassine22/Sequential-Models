import nltk
import torch
import torch.nn as nn
import re
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

# function to replace the contracted words with their longer counterparts
def decontract_words(text):
    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    return " ".join(new_text)


# function to format words and remove unwanted characters using regex
def format_text_regex(text):
    # ^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%.\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%\+.~#?&\/=]*)$
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #clean all URLs
    text = re.sub(r'\<a href', ' ', text) #clean html style URL
    text = re.sub(r'&amp;', '', text) #remove &amp; chars
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text) #remove special characters
    text = re.sub(r'<br />', ' ', text) #remove html style <br>
    text = re.sub(r'\'', ' ', text)
    return text

# function to remove stop words usng the nltk
def remove_stopwords(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    return " ".join(text)



# function that groups logic from other preprocessing functions to clean text
def clean_text(text):
    # Convert words to lower case
    text = text.lower()
    # Use other preprocessing functions
    text = decontract_words(text)
    text = format_text_regex(text)
    text = remove_stopwords(text)
    # Tokenize each word
    text =  nltk.WordPunctTokenizer().tokenize(text)
    return text



# function to lemmatize words in text cleaned and create a new column lemmatized text and store them there
def lemmatized_words(text):
    lemm = nltk.stem.WordNetLemmatizer()
    lemmatized_text = [lemm.lemmatize(word) for word in text]
    return lemmatized_text

path=r"pretrained models\Embeddings Model\Word2Vector_model"
model_cbow = Word2Vec.load(path)

def convert_sequences_to_tensor(model=model_cbow,sequences=[], num_tokens_in_sequence=125, embedding_size=300):
    '''
    We want a torch.FloatTensor() of size (num_sequences, num_tokens_in_sequence, embedding_size)
    '''
    num_sequences = len(sequences)
    print((num_sequences, num_tokens_in_sequence, embedding_size))

    data_tensor = torch.zeros((num_sequences, num_tokens_in_sequence, embedding_size))

    for index, review in enumerate(list(sequences)):
        # Create a word embedding for each word in the review (where a review is a sequence)
        truncated_clean_review = review[:num_tokens_in_sequence]  # truncate to sequence length limit

        if len(truncated_clean_review) == 0: # accounting for the case where some words might not be recognized by the word2vector model
            continue

        list_of_word_embeddings = [
            model.wv[word] if word in model.wv else [0.0] * embedding_size for word in truncated_clean_review
        ]

        # convert the review to a tensor
        sequence_tensor = torch.FloatTensor(list_of_word_embeddings)

        # add the review to our tensor of data
        review_length = sequence_tensor.shape[0]  # (review_length, embedding_size)
        data_tensor[index, :review_length, :] = sequence_tensor

    return data_tensor


# the function that do glues everything together
def perprocess_text(text):
    cleaned_text = clean_text(text)
    lemmatized_text = lemmatized_words(cleaned_text)
    tensor = convert_sequences_to_tensor(sequences=[lemmatized_text])
    # print('text:', text)
    # print('cleaned_text:', cleaned_text)
    # print('lemmatized_text:', lemmatized_text)
    # print('tensor:', tensor)
    # print('tensor shape:', tensor.shape)
    return tensor 

# testing before connecting to the model
# text ='Thank God for everything good and bad, Forever Grateful'
# perprocess_text(text)