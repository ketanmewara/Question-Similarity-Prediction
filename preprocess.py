from bs4 import BeautifulSoup
import re
import pickle
import pickle
import numpy as np

tfidf_cv = pickle.load(open('Models/tfidf_cv.pkl','rb'))


def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)



def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))


def preprocess(data):
    
    data = str(data).lower().strip()
    
    # Replace special characters with their string edatauivalents.
    data = data.replace('%', ' percent')
    data = data.replace('$', ' dollar ')
    data = data.replace('₹', ' rupee ')
    data = data.replace('€', ' euro ')
    data = data.replace('@', ' at ')
    data = data.replace('#', '')
    data = data.replace('u.s.', 'usa')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    data = data.replace('[math]', '')
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
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
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    data_decontracted = []

    for word in data.split():
        if word in contractions:
            word = contractions[word]

        data_decontracted.append(word)

    data = ' '.join(data_decontracted)
    data = data.replace("'ve", " have")
    data = data.replace("n't", " not")
    data = data.replace("'re", " are")
    data = data.replace("'ll", " will")
    
    # Removing HTML tags
    data = BeautifulSoup(data)
    data = data.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    data = re.sub(pattern, ' ', data).strip()

    
    return data



def preprocessing(q1,q2):
    
    input_data = []
    
    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # fetch basic features
    input_data.append(len(q1))
    input_data.append(len(q2))
    
    input_data.append(len(q1.split(" ")))
    input_data.append(len(q2.split(" ")))
    
    input_data.append(test_common_words(q1,q2))
    input_data.append(test_total_words(q1,q2))
    input_data.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))
    
    # bow feature for q1
    q1_bow = tfidf_cv.transform([q1]).toarray()
    
    # bow feature for q2
    q2_bow = tfidf_cv.transform([q2]).toarray()
    
    
    
    return np.hstack((np.array(input_data).reshape(1,7),q1_bow,q2_bow))
    
