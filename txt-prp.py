import string 
import re 


from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

import nltk
nltk.download('punkt')



def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")
                
df['full_text'] = df['full_text'].apply(remove_tweet_special)


def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['full_text'] = df['full_text'].apply(remove_number)


def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['full_text'] = df['full_text'].apply(remove_punctuation)


def remove_whitespace_LT(text):
    return text.strip()

df['full_text'] = df['full_text'].apply(remove_whitespace_LT)


def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

df['full_text'] = df['full_text'].apply(remove_whitespace_multiple)


def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

df['full_text'] = df['full_text'].apply(remove_singl_char)


def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['tweet_tokens'] = df['full_text'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(df['tweet_tokens'].head())
print('\n\n\n')