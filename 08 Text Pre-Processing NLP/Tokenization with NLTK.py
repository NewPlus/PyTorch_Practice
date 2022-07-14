import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
en_text = "A Dog Run back corner near spare bedrooms"
print(word_tokenize(en_text))