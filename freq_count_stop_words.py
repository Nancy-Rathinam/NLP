from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

file_path = 'sample.txt'
with open(file_path, 'r') as file:
        text = file.read().lower()
stop_words = set(stopwords.words('english'))
stop_words.extend('.')

word_tokens = word_tokenize(text)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
print("The Word Tokens are:")
print(word_tokens)
print("The filtered sentence without stopwords: ")
print(filtered_sentence)
word_count = Counter(filtered_sentence)
for word, freq in word_count.items():
    print(f"{word}: {freq}")
