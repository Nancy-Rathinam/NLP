import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords from NLTK (only need to run this once)
#nltk.download('stopwords')
#nltk.download('punkt')

# Function to calculate word frequency after removing stopwords
def word_frequency_without_stopwords(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        text = file.read().lower()  # Convert text to lowercase to ensure consistency
    
    # Tokenize the text into words (remove punctuation)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha()]  # Remove non-alphabetical tokens

    # Get the list of stopwords in English
    stop_words = set(stopwords.words('english'))

    # Remove stopwords from the list of words
    filtered_words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_count = Counter(filtered_words)

    # Return the word frequency as a dictionary
    return word_count

# Example usage
file_path = 'sample.txt'  # Replace with the path to your file
word_freq = word_frequency_without_stopwords(file_path)

wordcloud = WordCloud(width=900,height=500, background_color='SeaGreen').generate_from_frequencies(word_freq)
# Print word frequencies
for word, freq in word_freq.items():
    print(f"{word}: {freq}")

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
