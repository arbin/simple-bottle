import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer for sentiment analysis
sid = SentimentIntensityAnalyzer()


# Sample text for sentiment analysis
sample_text = "This is a very good movie! I loved the plot and the characters."

# Tokenize the text (split into words)
tokens = word_tokenize(sample_text.lower())  # Convert text to lowercase for consistency

# Remove stopwords (common words like 'the', 'is', 'a', etc.)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]


# Calculate sentiment scores using VADER (Valence Aware Dictionary and sEntiment Reasoner)
sentiment_scores = sid.polarity_scores(" ".join(filtered_tokens))

# Interpret sentiment scores
sentiment_label = 'positive' if sentiment_scores['compound'] >= 0 else 'negative'

# Print sentiment analysis results
print("Sample Text:", sample_text)
print("Sentiment:", sentiment_label)
print("Sentiment Scores:", sentiment_scores)


"""
Explanation of Steps:
Step 1: We import necessary libraries including nltk for natural language processing tasks such as tokenization, stopword removal, and sentiment analysis using VADER.
Step 2: We download required NLTK resources (punkt for tokenization, stopwords for stopwords, and vader_lexicon for VADER sentiment analysis).
Step 3: We define a sample text and preprocess it by tokenizing the text into words using word_tokenize(), converting the text to lowercase for consistency, and removing stopwords to filter out common words.
Step 4: We use the VADER sentiment analyzer (SentimentIntensityAnalyzer) to calculate sentiment scores (positive, negative, neutral, and compound score) for the preprocessed text. The compound score represents the overall sentiment polarity of the text. We interpret the sentiment label based on the compound score.


This example demonstrates a basic NLP workflow for sentiment analysis using NLTK in Python. 
You can expand on this by applying more advanced techniques such as part-of-speech tagging, named entity recognition, 
or using more sophisticated machine learning models for sentiment classification based on labeled datasets.

"""