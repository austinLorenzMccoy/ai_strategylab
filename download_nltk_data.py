"""
Download required NLTK data packages for the sentiment analysis service.
"""

import nltk

# Download required NLTK data packages
nltk.download('punkt')
nltk.download('stopwords')

print("NLTK data packages downloaded successfully.")
