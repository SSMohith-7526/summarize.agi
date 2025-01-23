import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Download the 'punkt_tab' resource
nltk.download('punkt_tab') # This line is added to fix the LookupError

def text_summarizer(text, num_sentences=3):
    """
    Summarizes the given text using extractive summarization.

    Parameters:
    - text (str): The input text to summarize.
    - num_sentences (int): The number of sentences to include in the summary.

    Returns:
    - str: The summarized text.
    """
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text  # If text has fewer sentences than requested summary length

    # Preprocess the text
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = [
        ' '.join([word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words])
        for sentence in sentences
    ]

    # Compute TF-IDF scores for sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Calculate sentence scores by summing TF-IDF values
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Rank sentences by their scores
    ranked_sentences = [sentences[i] for i in np.argsort(-sentence_scores)]

    # Select the top N sentences for the summary
    summary = ' '.join(ranked_sentences[:num_sentences])

    return summary

if __name__ == "__main__":
    # Example text input
    input_text = (
        """Shivaji I (Shivaji Shahaji Bhonsale, Marathi pronunciation: [ʃiˈʋaːdʑiː ˈbʱos(ə)le]; c. 19 February 1630 – 3 April 1680)[6] was an Indian ruler and a member of the Bhonsle dynasty.[7] Shivaji carved out his own independent kingdom from the Sultanate of Bijapur that formed the genesis of the Maratha Confederacy.

Over the course of his life, Shivaji engaged in both alliances and hostilities with the Mughal Empire, the Sultanate of Golconda, the Sultanate of Bijapur and the European colonial powers. Shivaji offered passage and his service to Aurangzeb to invade the declining Sultanate of Bijapur. After Aurangzeb's departure for the north due to a war of succession, Shivaji conquered territories ceded by Bijapur in the name of the Mughals.[8] : 63  Following the Battle of Purandar, Shivaji entered into vassalage with the Mughal empire, assuming the role of a Mughal chief and was conferred with the title of Raja by Aurangzeb.[9] He undertook military expeditions on behalf of the Mughal empire for a brief duration.[10]

In 1674, Shivaji was coronated as the king despite opposition from local Brahmins.[8] : 87 [11] Praised for his chivalrous treatment of women,[12] Shivaji employed people of all castes and religions, including Muslims[13] and Europeans, in his administration and armed forces.[14] Shivaji's military forces expanded the Maratha sphere of influence, capturing and building forts, and forming a Maratha navy.

Shivaji's legacy was revived by Jyotirao Phule about two centuries after his death. Later on, he came to be glorified by Indian nationalists such as Bal Gangadhar Tilak, and appropriated by Hindutva activists.[15][16][17][18][19]"""
    )

    # Call the summarizer
    summary = text_summarizer(input_text)

    # Print the result
    print("Original Text:\n", input_text)
    print("\nSummarized Text:\n", summary)