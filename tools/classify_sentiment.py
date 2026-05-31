from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def classify_sentiment(text: str) -> str:
    """
    Classifies the sentiment of the provided text.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL').
    """
    if not text.strip():
        return "Input text is empty."

    result = sentiment_pipeline(text)[0]
    label = result['label'].upper()

    # Normalize the label to 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
    if label in ['POSITIVE', 'NEGATIVE']:
        return label
    else:
        return 'NEUTRAL'
