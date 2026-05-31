from transformers import pipeline

# Initialize the summarization pipeline with a pre-trained model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text: str) -> str:
    """
    Summarizes the input text using a pre-trained transformer model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    if not text.strip():
        return "Input text is empty."

    # Generate the summary
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]["summary_text"]
