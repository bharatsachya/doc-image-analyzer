from rake_nltk import Rake
from src.exception import CustomException
import sys
from src.logger import logging
def highlight_key_points(summary):
    # Use spaCy NER to find important entities (optional)
    logging.info("Highlighting key points in the summary")
    try:
        rake = Rake()
        rake.extract_keywords_from_text(summary)
        
        # Get the ranked phrases
        ranked_phrases = rake.get_ranked_phrases()
        
        # Highlight the important keywords/phrases
        highlighted_summary = summary
        for phrase in ranked_phrases[:5]:  # Select top 5 phrases
            highlighted_summary = highlighted_summary.replace(phrase, f"**{phrase}**")
        
        return highlighted_summary
    except Exception as e:
        raise CustomException(e,sys)

