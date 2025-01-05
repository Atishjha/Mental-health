import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Download necessary NLTK resources (only needed once)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Define mental health-related keywords
MENTAL_HEALTH_KEYWORDS = [
    "depression", "anxiety", "stress", "suicide", "hopeless", "panic",
    "fear", "sadness", "lonely", "isolation", "overwhelmed", "worthless",
    "self-harm", "insomnia", "fatigue", "anger", "guilt"
]

# Solutions or advice for each keyword
MENTAL_HEALTH_SOLUTIONS = {
    "depression": "Consider reaching out to a mental health professional and maintaining a daily journal to track your emotions.",
    "anxiety": "Practice mindfulness and deep breathing exercises. Regular physical activity can also help.",
    "stress": "Try managing your time effectively, prioritize tasks, and engage in relaxing activities such as yoga or meditation.",
    "suicide": "Please seek immediate help from a professional or contact a crisis hotline. You are not alone.",
    "hopeless": "Connect with supportive friends or family. Sometimes talking about your feelings can help.",
    "panic": "Try grounding techniques, such as focusing on your breathing or describing your surroundings.",
    "fear": "Identify the source of your fear and take small steps to face it. Support groups can also be helpful.",
    "sadness": "Engage in activities that you enjoy and connect with loved ones for emotional support.",
    "lonely": "Join social groups or online communities that align with your interests.",
    "isolation": "Consider reaching out to friends or family members and engaging in community activities.",
    "overwhelmed": "Break tasks into smaller steps, delegate responsibilities, and take breaks when needed.",
    "worthless": "Remember that self-worth is not defined by external factors. Engage in activities that boost self-esteem.",
    "self-harm": "Reach out to a trusted friend or counselor. There are healthier ways to cope with difficult emotions.",
    "insomnia": "Maintain a consistent sleep schedule and avoid screens before bedtime. Consider relaxation techniques.",
    "fatigue": "Ensure you're getting enough rest and balanced nutrition. If persistent, consult a healthcare professional.",
    "anger": "Practice anger management techniques like deep breathing and counting to ten before reacting.",
    "guilt": "Acknowledge the source of your guilt, make amends if possible, and focus on self-forgiveness."
}

def clean_text(text):
    """ 
    Cleans the input text by removing special characters and stopwords.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def analyze_mental_health(text):
    """
    Analyze the mental health of a person based on the input text.
    Parameters:
        text (str): The input text to analyze.
    Returns:
        dict: A dictionary containing the sentiment score, detected mental health indicators, and solutions.
    """
    cleaned_text = clean_text(text)

    # Sentiment Analysis
    sentiment_score = sia.polarity_scores(cleaned_text)

    # Keyword Detection
    detected_keywords = [word for word in MENTAL_HEALTH_KEYWORDS if word in cleaned_text]

    # Mental health status based on sentiment and keywords
    if sentiment_score['compound'] <= -0.3 or len(detected_keywords) > 2:
        mental_health_status = "Potential Mental Health Concern"
    else:
        mental_health_status = "No Significant Concern"

    # Provide solutions based on detected keywords
    solutions = [MENTAL_HEALTH_SOLUTIONS[keyword] for keyword in detected_keywords]

    # Result Dictionary
    result = {
        "Sentiment Score": sentiment_score,
        "Detected Keywords": detected_keywords,
        "Mental Health Status": mental_health_status,
        "Solutions": solutions if solutions else ["Keep monitoring your mental health and focus on positive activities."]
    }

    return result


if __name__ == "__main__":
    text_input = input("Tell me about your Problems: ")
    
    analysis_result = analyze_mental_health(text_input)
    print("\nMental Health Analysis Result:")
    for key, value in analysis_result.items():
        print(f"{key}: {value}")
