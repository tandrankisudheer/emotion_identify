import gradio as gr
from transformers import pipeline

# Load the emotion classification pipeline
emotion_analysis = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Dictionary of remedies for each emotion
remedies = {
    "joy": "Keep enjoying the moment and share your happiness with others!",
    "anger": "Take deep breaths, go for a walk, or try some relaxation exercises.",
    "envy": "Focus on your own strengths and achievements. Practice gratitude.",
    "depression": "Consider talking to a friend or mental health professional. Engage in activities you enjoy.",
    "anxiety": "Practice mindfulness, deep breathing exercises, or yoga.",
    "embarrassment": "Remember that everyone makes mistakes. Learn from the experience and move on.",
    "shyness": "Start with small social interactions and gradually increase your exposure.",
    "sadness": "Talk to someone you trust, engage in physical activity, or do something creative.",
    "disgust": "Try to identify the source of your feeling and address it constructively."
}

# Function to analyze emotions
def analyze_emotions(message):
    analysis = emotion_analysis(message)[0]
    # Extracting the most likely emotion and its score
    emotion = max(analysis, key=lambda x: x['score'])
    emotion_label = emotion['label']
    score = emotion['score']

    # Get the remedy for the identified emotion
    remedy = remedies.get(emotion_label.lower(), "Take care of yourself and seek help if needed.")

    response = f"Identified Emotion: {emotion_label}, Score: {score:.2f}\nRemedy: {remedy}"
    return response

# Create the Gradio interface
interface = gr.Interface(
    fn=analyze_emotions,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="Emotion Identifier Chatbot",
    description="This chatbot identifies your emotions based on your messages and provides a remedy."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
