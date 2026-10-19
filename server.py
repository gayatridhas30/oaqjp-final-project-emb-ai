"""Flask server that deploys the Emotion Detector application on the web."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

INVALID_TEXT_MESSAGE = (
    "Invalid text! Please enter a non-empty text string to analyze."
)


@app.route("/emotionDetector")
def emotion_analyzer():
    """Analyse the submitted text and return the formatted emotion report."""
    text_to_analyse = request.args.get("textToAnalyze")
    if text_to_analyse is None or not text_to_analyse.strip():
        return INVALID_TEXT_MESSAGE, 400
    result = emotion_detector(text_to_analyse)
    if result["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE, 400
    return (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. The dominant emotion is "
        f"{result['dominant_emotion']}."
    )


@app.route("/")
def render_index_page():
    """Render the page that hosts the emotion detection interface."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
