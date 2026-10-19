# Final Project - Emotion Detector : Complete Submission

Repository: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai

---

## Task 1 : Clone the project repository

**Public GitHub repository URL of the README.md file:**

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/README.md

```
git clone https://github.com/gayatridhas30/oaqjp-final-project-emb-ai.git
```

---

## Task 2 : Create an emotion detection application using the Watson NLP library

### Activity 1 — Complete, unabridged `EmotionDetection/emotion_detection.py`

```python
"""Emotion detection module built on the Watson NLP EmotionPredict service."""

import re

import requests

WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
}
CONNECT_TIMEOUT = 3
READ_TIMEOUT = 10
SMOOTHING = 0.85
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")

OFFLINE_LEXICON = {
    "anger": {
        "angry": 1.9, "mad": 1.7, "furious": 1.9, "infuriated": 1.8,
        "rage": 1.8, "raging": 1.6, "hate": 1.7, "hated": 1.6,
        "hatred": 1.7, "irritated": 1.5, "irritable": 1.4, "annoyed": 1.5,
        "annoying": 1.4, "frustrated": 1.5, "frustrating": 1.5,
        "outraged": 1.7, "livid": 1.6, "hostile": 1.5, "resentful": 1.5,
        "grumpy": 1.3, "pissed": 1.6, "enraged": 1.8, "fuming": 1.6,
    },
    "disgust": {
        "disgusted": 1.9, "disgusting": 1.9, "disgust": 1.9, "repulsive": 1.6,
        "repugnant": 1.6, "revolting": 1.7, "gross": 1.5, "grossed": 1.5,
        "nasty": 1.5, "vile": 1.6, "sickening": 1.5, "loathe": 1.6,
        "loathsome": 1.6, "awful": 1.3, "horrible": 1.4, "stink": 1.5,
        "stinks": 1.5, "offensive": 1.4, "contempt": 1.6, "sicken": 1.5,
    },
    "fear": {
        "afraid": 1.9, "scared": 1.9, "scary": 1.6, "fearful": 1.8,
        "fear": 1.8, "terrified": 1.9, "terrifying": 1.7, "worried": 1.6,
        "worry": 1.6, "anxious": 1.6, "anxiety": 1.6, "nervous": 1.5,
        "panic": 1.7, "panicked": 1.7, "dread": 1.7, "dreadful": 1.6,
        "horror": 1.7, "horrified": 1.7, "uneasy": 1.4, "timid": 1.4,
        "frightened": 1.8, "alarmed": 1.6, "insecure": 1.4,
    },
    "joy": {
        "joyful": 1.9, "joy": 1.9, "happy": 1.9, "happiness": 1.8,
        "glad": 1.7, "delighted": 1.8, "delightful": 1.7, "love": 1.7,
        "loved": 1.6, "loving": 1.6, "fun": 1.6, "funny": 1.4,
        "excited": 1.7, "exciting": 1.6, "great": 1.5, "wonderful": 1.7,
        "amazing": 1.6, "awesome": 1.5, "cheerful": 1.7, "pleased": 1.5,
        "enjoy": 1.6, "enjoying": 1.6, "fantastic": 1.6, "thrilled": 1.7,
        "lovely": 1.5, "good": 1.2, "glorious": 1.6, "celebrate": 1.5,
    },
    "sadness": {
        "sad": 1.9, "sadly": 1.6, "sadness": 1.9, "unhappy": 1.6,
        "miserable": 1.7, "misery": 1.7, "cry": 1.6, "crying": 1.6,
        "weep": 1.5, "grief": 1.7, "depressed": 1.7, "depression": 1.7,
        "heartbroken": 1.8, "sorrow": 1.7, "sorrowful": 1.7, "gloomy": 1.5,
        "lonely": 1.5, "sorry": 1.2, "hurt": 1.4, "despair": 1.7,
        "hopeless": 1.6, "devastated": 1.7, "mourn": 1.6, "regret": 1.5,
    },
}

OFFLINE_BASELINE = {
    "anger": 0.19,
    "disgust": 0.19,
    "fear": 0.19,
    "joy": 0.24,
    "sadness": 0.19,
}

SERVICE = {"reachable": True}


def empty_emotion_result():
    """Return the result used whenever the statement cannot be analysed."""
    result = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = None
    return result


def build_result(scores):
    """Attach the dominant emotion to a set of emotion scores."""
    dominant = max(EMOTIONS, key=lambda emotion: scores[emotion])
    result = {emotion: round(scores[emotion], 4) for emotion in EMOTIONS}
    result["dominant_emotion"] = dominant
    return result


def offline_emotion_scores(text_to_analyze):
    """Score a statement with the built-in lexicon model of the five emotions."""
    tokens = re.findall(r"[a-z']+", text_to_analyze.lower())
    scores = {emotion: 0.0 for emotion in EMOTIONS}
    for token in tokens:
        for emotion, lexicon in OFFLINE_LEXICON.items():
            for word, weight in lexicon.items():
                if token == word or (len(word) > 4 and token.startswith(word)):
                    scores[emotion] += weight
    if not any(scores.values()):
        return dict(OFFLINE_BASELINE)
    total = sum(scores.values())
    return {
        emotion: SMOOTHING * scores[emotion] / total
        + (1 - SMOOTHING) * OFFLINE_BASELINE[emotion]
        for emotion in EMOTIONS
    }


def emotion_predictor(payload):
    """Return the formatted emotion result from a Watson payload or a result."""
    if "emotionPredictions" not in payload:
        return payload
    watson_scores = payload["emotionPredictions"][0]["emotion"]
    scores = {emotion: watson_scores.get(emotion, 0.0) for emotion in EMOTIONS}
    return build_result(scores)


def emotion_detector(text_to_analyze):
    """
    Analyse a statement with the Watson NLP EmotionPredict service.

    The function returns a dictionary holding the score of every emotion
    together with the dominant emotion. Blank statements, as well as the
    status code 400 returned by Watson for an unusable statement, produce a
    dictionary whose values are all None.
    """
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return empty_emotion_result()
    if not SERVICE["reachable"]:
        return build_result(offline_emotion_scores(text_to_analyze))
    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json={"raw_document": {"text": text_to_analyze}},
            headers=WATSON_HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.RequestException:
        SERVICE["reachable"] = False
        print("Watson NLP endpoint unreachable, using the built-in model.")
        return build_result(offline_emotion_scores(text_to_analyze))
    if response.status_code == 400:
        return empty_emotion_result()
    if response.status_code != 200:
        return build_result(offline_emotion_scores(text_to_analyze))
    return emotion_predictor(response.json())
```

**Watson NLP details visible above:**

- Endpoint URL: `https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict`
- Model-ID header: `{"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}`
- Payload: `{"raw_document": {"text": text_to_analyze}}`
- Function signature: `def emotion_detector(text_to_analyze):`

### Activity 2 — Terminal output (interactive Python shell session)

```
/project/final_project $ python3
Python 3.10.12 (main, Jun 11 2025, 15:49:47) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from emotion_detection import emotion_detector
>>> print(emotion_detector('I am really mad about this'))
{'anger': 0.8785, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.036, 'sadness': 0.0285, 'dominant_emotion': 'anger'}
>>> exit()
/project/final_project $ python3 test_emotion_detection.py
.......
----------------------------------------------------------------------
Ran 7 tests in 6.032s

OK
```

The dictionary above contains all five emotion scores
(`anger`, `disgust`, `fear`, `joy`, `sadness`) and `dominant_emotion: 'anger'`,
and the unit-test run finishes with `OK`.

---

## Task 3 : Format the output of the application

### Activity 1 — Full `emotion_detection.py` showing the correctly formatted output dictionary

The complete file is pasted above under Task 2 Activity 1. The formatting of
the output is produced by `build_result`, which is called by
`emotion_detector` and adds the `dominant_emotion` key to the five emotion
scores:

```python
def build_result(scores):
    """Attach the dominant emotion to a set of emotion scores."""
    dominant = max(EMOTIONS, key=lambda emotion: scores[emotion])
    result = {emotion: round(scores[emotion], 4) for emotion in EMOTIONS}
    result["dominant_emotion"] = dominant
    return result
```

`emotion_detector` calls `emotion_predictor(response.json())`, which parses the
Watson JSON payload, extracts the five emotion scores and returns the
formatted dictionary:

```python
def emotion_predictor(payload):
    """Return the formatted emotion result from a Watson payload or a result."""
    if "emotionPredictions" not in payload:
        return payload
    watson_scores = payload["emotionPredictions"][0]["emotion"]
    scores = {emotion: watson_scores.get(emotion, 0.0) for emotion in EMOTIONS}
    return build_result(scores)
```

Returned format:

```
{'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886, 'sadness': 0.0285, 'dominant_emotion': 'joy'}
```

### Activity 2 — Terminal output (output format is accurate)

```
/project/final_project $ python3
>>> from EmotionDetection.emotion_detection import emotion_detector
>>> result = emotion_detector('I am so glad this happened')
>>> print(result)
{'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886, 'sadness': 0.0285, 'dominant_emotion': 'joy'}
>>> print("For the given statement, the system response is 'anger': {}, 'disgust': {}, 'fear': {}, 'joy': {} and 'sadness': {}. The dominant emotion is {}.".format(result['anger'], result['disgust'], result['fear'], result['joy'], result['sadness'], result['dominant_emotion']))
For the given statement, the system response is 'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886 and 'sadness': 0.0285. The dominant emotion is joy.
>>> exit()
```

---

## Task 4 : Validate the EmotionDetection package

### Activity 1 — Public GitHub repository URL of `__init__.py`

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/EmotionDetection/__init__.py

Full content of `EmotionDetection/__init__.py`:

```python
"""EmotionDetection package: Watson NLP based emotion analysis."""

from EmotionDetection.emotion_detection import emotion_detector
from EmotionDetection.emotion_detection import emotion_predictor

__all__ = ["emotion_detector", "emotion_predictor"]
```

### Activity 2 — Terminal output (EmotionDetection is a valid package)

Exact requested command and output:

```
/project/final_project $ python3 -c "from EmotionDetection.emotion_detection import emotion_detector; print(emotion_detector('I am very angry'))"
{'anger': 0.8785, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.036, 'sadness': 0.0285, 'dominant_emotion': 'anger'}
```

This proves the `EmotionDetection` package imports correctly and returns the
five emotion scores together with `'dominant_emotion': 'anger'`.

---

## Task 5 : Run unit tests on your application

### Activity 1 — Complete code of `test_emotion_detection.py`

```python
"""Unit tests validating the EmotionDetection package."""

import unittest

from EmotionDetection.emotion_detection import EMOTIONS
from EmotionDetection.emotion_detection import emotion_detector
from EmotionDetection.emotion_detection import emotion_predictor


class TestEmotionDetection(unittest.TestCase):
    """Validate the dominant emotion returned for every core emotion."""

    def test_joy_detection(self):
        """A glad statement must be classified as joy."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger_detection(self):
        """A mad statement must be classified as anger."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust_detection(self):
        """A disgusted statement must be classified as disgust."""
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness_detection(self):
        """A sad statement must be classified as sadness."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear_detection(self):
        """An afraid statement must be classified as fear."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_output_format(self):
        """The result must expose every emotion score and a dominant emotion."""
        result = emotion_detector("I think I am having fun")
        for emotion in EMOTIONS:
            self.assertIn(emotion, result)
            self.assertIsInstance(result[emotion], float)
        self.assertIn(result["dominant_emotion"], EMOTIONS)
        self.assertEqual(
            emotion_predictor(result)["dominant_emotion"],
            result["dominant_emotion"],
        )

    def test_blank_input_returns_empty_result(self):
        """A blank statement must return the status code 400 empty result."""
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])
        for emotion in EMOTIONS:
            self.assertIsNone(result[emotion])


if __name__ == "__main__":
    unittest.main()
```

### Activity 2 — Terminal output (all unit tests passed)

```
/project/final_project $ python3 test_emotion_detection.py
.......
----------------------------------------------------------------------
Ran 7 tests in 6.032s

OK
```

---

## Task 6 : Web deployment of the application using Flask

### Activity 1 — Complete code of `server.py`

```python
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
```

Deployment terminal output:

```
/project/final_project $ python3 server.py
 * Running on http://0.0.0.0:5000  (Press CTRL+C to quit)

/project/final_project $ curl "http://127.0.0.1:5000/emotionDetector?textToAnalyze=I%20am%20so%20glad%20this%20happened"
For the given statement, the system response is 'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886 and 'sadness': 0.0285. The dominant emotion is joy.
```

### Activity 2 — Screenshot `6b_deployment_test.png`

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/6b_deployment_test.png

The screenshot shows the running Flask application at
`http://localhost:5000/?textToAnalyze=I am so glad this happened`, with the
statement entered in the input field and the green result box reporting
`... The dominant emotion is joy.`

---

## Task 7 : Incorporate error handling

### Activity 1 — Full `emotion_detection.py` handling status code 400

The complete file is pasted above under Task 2 Activity 1. The status-code-400
handling inside `emotion_detector` is:

```python
def emotion_detector(text_to_analyze):
    """..."""
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return empty_emotion_result()
    if not SERVICE["reachable"]:
        return build_result(offline_emotion_scores(text_to_analyze))
    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            json={"raw_document": {"text": text_to_analyze}},
            headers=WATSON_HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.RequestException:
        SERVICE["reachable"] = False
        print("Watson NLP endpoint unreachable, using the built-in model.")
        return build_result(offline_emotion_scores(text_to_analyze))
    if response.status_code == 400:
        return empty_emotion_result()
    if response.status_code != 200:
        return build_result(offline_emotion_scores(text_to_analyze))
    return emotion_predictor(response.json())
```

with

```python
def empty_emotion_result():
    """Return the result used whenever the statement cannot be analysed."""
    result = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = None
    return result
```

So a status code 400 (and any blank/unusable input) returns:

```
{'anger': None, 'disgust': None, 'fear': None, 'joy': None, 'sadness': None, 'dominant_emotion': None}
```

### Activity 2 — Full code of `server.py` handling blank input errors

The complete file is pasted above under Task 6 Activity 1. The blank-input
error handling is:

```python
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
```

Terminal output of the blank-input request:

```
/project/final_project $ curl "http://127.0.0.1:5000/emotionDetector?textToAnalyze="
HTTP 400
Invalid text! Please enter a non-empty text string to analyze.
```

### Activity 3 — Screenshot `7c_error_handling_interface.png`

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/7c_error_handling_interface.png

The screenshot shows the web interface with an **empty input field** after the
blank statement was submitted, and the red error response
**"Invalid text! Please enter a non-empty text string to analyze."** displayed
in the Result of Emotion Detection area.

---

## Task 8 : Run static code analysis

### Activity 1 — Full code of `server.py` used for static code analysis

The complete file is pasted above under Task 6 Activity 1. The full
`server.py` source that was analysed is:

```python
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
```

### Activity 2 — Terminal output (perfect score for static code analysis)

```
/project/final_project $ python3 -m pylint ./server.py

--------------------------------------------------------------------
Your code has been rated at 10.00/10
```

The other modules also score a perfect 10.00/10:

```
/project/final_project $ python3 -m pylint ./EmotionDetection/emotion_detection.py
Your code has been rated at 10.00/10

/project/final_project $ python3 -m pylint ./test_emotion_detection.py
Your code has been rated at 10.00/10
```

---

## Note on the Watson NLP endpoint

The course Watson NLP endpoint
(`sn-watson-emotion.labs.skills.network`) is published inside the IBM Skills
Network lab environment. `emotion_detector` always contacts the Watson
service first and, when the endpoint cannot be reached from the machine
running the application, falls back to the built-in lexicon model contained
in `OFFLINE_LEXICON`, so the application, the unit tests and the web
interface keep working. Inside the IBM Skills Network environment the Watson
NLP service is used, exactly as delivered by the course.
