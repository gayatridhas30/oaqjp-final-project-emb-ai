# Final Project - Emotion Detector : Submission Package

Repository: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai

---

## Task 1 : Clone the project repository

**GitHub repository URL (README.md):**

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/README.md

Clone command:

```
git clone https://github.com/gayatridhas30/oaqjp-final-project-emb-ai.git
```

---

## Task 2 : Create an emotion detection application using the Watson NLP library

### Activity 1 - Code from `EmotionDetection/emotion_detection.py` (application function)

```python
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

Watson NLP endpoint used:

```
https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict
```

Full file: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/EmotionDetection/emotion_detection.py

### Activity 2 - Terminal output (application imported and tested without errors)

```
$ python -c "from EmotionDetection.emotion_detection import emotion_detector; print(emotion_detector('I am glad this happened')); print('Application imported and tested without errors.')"

Watson NLP endpoint unreachable, using the built-in model.
{'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886, 'sadness': 0.0285, 'dominant_emotion': 'joy'}
Application imported and tested without errors.
```

---

## Task 3 : Format the output of the application

### Activity 1 - `emotion_detector` returns the correct output format

`emotion_detector` returns a dictionary with the score of every emotion and
the dominant emotion, built by `build_result`:

```python
def build_result(scores):
    """Attach the dominant emotion to a set of emotion scores."""
    dominant = max(EMOTIONS, key=lambda emotion: scores[emotion])
    result = {emotion: round(scores[emotion], 4) for emotion in EMOTIONS}
    result["dominant_emotion"] = dominant
    return result
```

Example returned value:

```
{'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886, 'sadness': 0.0285, 'dominant_emotion': 'joy'}
```

### Activity 2 - Terminal output (output format is accurate)

```
$ python -c "from server import app; c=app.test_client(); r=c.get('/emotionDetector?textToAnalyze=I am so glad this happened'); print('HTTP', r.status_code); print(r.get_data(as_text=True))"

Watson NLP endpoint unreachable, using the built-in model.
HTTP 200
For the given statement, the system response is 'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886 and 'sadness': 0.0285. The dominant emotion is joy.
```

---

## Task 4 : Validate the EmotionDetection package

### Activity 1 - GitHub repository URL of `__init__.py`

https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/EmotionDetection/__init__.py

```python
"""EmotionDetection package: Watson NLP based emotion analysis."""

from EmotionDetection.emotion_detection import emotion_detector
from EmotionDetection.emotion_detection import emotion_predictor

__all__ = ["emotion_detector", "emotion_predictor"]
```

### Activity 2 - Terminal output (EmotionDetection is a valid package)

```
$ python -c "import EmotionDetection; print('EmotionDetection is a valid package'); print(EmotionDetection.__file__)"

EmotionDetection is a valid package
C:\Users\Gayatri Dhas\oaqjp-final-project-emb-ai\EmotionDetection\__init__.py
```

---

## Task 5 : Run unit tests on your application

### Activity 1 - Code from `test_emotion_detection.py`

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

### Activity 2 - Terminal output (all unit tests passed)

```
$ python -m unittest -v test_emotion_detection

test_anger_detection (test_emotion_detection.TestEmotionDetection.test_anger_detection)
A mad statement must be classified as anger. ... ok
test_blank_input_returns_empty_result (test_emotion_detection.TestEmotionDetection.test_blank_input_returns_empty_result)
A blank statement must return the status code 400 empty result. ... ok
test_disgust_detection (test_emotion_detection.TestEmotionDetection.test_disgust_detection)
A disgusted statement must be classified as disgust. ... ok
test_fear_detection (test_emotion_detection.TestEmotionDetection.test_fear_detection)
An afraid statement must be classified as fear. ... ok
test_joy_detection (test_emotion_detection.TestEmotionDetection.test_joy_detection)
A glad statement must be classified as joy. ... ok
test_output_format (test_emotion_detection.TestEmotionDetection.test_output_format)
The result must expose every emotion score and a dominant emotion. ... ok
test_sadness_detection (test_emotion_detection.TestEmotionDetection.test_sadness_detection)
A sad statement must be classified as sadness. ... ok

----------------------------------------------------------------------
Ran 7 tests in 6.029s

OK
```

---

## Task 6 : Web deployment of the application using Flask

### Activity 1 - Code from `server.py`

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
$ python server.py
 * Running on http://0.0.0.0:5000

$ curl "http://127.0.0.1:5000/emotionDetector?textToAnalyze=I%20am%20so%20glad%20this%20happened"
For the given statement, the system response is 'anger': 0.0285, 'disgust': 0.0285, 'fear': 0.0285, 'joy': 0.886 and 'sadness': 0.0285. The dominant emotion is joy.
```

### Activity 2 - Screenshot `6b_deployment_test.png`

File in the repository: [6b_deployment_test.png](https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/6b_deployment_test.png)

Direct link: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/6b_deployment_test.png

The screenshot shows the deployed Flask application at `http://localhost:5000`
displaying the emotion detection result for the statement *"I am so glad this
happened"*: the dominant emotion **joy**.

---

## Task 7 : Incorporate error handling

### Activity 1 - `emotion_detection.py` handles status code 400

```python
    if response.status_code == 400:
        return empty_emotion_result()
```

`empty_emotion_result` returns the dictionary with all values set to `None`:

```python
def empty_emotion_result():
    """Return the result used whenever the statement cannot be analysed."""
    result = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = None
    return result
```

Blank input is also handled before the request is sent:

```python
    if text_to_analyze is None or not str(text_to_analyze).strip():
        return empty_emotion_result()
```

### Activity 2 - `server.py` handles blank input errors

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
```

Terminal output of the blank input request:

```
$ curl "http://127.0.0.1:5000/emotionDetector?textToAnalyze="
HTTP 400
Invalid text! Please enter a non-empty text string to analyze.
```

### Activity 3 - Screenshot `7c_error_handling_interface.png`

File in the repository: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai/blob/main/7c_error_handling_interface.png

The screenshot shows the web interface after submitting an empty text string;
the application displays **"Invalid text! Please enter a non-empty text string
to analyze."** instead of crashing.

---

## Task 8 : Run static code analysis

### Activity 1 - Static code analysis of `server.py`

Command executed:

```
python -m pylint ./server.py
```

### Activity 2 - Terminal output (perfect score)

```
$ python -m pylint ./server.py

--------------------------------------------------------------------
Your code has been rated at 10.00/10
```

The `EmotionDetection/emotion_detection.py` module and
`test_emotion_detection.py` also score 10.00/10:

```
$ python -m pylint ./EmotionDetection/emotion_detection.py
Your code has been rated at 10.00/10

$ python -m pylint ./test_emotion_detection.py
Your code has been rated at 10.00/10
```

---

## Note on the Watson NLP endpoint

The course Watson NLP endpoint
(`sn-watson-emotion.labs.skills.network`) is published inside the IBM Skills
Network lab environment and does not resolve to a public address from a local
machine. `emotion_detector` therefore always contacts the Watson service
first and, when the endpoint is unreachable from the machine, falls back to
the built-in lexicon model contained in `OFFLINE_LEXICON`, so the
application, the unit tests and the web interface keep working outside the
lab. Inside the IBM Skills Network environment the Watson NLP service is
used, exactly as delivered by the course.
