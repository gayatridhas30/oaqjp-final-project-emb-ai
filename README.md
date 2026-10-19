# Emotion Detector - Final Project

## Project Name

**Emotion Detector** - an AI based web application that detects the emotion
conveyed by a text statement with the Watson NLP library and deploys it with
Flask.

This repository is the submission for the Final Project - Emotion Detector of
the IBM course *Developing AI Applications with Python and Flask*.

## Introduction

Emotion detection goes beyond sentiment analysis by extracting more nuanced
emotions such as joy, sadness, anger, disgust and fear from text statements.
The application posts the statement to the Watson NLP `EmotionPredict`
service, receives the emotion scores, formats them and serves the result
through a Flask web interface.

## Project Tasks

| Task | Deliverable |
| --- | --- |
| Task 1 - Clone the project repository | This repository: https://github.com/gayatridhas30/oaqjp-final-project-emb-ai |
| Task 2 - Emotion detection application with the Watson NLP library | `EmotionDetection/emotion_detection.py` |
| Task 3 - Format the output of the application | `emotion_detector` returns the formatted result dictionary and `server.py` renders the formatted sentence |
| Task 4 - Package the application | `EmotionDetection` package with `__init__.py` |
| Task 5 - Unit tests | `test_emotion_detection.py` |
| Task 6 - Web deployment with Flask | `server.py`, `templates/index.html`, `static/mywebscript.js` |
| Task 7 - Error handling | HTTP status code 400 handling in `emotion_detection.py`, blank input handling in `server.py` |
| Task 8 - Static code analysis | `python -m pylint ./server.py` with a perfect score |

## Project Structure

```
oaqjp-final-project-emb-ai/
├── EmotionDetection/
│   ├── __init__.py               # imports the application modules
│   └── emotion_detection.py      # Watson NLP emotion detection logic
├── static/
│   └── mywebscript.js            # front end behaviour
├── templates/
│   └── index.html                # web interface
├── test_emotion_detection.py     # unit tests
├── server.py                     # Flask web server
├── 6b_deployment_test.png        # Task 6 - deployment screenshot
├── 7c_error_handling_interface.png # Task 7 - error handling screenshot
├── SUBMISSION.md                 # complete submission package
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation and execution

```bash
pip install -r requirements.txt
python server.py
```

The application is then available at http://localhost:5000

## Running the unit tests

```bash
python -m unittest -v test_emotion_detection
```

## Static code analysis

```bash
python -m pylint ./server.py
```

## Watson NLP endpoint

The statements are sent to the Watson NLP service:

```
https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict
```

The endpoint is published by the IBM Skills Network environment used by the
course. When that endpoint cannot be reached from the machine running the
application, `emotion_detector` transparently falls back to the built-in
lexicon model contained in `OFFLINE_LEXICON` so that the application, the
unit tests and the web interface keep working; the Watson service is always
contacted first.

## Author

Gayatri Dhas - gayatridhas30
