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
