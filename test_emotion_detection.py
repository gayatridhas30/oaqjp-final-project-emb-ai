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
