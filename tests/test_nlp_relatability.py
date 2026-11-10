"""Tests for NLP Sentiment & Relatability Classifier."""

from src.nlp.relatability_classifier import RelatabilityClassifier


def test_relatability_rule_trigger():
    classifier = RelatabilityClassifier()
    assert classifier.predict("Literally me every morning")[0] == "Relatable"
    assert classifier.predict("Felt this in my soul")[0] == "Relatable"
    assert classifier.predict("The video quality is 1080p")[0] == "Neutral"


def test_train_and_predict():
    classifier = RelatabilityClassifier()
    texts = [
        "so true literally me",
        "i feel this so much",
        "random post with no relation",
        "what is the camera model"
    ]
    labels = ["Relatable", "Relatable", "Neutral", "Neutral"]
    classifier.fit(texts, labels)
    preds = classifier.predict(["so true this happened to me", "what is the camera"])
    assert preds[0] == "Relatable"
    assert preds[1] == "Neutral"

