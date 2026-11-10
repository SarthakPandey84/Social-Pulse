"""Audit helper script for Module 3 Methodological Verification."""

from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.nlp.text_preprocessor import TextPreprocessor
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.nlp.relatability_classifier import RelatabilityClassifier

prep = TextPreprocessor()
sent = SentimentAnalyzer()
clf = RelatabilityClassifier(model_path=root_dir / "models" / "relatability_classifier.joblib")

print("=" * 60)
print("1. LANGUAGE DETECTION TESTS")
print("=" * 60)
lang_cases = [
    ("Short English", "So true!"),
    ("Short English 2", "Great job!"),
    ("Hinglish in Latin script", "bhai ye toh meri hi kahani hai bilkul sach bola aapne"),
    ("Devanagari Hindi", "यह बिल्कुल सच है मेरे साथ भी ऐसा ही होता है"),
    ("Mixed Hindi-English", "Bro this is so true yaar, mera bhi same problem hai"),
    ("English with foreign loanword", "The cinematography has a wonderful joie de vivre and aesthetic!"),
    ("Spanish example", "este video me describe perfectamente amigo mio"),
    ("French example", "C'est tellement vrai pour moi aussi dans ma vie."),
    ("German example", "Das kenne ich zu gut aus eigener Erfahrung."),
    ("Emoji-only comment", "🔥🔥❤️"),
    ("Very short comment", "ok")
]

for label, txt in lang_cases:
    res = prep.preprocess(txt)
    print(f"[{label}]")
    print(f"  Input: '{txt}'")
    print(f"  Valid: {res.is_valid} | Status: {res.status.value} | Detected Lang: {res.language}")
    if res.rejection_reason:
        print(f"  Reason: {res.rejection_reason}")

print("\n" + "=" * 60)
print("2. FOUR-WAY SENTIMENT POLARITY TESTS (INCLUDING MIXED)")
print("=" * 60)
sent_cases = [
    ("Clearly Positive", "Amazing video, loved every second of this explanation and great pacing!"),
    ("Clearly Negative", "Terrible advice, worst video I have seen all week, awful quality."),
    ("Neutral Information", "What camera and lens did you use for the opening transition?"),
    ("Mixed Positive & Negative", "I hated this topic at first and felt terrible, but your explanation was amazing and so helpful!"),
    ("Mixed 2 (Contrasting emotions)", "I love the editing and great visuals, but I hate the awful advice.")
]

for label, txt in sent_cases:
    res = sent.analyze_text(txt)
    print(f"[{label}]")
    print(f"  Input: '{txt}'")
    print(f"  Sentiment Label: {res['sentiment_label']} | Polarity: {res['polarity']} | Confidence: {res['sentiment_confidence']}")

print("\n" + "=" * 60)
print("3. RELATABILITY EDGE CASES & LAYER RESOLUTION")
print("=" * 60)
rel_cases = [
    ("Negative but Relatable", "My anxiety has been telling me this all week, I struggle daily."),
    ("Positive but NOT Relatable", "Great video, keep up the awesome content! Such high production quality."),
    ("Personal Experience without 1st-person pronouns", "Story of my life right now, can't even focus for 10 minutes."),
    ("1st-person pronouns without Relatability", "I bought that camera last year and returned it. I prefer Sony."),
    ("Technical question", "What camera did you use to film this video?"),
    ("Sarcasm", "Oh sure, waking up early magically solves rent lol. Groundbreaking advice."),
    ("Personal Disagreement", "I disagree with your second argument and methodology."),
    ("Generic Praise", "Top tier visuals as always! Awesome editing, what a visual treat!"),
    ("Ambiguous / Insufficient Context", "I think this depends on the specific job situation."),
    ("Very short comment", "ok")
]

for label, txt in rel_cases:
    diag = clf.classify_comment(txt)
    print(f"[{label}]")
    print(f"  Input: '{txt}'")
    print(f"  Label: {diag['relatability_label']} | Conf: {diag['relatability_confidence']} | Status: {diag['classification_status']}")
    if diag.get("rejection_reason"):
        print(f"  Reason: {diag['rejection_reason']}")
