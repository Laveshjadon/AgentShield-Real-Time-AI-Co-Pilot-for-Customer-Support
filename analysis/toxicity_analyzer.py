"""
 Toxicity Analyser
Scores customer frustration, abuse, and threats.

The analyser uses three layers:
  Layer 1: simple regex/keyword matching (english + hindi)
  Layer 2: some basic sentiment scoring
  Layer 3: calls out to groq LLM if the first two layers find something weird

"""

import re
import time
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import Settings
from config.logger import get_logger

logger = get_logger("analysis.toxicity")
settings = Settings()





@dataclass
class ToxicityResult:
    text: str
    score: float          
    level: str            
    flags: list           
    is_toxic: bool        
    alert_message: str    
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = time.time()







THREAT_KEYWORDS_EN = [
    "i will destroy", "you'll regret", "i will kill", "you're dead",
    "coming to your office", "hurt you"
]

THREAT_KEYWORDS_HI = [
    "court", "police", "case karunga", "case karungi", "complaint karunga",
    "dekhta hun", "dekhti hun", "chhod nahi", "maf nahi", "barbad kar dunga",
    "lawsuit", "court main", "thana", "report karunga", "report karungi",
    "main complaint karunga", "main complaint karungi", "main report karunga",
    "main report karungi", "police me report", "police mein report",
    "consumer court jaunga", "consumer court jaungi", "case file karunga",
    "case file karungi", "tumhe dekh lunga", "tumhe dekh lungi",
    "company ko report karunga", "company ko report karungi",
    "social media pe dalunga", "social media pe dalungi",
    "sabko bata dunga", "sabko bata dungi",
    "शिकायत करूंगा", "शिकायत करूंगी", "कंप्लेंट करूंगा", "कंप्लेंट करूंगी",
    "पुलिस", "कोर्ट", "केस करूंगा", "केस करूंगी", "रिपोर्ट करूंगा",
    "रिपोर्ट करूंगी", "बर्बाद कर दूंगा", "बर्बाद कर दूंगी", "छोडूंगा नहीं",
    "छोड़ूंगा नहीं", "देख लूंगा", "देख लूंगी"
]


ABUSE_PATTERNS_EN = [
    r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\buseless\b",
    r"\bworthless\b", r"\bincompetent\b", r"\bscammer\b", r"\bfraud\b",
    r"\bliar\b", r"\bthief\b", r"\bcriminal\b", r"\bscam\b"
]

ABUSE_PATTERNS_HI = [
    r"\bbewakoof\b", r"\bullu\b", r"\bkamina\b", r"\bcheat\b",
    r"\bdhokha\b", r"\bfraudi\b", r"\bbekaar\b", r"\bganwar\b",
    r"\bfraud company\b", r"\bbakwas service\b", r"\bghatiya\b",
    r"\bghatiya service\b", r"\bnikamme\b", r"\bnalayak\b",
    r"\bjhoothe\b", r"\bchor\b", r"\blooter[a-z]*\b",
    r"बेकार", r"बकवास", r"धोखा", r"फ्रॉड", r"चोर", r"झूठे",
    r"निकम्मे", r"घटिया", r"नालायक"
]


FRUSTRATION_SIGNALS = [
    "never again", "cancel my account", "want my money back", "this is ridiculous",
    "waste of time", "no one helps", "worst service", "terrible", "horrible",
    "disgusting", "pathetic", "what a joke", "fed up", "done with you",
    "koi kaam ka nahi", "bakwas", "bekar", "time waste", "paisa wapas",
    "paise wapas", "paisa refund", "refund do", "refund chahiye",
    "bahut pareshan", "main pareshan hoon", "main pareshaan hoon",
    "koi help nahi", "koi madad nahi", "help nahi kar rahe",
    "madad nahi kar rahe", "service bahut kharab", "bahut kharab service",
    "tumhari service kharab", "tumhari service bekaar",
    "ye kya mazak hai", "yeh kya mazak hai", "mujhe gussa aa raha hai",
    "main tang aa gaya", "main tang aa gayi", "ab bardasht nahi",
    "पैसा वापस", "बहुत खराब", "खराब सेवा", "मदद नहीं", "कोई मदद नहीं",
    "समय बर्बाद", "टाइम वेस्ट", "तंग आ गया", "तंग आ गई", "बहुत परेशान",
    "ये क्या मजाक है", "यह क्या मजाक है", "कैंसल कर दूंगा", "कैंसल कर दूंगी"
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

COMPLAINT_ESCALATION_SIGNALS = [
    "i will sue", "i'll sue", "lawsuit", "lawyer", "legal action",
    "going to report", "file a complaint", "report this", "consumer court",
    "complaint karunga", "complaint karungi", "report karunga",
    "report karungi", "case karunga", "case karungi", "police mein report",
    "police me report", "rbi complaint", "ombudsman",
]





def _score_to_level(score: float) -> str:
    if score < 0.25:
        return "safe"
    elif score < 0.50:
        return "warning"
    elif score < 0.75:
        return "danger"
    else:
        return "critical"


def _build_alert(level: str, flags: list) -> str:
    if level == "safe":
        return ""
    elif level == "warning":
        return f"Customer showing frustration. Triggers: {', '.join(flags[:2])}"
    elif level == "danger":
        return f"ALERT: Hostile language detected! Triggers: {', '.join(flags[:3])}"
    else:
        return f"CRITICAL: Abusive/threatening language! Consider escalating. Triggers: {', '.join(flags[:3])}"





class ToxicityAnalyzer:

    def __init__(self):
        logger.info("Initializing ToxicityAnalyzer...")

        
        self._abuse_patterns_en = [re.compile(p, re.IGNORECASE) for p in ABUSE_PATTERNS_EN]
        self._abuse_patterns_hi = [re.compile(p, re.IGNORECASE) for p in ABUSE_PATTERNS_HI]

        
        try:
            self._llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0,
            )
            self._deep_analysis_chain = self._build_deep_analysis_chain()
            logger.info("Groq LLM connected for deep toxicity analysis.")
        except Exception as e:
            logger.warning(f"Groq not available for deep analysis: {e}")
            self._llm = None
            self._deep_analysis_chain = None

        logger.info("ToxicityAnalyzer ready.")

    def _build_deep_analysis_chain(self):
        """creates the langchain pipeline to ask the LLM about the text"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a toxicity detection system for a call center.
Analyse the given customer statement and respond ONLY with a JSON object.

Respond with this exact format:
{{"score": 0.0, "is_abusive": false, "is_threatening": false, "summary": "calm"}}

Where:
- score: float from 0.0 (calm) to 1.0 (extremely toxic/threatening)
- is_abusive: true if personal insults or profanity are present
- is_threatening: true only for credible physical harm or intimidation threats.
  Legal action and regulator or consumer complaints are escalation intent, not
  toxicity by themselves.
- summary: one word describing the tone

ONLY return the JSON. No other text."""),
            ("user", "Customer said: {text}")
        ])
        return prompt | self._llm | StrOutputParser()

    def _layer1_keyword_scan(self, text: str) -> tuple[float, list]:
        """
        Layer 1 performs fast keyword and pattern checks.
        returns score and what it found.
        """
        text_lower = text.lower()
        flags = []
        score = 0.0

        complaint_signals = [
            signal for signal in COMPLAINT_ESCALATION_SIGNALS
            if signal in text_lower
        ]
        flags.extend(
            f"complaint_escalation:{signal}" for signal in complaint_signals
        )

        
        
        for kw in THREAT_KEYWORDS_EN + THREAT_KEYWORDS_HI:
            if kw in text_lower and not any(
                kw in complaint_signal or complaint_signal in kw
                for complaint_signal in complaint_signals
            ):
                flags.append(f"threat:{kw}")
                score += 0.55

        
        for pattern in self._abuse_patterns_en + self._abuse_patterns_hi:
            if pattern.search(text_lower):
                flags.append(f"abuse:{pattern.pattern}")
                score += 0.25

        
        frustration_hits = 0
        for signal in FRUSTRATION_SIGNALS:
            if signal in text_lower:
                flags.append(f"frustration:{signal}")
                frustration_hits += 1
                score += 0.10

        if frustration_hits >= 2:
            score += 0.08

        
        
        if DEVANAGARI_RE.search(text):
            if any(term in text for term in ["गुस्सा", "परेशान", "खराब", "शिकायत", "रिपोर्ट", "कंप्लेंट"]):
                flags.append("hindi:angry_or_complaint")
                score += 0.18

        return min(score, 1.0), flags

    def _layer2_deep_analysis(self, text: str) -> float:
        """
        layer 2: asking the LLM what it thinks.
        returns score from 0 to 1.
        This runs only when Layer 1 detects a signal, limiting external calls.
        """
        if not self._deep_analysis_chain:
            return 0.0

        try:
            import json
            response = self._deep_analysis_chain.invoke({"text": text})
            
            clean = response.strip().strip("```json").strip("```").strip()
            data = json.loads(clean)
            return float(data.get("score", 0.0))
        except Exception as e:
            logger.debug(f"Deep analysis parse error: {e}")
            return 0.0

    def analyse(self, text: str) -> ToxicityResult:
        """
        main entry point. takes text, returns the result object.
        """
        if not text or len(text.strip()) < 2:
            return ToxicityResult(
                text=text, score=0.0, level="safe",
                flags=[], is_toxic=False, alert_message=""
            )

        
        l1_score, flags = self._layer1_keyword_scan(text)

        
        final_score = l1_score
        if l1_score > 0.2 and self._deep_analysis_chain:
            l2_score = self._layer2_deep_analysis(text)
            
            final_score = (l1_score * 0.4) + (l2_score * 0.6)

        level = _score_to_level(final_score)
        is_toxic = final_score >= 0.25
        alert = _build_alert(level, flags)

        result = ToxicityResult(
            text=text,
            score=round(final_score, 3),
            level=level,
            flags=flags,
            is_toxic=is_toxic,
            alert_message=alert
        )

        if is_toxic:
            logger.warning(f"Toxicity detected [{level}] score={final_score:.2f}: '{text[:60]}'")

        return result


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AgentShield - Toxicity Analyser Test")
    print("="*55)

    analyser = ToxicityAnalyzer()

    test_cases = [
        ("I'd like to know more about your refund policy.", "SAFE"),
        ("This is the worst service I have ever experienced.", "FRUSTRATION"),
        ("You are completely useless and a total scammer!", "ABUSE"),
        ("I will sue your company and take you to court!", "THREAT"),
        ("Yeh bekaar service hai, paisa wapas karo!", "HINDI FRUSTRATION"),
        ("Case karunga tumhare against, fraud company!", "HINDI THREAT"),
    ]

    for text, label in test_cases:
        result = analyser.analyse(text)
        print(f"\n  [{label}]")
        print(f"  Text   : {text}")
        print(f"  Score  : {result.score} | Level: {result.level.upper()}")
        if result.alert_message:
            print(f"  Alert  : {result.alert_message}")

    print("\n" + "="*55 + "\n")
