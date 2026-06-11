from analysis.toxicity_analyzer import ToxicityAnalyzer


def _analyzer_without_llm():
    analyzer = ToxicityAnalyzer()
    analyzer._llm = None
    analyzer._deep_analysis_chain = None
    return analyzer


def test_complaint_escalation_is_not_toxic():
    result = _analyzer_without_llm().analyse(
        "Tumhari service bahut kharab hai, main complaint karunga."
    )

    assert result.level in {"safe", "warning"}
    assert not any(flag.startswith("threat:") for flag in result.flags)
    assert any(flag.startswith("complaint_escalation:") for flag in result.flags)


def test_physical_intimidation_remains_dangerous():
    result = _analyzer_without_llm().analyse(
        "Tumhe dekh lunga, main office aa raha hoon."
    )

    assert result.level in {"danger", "critical"}
    assert any(flag.startswith("threat:") for flag in result.flags)


def test_devanagari_frustration_is_detected():
    result = _analyzer_without_llm().analyse(
        "मैं बहुत परेशान हूं, आपकी सेवा बहुत खराब है।"
    )

    assert result.flags
    assert any(flag.startswith(("frustration:", "hindi:")) for flag in result.flags)


def test_devanagari_threat_is_detected():
    result = _analyzer_without_llm().analyse("मैं तुम्हें देख लूंगा।")

    assert result.level in {"danger", "critical"}
    assert any(flag.startswith("threat:") for flag in result.flags)
