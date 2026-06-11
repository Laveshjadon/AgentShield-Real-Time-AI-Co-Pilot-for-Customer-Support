"""
just making sure the pii redaction actually works.
don't want any sensitive info leaking.
"""
import pytest
import time
from analysis.pii_service import PIIService

@pytest.fixture(scope="module")
def pii_service():
    return PIIService()

def test_redact_email(pii_service):
    text = "Contact me at test@example.com please."
    result = pii_service.redact(text)
    assert "[EMAIL_ADDRESS_REDACTED]" in result
    assert "test@example.com" not in result

def test_redact_phone(pii_service):
    text = "My number is 555-019-2834."
    result = pii_service.redact(text)
    assert "[PHONE_NUMBER_REDACTED]" in result
    assert "555-019-2834" not in result

def test_redact_aadhaar(pii_service):
    text = "Aadhaar is 1234 5678 9123."
    result = pii_service.redact(text)
    assert "[AADHAAR_REDACTED]" in result
    assert "1234 5678 9123" not in result

def test_redact_pan(pii_service):
    text = "My PAN is ABCDE1234F"
    result = pii_service.redact(text)
    assert "[PAN_REDACTED]" in result
    assert "ABCDE1234F" not in result

def test_redact_bank_account(pii_service):
    text = "Account number 1234567890123."
    result = pii_service.redact(text)
    assert "[BANK_ACCOUNT_REDACTED]" in result
    assert "1234567890123" not in result

def test_redact_multiple(pii_service):
    text = "Email test@test.com and PAN ABCDE1234F."
    result = pii_service.redact(text)
    assert "[EMAIL_ADDRESS_REDACTED]" in result
    assert "[PAN_REDACTED]" in result

@pytest.mark.asyncio
async def test_aredact_performance(pii_service):
    text = "Hi, my name is John and my email is john@gmail.com. My PAN is XXXXX1234Y and phone is 999-999-9999."
    
    
    await pii_service.aredact(text)
    
    
    t0 = time.perf_counter()
    result = await pii_service.aredact(text)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    assert "[EMAIL_ADDRESS_REDACTED]" in result
    
    assert latency_ms < 100, f"Redaction took {latency_ms:.1f}ms, which is > 100ms"
