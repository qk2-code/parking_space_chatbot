import pytest
from guardrails import PII_Detector, PII_Masker, InputValidator, ResponseGuard


def test_pii_detection_and_masking():
    test_text = "Мій номер АА1234АА, звоніть на +38 050 123 4567"

    pii = PII_Detector.scan_for_pii(test_text)
    detected_types = [v for v in pii.values() if v]
    assert len(detected_types) > 0

    masked = PII_Masker.mask_all_pii(test_text)
    assert "АА1234АА" not in masked
    assert "+38 050 123 4567" not in masked


def test_input_validation():
    test_text = "Мій номер АА1234АА, звоніть на +38 050 123 4567"
    validation = InputValidator.validate_input(test_text)

    assert validation['is_valid'] is True
    assert validation['contains_pii'] is True


def test_malicious_input():
    bad_input = "DROP TABLE users; DELETE FROM data"
    bad_validation = InputValidator.validate_input(bad_input)

    assert bad_validation['is_valid'] is False


def test_response_sanitization():
    response = "User АА1234АА called +38 050 123 4567"
    sanitized = ResponseGuard.sanitize_response(response, mask_pii=True)
    if sanitized:
        assert "АА1234АА" not in sanitized
