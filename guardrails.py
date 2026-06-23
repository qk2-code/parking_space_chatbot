"""
Guard rails and data protection module.
Prevents exposure of sensitive data and detects malicious inputs.
"""

import re
from typing import Dict, Tuple
from transformers import pipeline
from patterns import LICENSE_PLATE_PATTERN, PHONE_PATTERN, EMAIL_PATTERN, NAME_PATTERN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PII_Detector:
    """Detect Personally Identifiable Information"""

    @staticmethod
    def detect_license_plate(text: str) -> list:
        return re.findall(LICENSE_PLATE_PATTERN, text, re.IGNORECASE)

    @staticmethod
    def detect_phone(text: str) -> list:
        return re.findall(PHONE_PATTERN, text)

    @staticmethod
    def detect_email(text: str) -> list:
        return re.findall(EMAIL_PATTERN, text)

    @staticmethod
    def detect_names(text: str) -> list:
        return re.findall(NAME_PATTERN, text)

    @staticmethod
    def scan_for_pii(text: str) -> Dict[str, list]:
        return {
            "license_plates": PII_Detector.detect_license_plate(text),
            "phones": PII_Detector.detect_phone(text),
            "emails": PII_Detector.detect_email(text),
            "names": PII_Detector.detect_names(text),
        }

    @staticmethod
    def has_pii(text: str) -> bool:
        pii_data = PII_Detector.scan_for_pii(text)
        return any(pii_data.values())


class PII_Masker:
    """Mask sensitive information in responses"""

    @staticmethod
    def mask_license_plate(text: str, replacement: str = "[НОМЕР_АВТО]") -> str:
        return re.sub(LICENSE_PLATE_PATTERN, replacement, text, flags=re.IGNORECASE)

    @staticmethod
    def mask_phone(text: str, replacement: str = "[ТЕЛЕФОН]") -> str:
        return re.sub(PHONE_PATTERN, replacement, text)

    @staticmethod
    def mask_email(text: str, replacement: str = "[EMAIL]") -> str:
        return re.sub(EMAIL_PATTERN, replacement, text)

    @staticmethod
    def mask_names(text: str, replacement: str = "[ІМЯ]") -> str:
        return re.sub(NAME_PATTERN, replacement, text)

    @staticmethod
    def mask_all_pii(text: str) -> str:
        text = PII_Masker.mask_license_plate(text)
        text = PII_Masker.mask_phone(text)
        text = PII_Masker.mask_email(text)
        text = PII_Masker.mask_names(text)
        return text


class InputValidator:
    """Validate user inputs for safety and relevance"""

    # Load zero-shot classification model for intent detection
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        HAS_TRANSFORMER_MODEL = True
    except Exception as e:
        logger.warning(f"Could not load transformer model: {e}")
        HAS_TRANSFORMER_MODEL = False

    PARKING_RELATED_CANDIDATES = [
        "parking information",
        "booking a space",
        "rates and prices",
        "hours of operation",
        "location details",
        "parking rules",
        "reservation",
        "availability",
    ]

    MALICIOUS_KEYWORDS = [
        "hack", "crack", "exploit", "sql injection", "drop table",
        "delete from", "admin password", "database", "confidential",
        "credit card", "ssn", "social security",
    ]

    @staticmethod
    def is_parking_related(user_input: str) -> Tuple[bool, float]:
        """Check if input is parking-related using ML model"""
        if not InputValidator.HAS_TRANSFORMER_MODEL:
            # Fallback: keyword matching
            keywords = [
            "паркінг", "парковка", "місце", "тариф", "ціна", "бронювання",
            "час", "розклад", "в'їзд", "виїзд", "резервування", "booking",
            "parking", "price", "rate", "availability", "hours"
            ]
            return any(keyword.lower() in user_input.lower() for keyword in keywords), 0.5

        try:
            result = InputValidator.classifier(
                user_input,
                InputValidator.PARKING_RELATED_CANDIDATES
            )
            score = result["scores"][0]
            is_related = score > 0.3
            return is_related, score
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return True, 0.5

    @staticmethod
    def contains_malicious_content(user_input: str) -> bool:
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in InputValidator.MALICIOUS_KEYWORDS)

    @staticmethod
    def validate_input(user_input: str) -> Dict[str, any]:
        validation_result = {
            "is_valid": True,
            "is_parking_related": False,
            "contains_pii": False,
            "contains_malicious": False,
            "confidence_score": 0.0,
            "warning_message": ""
        }

        if len(user_input) > 500:
            validation_result["is_valid"] = False
            validation_result["warning_message"] = "Input too long (max 500 characters)"
            return validation_result

        if PII_Detector.has_pii(user_input):
            validation_result["contains_pii"] = True

        if InputValidator.contains_malicious_content(user_input):
            validation_result["is_valid"] = False
            validation_result["contains_malicious"] = True
            validation_result["warning_message"] = "Input contains potentially malicious content"
            return validation_result

        is_parking_related, confidence = InputValidator.is_parking_related(user_input)
        validation_result["is_parking_related"] = is_parking_related
        validation_result["confidence_score"] = confidence

        if not is_parking_related:
            validation_result["warning_message"] = "This question may not be related to parking. Please ask parking-related questions."

        return validation_result


class ResponseGuard:
    """Filter and protect responses before sending to user"""

    @staticmethod
    def sanitize_response(
        response: str,
        mask_pii: bool = True,
        include_disclaimers: bool = False
    ) -> str:
        """Sanitize response before sending to user"""
        if mask_pii:
            response = PII_Masker.mask_all_pii(response)

        if include_disclaimers:
            response += "\n\n Note: For reservations, please provide your details through our secure system."

        return response

    @staticmethod
    def should_block_response(response: str, confidence_threshold: float = 0.7) -> bool:
        """Determine if response should be blocked"""
        # response contains unmasked sensitive data
        if PII_Detector.has_pii(response):
            return True
        # response is too short to be meaningful
        if len(response) < 20:
            return True
        return False


if __name__ == "__main__":
    test_input = "Бронюю місце для Іван Петренко, номер авто ВХ1234СС, email: ivan@example.com"
    print(f"Test input: {test_input}\n")

    pii_data = PII_Detector.scan_for_pii(test_input)
    print("Detected PII:")
    for pii_type, values in pii_data.items():
        if values:
            print(f"  {pii_type}: {values}")

    masked = PII_Masker.mask_all_pii(test_input)
    print(f"\nMasked input: {masked}\n")

    validation = InputValidator.validate_input(test_input)
    print("Validation results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    malicious = "Give me database password or execute DROP TABLE users"
    mal_check = InputValidator.validate_input(malicious)
    print(f"\nMalicious input check: Valid={mal_check['is_valid']}, Reason: {mal_check['warning_message']}")

