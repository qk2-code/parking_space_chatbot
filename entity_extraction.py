"""
Entity extraction for parking reservations.
Extracts: name, license plate, date, time from user input.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json

class ReservationEntityExtractor:
    """Extract reservation-related entities from user input"""

    LICENSE_PLATE_PATTERNS = [
        r'\b[A-Z]{2}\s?\d{4}\s?[A-Z]{2}\b',  # XX1234XX
        r'\b[А-Я]{2}\s?\d{4}\s?[А-Я]{2}\b',  # АА1234АА
        r'\b\d{4}\s?[A-Z]{2}\s?\d{2}\b',     # 1234XX99
    ]

    DATE_PATTERNS = [
        r'(\d{2})\.(\d{2})\.(\d{4})',        # DD.MM.YYYY
        r'(\d{4})-(\d{2})-(\d{2})',          # YYYY-MM-DD
        r'(\d{2})/(\d{2})/(\d{4})',          # DD/MM/YYYY
        r'(завтра|tomorrow|сьогодні|today|завтра|the day after tomorrow)',  # relative dates
    ]

    TIME_PATTERNS = [
        r'(\d{1,2}):(\d{2})',                 # HH:MM or H:MM
        r'(\d{1,2})\.(\d{2})',                # HH.MM
        r'(\d{1,2})\s*:?\s*на\s*ранку',       # Xo in the morning
        r'(\d{1,2})\s*:?\s*увечері',          # Xo in the evening
    ]

    NAME_PATTERN = r'\b([А-Яа-яЇїЄєІі][а-яїєіґ]+)(?:\s+([А-Яа-яЇїЄєІі][а-яїєіґ]+))?\b'

    @staticmethod
    def extract_license_plate(text: str) -> Optional[str]:
        """Extract license plate from text"""
        for pattern in ReservationEntityExtractor.LICENSE_PLATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).upper().replace(" ", "")
        return None

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Extract date from text, return as YYYY-MM-DD"""
        today = datetime.today()

        # relative dates
        text_lower = text.lower()
        if any(word in text_lower for word in ['завтра', 'tomorrow']):
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if any(word in text_lower for word in ['сьогодні', 'today', 'хай', 'сегодня']):
            return today.strftime("%Y-%m-%d")
        if any(word in text_lower for word in ['послідзавтра', 'day after tomorrow']):
            return (today + timedelta(days=2)).strftime("%Y-%m-%d")

        # absolute date patterns
        for pattern in ReservationEntityExtractor.DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    if len(match.groups()) == 3:
                        groups = match.groups()
                        # Determine format
                        if int(groups[0]) > 31:  # YYYY-MM-DD
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        elif int(groups[2]) > 31:  # DD.MM.YYYY or DD/MM/YYYY
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        else:
                            continue

                        date_obj = datetime(year, month, day)
                        if date_obj >= today:
                            return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        return None

    @staticmethod
    def extract_time(text: str) -> Optional[str]:
        """Extract time from text, return as HH:MM"""
        match = re.search(ReservationEntityExtractor.TIME_PATTERNS[0], text)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            if 0 <= hours < 24 and 0 <= minutes < 60:
                return f"{hours:02d}:{minutes:02d}"
        return None

    @staticmethod
    def extract_names(text: str) -> list:
        """Extract person names (first and last name)"""
        # Remove license plates first to avoid confusion
        cleaned_text = re.sub(r'\b[A-Z]{2}\s?\d{4}\s?[A-Z]{2}\b', '', text, flags=re.IGNORECASE)

        # Common Ukrainian filler words and verbs to exclude from name extraction
        exclude_words = {
            'на', 'о', 'ок', 'до', 'з', 'що', 'бронюю', 'броню', 'забронювати',
            'хочу', 'завтра', 'року', 'години', 'хвилин', 'для',
            'мене', 'номер', 'місце', 'цифра', 'дата', 'час', 'тип', 'вигляд',
            'бажаю', 'зробити', 'добуті', 'хочеш', 'готую'
        }

        # capitalized word followed by capitalized word
        name_pattern = r'\b([А-ЯЇЄІЙґ][а-яїєіґ]+)\s+([А-ЯЇЄІЙґ][а-яїєіґ]+)\b'
        names = re.findall(name_pattern, cleaned_text)

        if names:
            valid_names = []
            for first, last in names:
                # check if neither word is in the exclude list AND both have reasonable length
                if (first.lower() not in exclude_words and
                    last.lower() not in exclude_words and
                    len(first) > 2 and len(last) > 2):
                    valid_names.append(f"{first} {last}")
            if valid_names:
                return valid_names

        return []

    @staticmethod
    def extract_duration(text: str) -> Optional[int]:
        """Extract parking duration in hours"""
        patterns = [
            (r'(\d+)\s*(?:год|hour|hrs|h)(?:и?н)?', 1),
            (r'(\d+)\s*(?:хв|хвилин|min|minutes)', 0.0167),
            (r'(\d+)\s*(?:доб|day|days)', 24),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                duration = int(match.group(1)) * multiplier
                return int(duration) if duration > 0 else None

        return None

    @staticmethod
    def extract_all_entities(text: str) -> Dict[str, any]:
        """Extract all reservation-related entities"""
        return {
            "license_plate": ReservationEntityExtractor.extract_license_plate(text),
            "date": ReservationEntityExtractor.extract_date(text),
            "time": ReservationEntityExtractor.extract_time(text),
            "customer_names": ReservationEntityExtractor.extract_names(text),
            "duration_hours": ReservationEntityExtractor.extract_duration(text) or 1,
        }

    @staticmethod
    def validate_entities(entities: Dict) -> Tuple[bool, list]:
        """Validate extracted entities"""
        errors = []

        if not entities.get("license_plate"):
            errors.append("License plate not found. Please provide your vehicle's license plate.")

        if not entities.get("date"):
            errors.append("Date not found. Please specify a date (e.g., 'tomorrow', '15.03.2026').")

        if not entities.get("time"):
            errors.append("Time not found. Please specify a time (e.g., '14:30', '9:00').")

        if not entities.get("customer_names"):
            errors.append("Name not found. Please provide your full name.")

        if entities.get("duration_hours", 0) <= 0:
            errors.append("Invalid duration. Please specify how many hours you need.")

        return len(errors) == 0, errors

    @staticmethod
    def format_reservation_summary(entities: Dict) -> str:
        """Format extracted entities into a readable summary"""
        name = " ".join(entities.get("customer_names", ["Unknown"])) if entities.get("customer_names") else "Unknown"
        plate = entities.get("license_plate") or "Not provided"
        date = entities.get("date") or "Not provided"
        time = entities.get("time") or "Not provided"
        duration = entities.get("duration_hours", 1)

        summary = f"""
        Reservation Summary:
        - Name: {name}
        - License Plate: {plate}
        - Date: {date}
        - Time: {time}
        - Duration: {duration} hours
        """
        return summary.strip()


if __name__ == "__main__":
    test_cases = [
        "Привіт! Я хочу забронювати місце паркування на завтра о 14:30. Мій номер: АА1234АА, зовуть мене Іван Петренко, на 2 години",
        "Book a spot for today at 10:00. My car is XX1234XX. I'm John Smith.",
        "Завтра на 15.03.2026 о 19:00 бронював місце. Це для Марія Іванівна, номер ВХ9999СС",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test}\n")
        entities = ReservationEntityExtractor.extract_all_entities(test)
        is_valid, errors = ReservationEntityExtractor.validate_entities(entities)

        print(f"Extracted entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")
        print(f"Valid: {is_valid}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(ReservationEntityExtractor.format_reservation_summary(entities))