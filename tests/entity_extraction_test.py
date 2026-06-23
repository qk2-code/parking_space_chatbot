import pytest
from entity_extraction import ReservationEntityExtractor

def test_ukrainian_extraction():
    ua_text = "Бронюю на завтра о 14:30, номер АА1234АА, Іван Петренко, на 2 години"
    entities = ReservationEntityExtractor.extract_all_entities(ua_text)
    is_valid, errors = ReservationEntityExtractor.validate_entities(entities)

    assert is_valid is True
    assert errors == [] or len(errors) == 0
    assert entities.get('license_plate') == "АА1234АА"
    assert "Іван" in str(entities.get('customer_names'))

def test_incomplete_data_detection():
    incomplete = "бронюю на завтра"
    entities = ReservationEntityExtractor.extract_all_entities(incomplete)
    is_valid, errors = ReservationEntityExtractor.validate_entities(entities)

    assert is_valid is False
    assert len(errors) > 0