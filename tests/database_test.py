import pytest
from datetime import datetime, timedelta
import random
from database import (
    init_database, ReservationManager, PricingManager,
    AvailabilityManager, AuditLog
)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_database()
    PricingManager.init_default_rates()


def test_pricing_and_rates():
    rates = PricingManager.get_active_rates()
    assert rates is not None

    cost = PricingManager.calculate_parking_cost(120)
    assert cost > 0
    assert isinstance(cost, (int, float))


def test_reservation_lifecycle():
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    unique_time = f"{10 + random.randint(0, 8)}:{30 + random.randint(0, 20)}"

    res_id = ReservationManager.create_reservation(
        customer_name="Test User",
        license_plate="ТЕ1234ТЕ",
        reservation_date=tomorrow,
        reservation_time=unique_time,
        duration_hours=2
    )
    assert res_id is not None, "Резервація не була створена (можливо, дублікат)"

    res = ReservationManager.get_reservation(res_id)
    assert res is not None
    assert res.customer_name == "Test User"
    assert res.license_plate == "ТЕ1234ТЕ"


def test_audit_logging():
    AuditLog.log_interaction(
        action="test_interaction",
        user_input="Test input",
        bot_response="Test response",
        contains_pii=False
    )

    logs = AuditLog.get_interaction_logs(limit=5)
    assert len(logs) > 0
    assert isinstance(logs, list)
