import sys
from datetime import datetime, timedelta

def test_database():
    try:
        from database import (
            init_database, ReservationManager, PricingManager,
            AvailabilityManager, AuditLog
        )

        init_database()
        PricingManager.init_default_rates()
        rates = PricingManager.get_active_rates()
        cost = PricingManager.calculate_parking_cost(120)
        print(f"Cost calculation: 120 min = {cost} грн")
        tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        import random
        unique_time = f"{10 + random.randint(0, 8)}:{30 + random.randint(0, 20)}"
        res_id = ReservationManager.create_reservation(
            customer_name="Test User",
            license_plate="ТЕ1234ТЕ",
            reservation_date=tomorrow,
            reservation_time=unique_time,
            duration_hours=2
        )
        print(f"Reservation created: #{res_id}")

        if res_id:
            res = ReservationManager.get_reservation(res_id)
            if res:
                print(f"Retrieved reservation for: {res.customer_name}")
            else:
                print(f"Could not retrieve reservation #{res_id}")
                return False
        else:
            print(f"Reservation creation returned None (likely duplicate)")
            return False

        AuditLog.log_interaction(
            action="test_interaction",
            user_input="Test input",
            bot_response="Test response",
            contains_pii=False
        )
        print("Audit log recorded")

        logs = AuditLog.get_interaction_logs(limit=5)
        print(f"Retrieved {len(logs)} audit logs")

        return True
    except Exception as e:
        print(f"Database test failed: {e}")
        return False


def test_guardrails():
    try:
        from guardrails import (
            PII_Detector, PII_Masker, InputValidator, ResponseGuard
        )

        test_text = "Мій номер АА1234АА, звоніть на +38 050 123 4567"
        pii = PII_Detector.scan_for_pii(test_text)
        print(f"PII detected: {len([v for v in pii.values() if v])} types")

        masked = PII_Masker.mask_all_pii(test_text)
        print(f"Text masked: {masked[:50]}...")

        validation = InputValidator.validate_input(test_text)
        print(f"Input valid: {validation['is_valid']}")
        print(f"Contains PII: {validation['contains_pii']}")
        print(f"Is parking related: {validation['is_parking_related']}")

        bad_input = "DROP TABLE users; DELETE FROM data"
        bad_validation = InputValidator.validate_input(bad_input)
        print(f"Malicious input detected: {not bad_validation['is_valid']}")

        response = "User АА1234АА called +38 050 123 4567"
        safe_response = ResponseGuard.sanitize_response(response, mask_pii=True)
        print(f"Response sanitized")

        return True
    except Exception as e:
        print(f"Guard rails test failed: {e}")
        return False


def test_entity_extraction():
    try:
        from entity_extraction import ReservationEntityExtractor

        ua_text = "Бронюю на завтра о 14:30, номер АА1234АА, Іван Петренко, на 2 години"
        entities_ua = ReservationEntityExtractor.extract_all_entities(ua_text)
        is_valid_ua, errors_ua = ReservationEntityExtractor.validate_entities(entities_ua)

        print(f"Ukrainian extraction: Valid={is_valid_ua}")
        print(f"Plate: {entities_ua.get('license_plate')}")
        print(f"Names: {entities_ua.get('customer_names')}")
        print(f"Date: {entities_ua.get('date')}")
        print(f"Time: {entities_ua.get('time')}")

        en_text = "Booking a spot for tomorrow at 15:00, my plate is XX5678XX, name is John Smith"
        entities_en = ReservationEntityExtractor.extract_all_entities(en_text)
        is_valid_en, errors_en = ReservationEntityExtractor.validate_entities(entities_en)

        print(f"English extraction: Valid={is_valid_en}")

        incomplete = "бронюю на завтра"
        entities_incomplete = ReservationEntityExtractor.extract_all_entities(incomplete)
        is_valid_incomplete, errors_incomplete = ReservationEntityExtractor.validate_entities(entities_incomplete)

        print(f"Incomplete data detection: Valid={is_valid_incomplete}, Errors={len(errors_incomplete)}")

        return True
    except Exception as e:
        print(f"Entity extraction test failed: {e}")
        return False


def test_evaluation():

    try:
        from evaluation import EvaluationMetrics, ResponseAccuracy, PerformanceTracker

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5", "doc6"]

        recall = EvaluationMetrics.recall_at_k(retrieved, relevant, k=3)
        precision = EvaluationMetrics.precision_at_k(retrieved, relevant, k=3)
        mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved, relevant)
        f1 = EvaluationMetrics.f1_score(precision, recall)

        print(f"Metrics calculated:")
        print(f"Recall@3: {recall}")
        print(f"Precision@3: {precision}")
        print(f"MRR: {mrr}")
        print(f"F1 Score: {f1}")

        response = "The parking is open 24/7 with a technical break from 3:00 to 3:15 AM"
        keywords = ["24/7", "3:00"]
        is_accurate, matches = ResponseAccuracy.keyword_match(response, keywords, min_matches=2)

        print(f"Response accuracy: {is_accurate} (matches={matches})")

        tracker = PerformanceTracker()
        tracker.log_query_metric(
            query_text="Test query",
            response_text="Test response",
            response_time_ms=150.5,
            recall_k3=0.85,
            precision_k3=0.90
        )
        print(f"Metric logged to database")

        summary = tracker.get_performance_summary()
        print(f"Performance summary retrieved: {summary['total_queries']} queries")

        return True
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        return False


def print_summary(results):
    tests = [
        ("Database", results[0]),
        ("Guard Rails", results[1]),
        ("Entity Extraction", results[2]),
        ("Evaluation Metrics", results[3]),
    ]

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for name, result in tests:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! System is ready to use.")
    else:
        print(f"\n{total - passed} test(s) failed. Check the errors above.")

    return passed == total


def main():
    results = [
        test_database(),
        test_guardrails(),
        test_entity_extraction(),
        test_evaluation(),
    ]

    success = print_summary(results)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)