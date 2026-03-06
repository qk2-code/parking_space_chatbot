import time
import logging
from evaluation import ResponseAccuracy, PerformanceTracker
from main import ParkingChatbot

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class PerformanceTest:
    def __init__(self):
        self.chatbot = None
        self.tracker = PerformanceTracker()
        self.results = []

    def initialize_chatbot(self):
        try:
            self.chatbot = ParkingChatbot()
            print("Chatbot initialized successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Could not initialize chatbot: {e}")
            return False

    def test_retrieval_metrics(self):

        test_cases = [
            {
                "question": "Які є тарифи на паркування?",
                "expected_keywords": ["тариф", "грн", "паркування"],
                "min_matches": 2
            },
            {
                "question": "Які години роботи паркінгу?",
                "expected_keywords": ["години", "робот", "паркінг"],
                "min_matches": 2
            },
            {
                "question": "Де знаходиться паркінг?",
                "expected_keywords": ["адреса", "місцезнаходження", "паркінг"],
                "min_matches": 1
            },
        ]

        for test in test_cases:
            print(f"Asking: {test['question']}")
            start_time = time.time()
            response = self.chatbot.chat(test['question'])
            response_time = (time.time() - start_time) * 1000

            if isinstance(response, dict):
                response_text = response.get("message", "")
            else:
                response_text = response

            print(f"Response: {response_text[:150]}...")
            print(f"Response time: {response_time:.2f}ms")

            is_accurate, matches = ResponseAccuracy.keyword_match(
                response_text,
                test['expected_keywords'],
                min_matches=test['min_matches']
            )

            no_hallucination = ResponseAccuracy.no_hallucination(response_text)

            print(f"Expected keywords: {test['expected_keywords']}")
            print(f"Keywords found: {matches}/{len(test['expected_keywords'])}")
            print(f"Accurate: {is_accurate}")
            print(f"No hallucination: {no_hallucination}")

            result = {
                "test_name": test["name"],
                "question": test["question"],
                "response": response_text,
                "response_time_ms": response_time,
                "expected_keywords": test['expected_keywords'],
                "keyword_matches": matches,
                "is_accurate": is_accurate,
                "no_hallucination": no_hallucination,
                "metric_type": "retrieval"
            }
            self.results.append(result)

            self.tracker.log_query_metric(
                query_text=test['question'],
                response_text=response_text,
                response_time_ms=response_time,
                is_accurate=is_accurate
            )

    def test_response_accuracy(self):

        test_cases = [
            {
                "question": "Скільки коштує паркування на годину?",
                "expected_keywords": ["грн", "тариф", "паркування"],
                "min_matches": 2
            },
            {
                "question": "Як забронювати місце паркування?",
                "expected_keywords": ["броню", "резерв", "місце"],
                "min_matches": 1
            },
            {
                "question": "Розкажіть про паркінг",
                "expected_keywords": ["паркінг", "інформація", "послуга"],
                "min_matches": 1
            },
        ]

        for test in test_cases:
            print(f"Asking: {test['question']}")
            start_time = time.time()
            response = self.chatbot.chat(test['question'])
            response_time = (time.time() - start_time) * 1000

            # Handle dict responses
            if isinstance(response, dict):
                response_text = response.get("message", "")
            else:
                response_text = response

            print(f"Response: {response_text[:150]}...")
            print(f"Response time: {response_time:.2f}ms")

            # Check keywords
            is_accurate, matches = ResponseAccuracy.keyword_match(
                response_text,
                test['expected_keywords'],
                min_matches=test['min_matches']
            )

            # Check hallucination
            no_hallucination = ResponseAccuracy.no_hallucination(response_text)

            # Check consistency with context
            context = "Smart City Park parking facility with various services"
            is_consistent, consistency_score = ResponseAccuracy.factual_consistency(
                response_text,
                context
            )

            print(f"Keywords found: {matches}/{len(test['expected_keywords'])}")
            print(f"Accurate: {is_accurate}")
            print(f"No hallucination: {no_hallucination}")
            print(f"Consistency score: {consistency_score:.2f}")

            result = {
                "test_name": test["name"],
                "question": test["question"],
                "response": response_text,
                "response_time_ms": response_time,
                "keyword_matches": matches,
                "is_accurate": is_accurate,
                "no_hallucination": no_hallucination,
                "consistency_score": consistency_score,
                "metric_type": "accuracy"
            }
            self.results.append(result)

            # Log to database
            self.tracker.log_query_metric(
                query_text=test['question'],
                response_text=response_text,
                response_time_ms=response_time,
                is_accurate=is_accurate
            )

    def test_performance_tracking(self):

        for i in range(3):
            self.tracker.log_query_metric(
                query_text=f"Test query {i + 1}",
                response_text=f"Test response {i + 1}",
                response_time_ms=100.0 + (i * 20),
                recall_k3=0.8 + (i * 0.05),
                precision_k3=0.75 + (i * 0.05),
                is_accurate=True
            )

        summary = self.tracker.get_performance_summary()
        print(f"\nDatabase Summary:")
        print(f"  Total queries: {summary['total_queries']}")
        print(f"  Avg response time: {summary['avg_response_time_ms']:.2f}ms")
        print(f"  Avg Recall@3: {summary['avg_recall_k3']:.4f}")
        print(f"  Avg Precision@3: {summary['avg_precision_k3']:.4f}")
        print(f"  Accuracy rate: {summary['accuracy_rate']:.1f}%")

    def print_overall_summary(self):
        retrieval_tests = [r for r in self.results if r["metric_type"] == "retrieval"]
        accuracy_tests = [r for r in self.results if r["metric_type"] == "accuracy"]

        if retrieval_tests:
            avg_response_time = sum(r["response_time_ms"] for r in retrieval_tests) / len(retrieval_tests)
            avg_accuracy = sum(1 for r in retrieval_tests if r["is_accurate"]) / len(retrieval_tests)
            avg_no_hallucination = sum(1 for r in retrieval_tests if r["no_hallucination"]) / len(retrieval_tests)
            avg_keyword_match = sum(r["keyword_matches"] / len(r["expected_keywords"]) for r in retrieval_tests) / len(
                retrieval_tests)

            print(f"\nRetrieval Tests (Real Chatbot Queries):")
            print(f"  Tests run: {len(retrieval_tests)}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  Accuracy rate: {avg_accuracy * 100:.1f}%")
            print(f"  Hallucination-free: {avg_no_hallucination * 100:.1f}%")
            print(f"  Keyword match rate: {avg_keyword_match * 100:.1f}%")

        if accuracy_tests:
            avg_accuracy = sum(1 for r in accuracy_tests if r["is_accurate"]) / len(accuracy_tests)
            avg_no_hallucination = sum(1 for r in accuracy_tests if r["no_hallucination"]) / len(accuracy_tests)
            avg_consistency = sum(r["consistency_score"] for r in accuracy_tests) / len(accuracy_tests)
            avg_response_time = sum(r["response_time_ms"] for r in accuracy_tests) / len(accuracy_tests)

            print(f"\nAccuracy Tests (Real Chatbot Queries):")
            print(f"  Tests run: {len(accuracy_tests)}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  Accuracy rate: {avg_accuracy * 100:.1f}%")
            print(f"  Hallucination-free: {avg_no_hallucination * 100:.1f}%")
            print(f"  Consistency score: {avg_consistency:.2f}")

        print(f"\nTotal tests executed: {len(self.results)}")

        # Get database statistics
        summary = self.tracker.get_performance_summary()
        print(f"\nDatabase Statistics:")
        print(f"  Total queries tracked: {summary['total_queries']}")
        print(f"  Accuracy rate: {summary['accuracy_rate']:.1f}%")

    def export_results(self, filename: str = "performance_results.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for i, result in enumerate(self.results, 1):
                f.write(f"Test {i}: {result['test_name']}\n")
                f.write(f"Question: {result.get('question', 'N/A')}\n")
                f.write(f"Response: {result['response'][:200]}...\n")
                f.write(f"Response time: {result['response_time_ms']:.2f}ms\n")
                f.write(f"Keywords found: {result['keyword_matches']}\n")
                f.write(f"Accurate: {result['is_accurate']}\n")
                f.write(f"No hallucination: {result['no_hallucination']}\n")
                if 'consistency_score' in result:
                    f.write(f"Consistency: {result['consistency_score']:.4f}\n")

                f.write("\n")

        print(f"\nResults exported to: {filename}")


def main():
    tester = PerformanceTest()

    if not tester.initialize_chatbot():
        print("Cannot proceed without chatbot")
        return

    tester.test_retrieval_metrics()
    tester.test_response_accuracy()
    tester.test_performance_tracking()
    tester.print_overall_summary()
    tester.export_results()

if __name__ == "__main__":
    main()