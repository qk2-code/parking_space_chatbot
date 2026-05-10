"""
Parking Space Chatbot with RAG, Vector DB, SQL DB, and Guard Rails
Step 1: RAG + Vector DB (Static Data)
Step 2: SQL DB (Dynamic Data) + Guard Rails
Step 3: Interactive Features (Reservations)
"""

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from typing import Dict, Tuple
import logging
import sqlite3

from database import (
    init_database, ReservationManager, PricingManager,
    AvailabilityManager, AuditLog
)
from guardrails import (
    InputValidator, PII_Detector, PII_Masker, ResponseGuard
)
from entity_extraction import ReservationEntityExtractor
from admin_agent import init_admin_agent, send_reservation_notification_sync

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParkingChatbot:

    def __init__(self, data_file: str = "parking.txt"):
        logger.info("Initializing Parking Chatbot...")

        init_database()
        PricingManager.init_default_rates()
        init_admin_agent()

        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        # Setup RAG system (static data from vector DB)
        self.rag_chain = self._setup_rag_system(data_file)

        # Setup database connection for queries
        self.mcp_tools = []
        self._setup_mcp_connection()

        logger.info("✓ Chatbot initialized successfully!")

    def _setup_rag_system(self, file_path: str):
        logger.info(f"Loading and indexing {file_path}...")

        loader = TextLoader(f"data/{file_path}", encoding='utf-8')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        client = chromadb.EphemeralClient()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=client,
            collection_name="parking_collection"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        template = """Ви — помічник із паркування для сервісу Smart City Park.
Використовуйте наданий контекст, щоб відповідати на запитання щодо паркування.
Якщо відповіді немає в контексті, скажіть: «У мене немає інформації з цього приводу. Зверніться до нашої служби підтримки».
Будьте ввічливими та надавайте вичерпну інформацію. Відповідайте тією ж мовою, якою поставлено запитання.

контекст: {context}
запитання: {question}

відповідь:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("✓ RAG system ready (static data from vector DB)")
        return rag_chain

    def _setup_mcp_connection(self):
        """Initialize database connection for reservation queries"""
        try:
            logger.info("Setting up database connection...")

            def query_reservations():
                """Fetch all parking reservations from the database."""
                try:
                    conn = sqlite3.connect('parking_data.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM reservations")
                    rows = cursor.fetchall()
                    conn.close()

                    if not rows:
                        return "No reservations found."

                    result = "Reservations:\n\n"
                    for row in rows:
                        result += f"ID: {row[0]}, Customer: {row[1]}, Plate: {row[2]}, Date: {row[3]}, Time: {row[4]}, Duration: {row[5]} hours\n"
                    return result
                except Exception as e:
                    logger.error(f"Database query error: {e}")
                    return f"Error querying database: {str(e)}"

            self.mcp_tools = [
                Tool(
                    name="get_reservations",
                    description="Fetch and display all parking reservations from the database.",
                    func=query_reservations
                )
            ]
            logger.info("✓ Database connection ready")
        except Exception as e:
            logger.warning(f"Database setup failed: {e}")
            self.mcp_tools = []

    def validate_input(self, user_input: str) -> Tuple[bool, Dict]:
        """Validate user input for safety and relevance"""
        validation = InputValidator.validate_input(user_input)

        if not validation["is_valid"]:
            return False, validation

        if not validation["is_parking_related"]:
            logger.warning(f"Non-parking input: {user_input}")

        return True, validation

    def process_reservation_request(self, user_input: str) -> Dict:
        """Attempt to extract and process reservation details"""
        logger.info("Processing reservation request...")

        entities = ReservationEntityExtractor.extract_all_entities(user_input)
        is_valid, errors = ReservationEntityExtractor.validate_entities(entities)

        if not is_valid:
            return {
                "status": "incomplete",
                "message": "Missing information for reservation:\n" + "\n".join([f"• {e}" for e in errors]),
                "entities": entities
            }

        reservation_id = ReservationManager.create_reservation(
            customer_name=" ".join(entities["customer_names"]),
            license_plate=entities["license_plate"],
            reservation_date=entities["date"],
            reservation_time=entities["time"],
            duration_hours=entities["duration_hours"]
        )

        if not reservation_id:
            return {
                "status": "duplicate",
                "message": "A reservation for this vehicle on this date and time already exists."
            }

        send_reservation_notification_sync(reservation_id)

        AuditLog.log_interaction(
            action="reservation_created",
            user_input=user_input,
            bot_response=f"Reservation #{reservation_id} created",
            contains_pii=PII_Detector.has_pii(user_input)
        )

        return {
            "status": "success",
            "reservation_id": reservation_id,
            "message": f"""✅ Резервування успішно створено!
            
            Номер резервування: #{reservation_id}{ReservationEntityExtractor.format_reservation_summary(entities)}
            Наш адміністратор розглянет ваш запит. Ви отримаєте підтвердження протягом 30 хвилин."""
        }

    def chat(self, user_input: str) -> str:
        """Main chatbot interaction method"""

        # Step 1: Validate input for safety
        is_valid, validation = self.validate_input(user_input)

        if not is_valid:
            AuditLog.log_interaction(
                action="invalid_input",
                user_input=user_input,
                bot_response="Invalid input detected",
                contains_pii=validation.get("contains_pii", False)
            )

            if validation.get("contains_malicious"):
                return "I cannot process this request. Please ask a legitimate parking-related question."

            return validation.get("warning_message", "I'm unable to process your request.")

        # Step 2: Check if this is a reservation request
        if any(word in user_input.lower() for word in ["броню", "бронюю", "booking", "резерв", "reserve", "place"]):
            return self.process_reservation_request(user_input)

        # Step 3: Check if user is asking to view database/reservations
        if any(word in user_input.lower() for word in ["print", "show", "display", "list", "data", "database", "резервування", "дані"]):
            if self.mcp_tools:
                try:
                    result = self.mcp_tools[0].func()
                    return result
                except Exception as e:
                    logger.error(f"Database query error: {e}")
                    return "Unable to retrieve database data at this moment."

        # Step 4: Query RAG system for static information
        try:
            bot_response = self.rag_chain.invoke(user_input)
        except Exception as e:
            logger.error(f"RAG chain error: {e}")
            bot_response = "Sorry, I encountered an error processing your question. Please try again."

        # Step 5: Apply guard rails - DON'T mask names in bot responses (only mask license plates/phones/emails)
        # Keep bot responses readable - only mask actual sensitive data, not legitimate content
        protected_response = ResponseGuard.sanitize_response(
            bot_response,
            mask_pii=False,  # Don't mask bot responses - they should be readable
            include_disclaimers=False
        )

        # Step 6: Log interaction
        AuditLog.log_interaction(
            action="query_response",
            user_input=user_input,
            bot_response=protected_response,
            contains_pii=PII_Detector.has_pii(user_input)
        )

        return protected_response


def interactive_chat():
    chatbot = ParkingChatbot()

    print("SMART CITY PARK - Parking Chatbot")
    print("Type 'exit' to quit, 'history' to see logs\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Bot: Дякуємо за звернення! До побачення!")
                break

            if user_input.lower() == "history":
                logs = AuditLog.get_interaction_logs(limit=5)
                print("\nRecent interactions:")
                for log in logs:
                    print(f"  [{log['action']}] {log['timestamp']}")
                continue

            response = chatbot.chat(user_input)

            # Handle both string and dict responses
            if isinstance(response, dict):
                print(f"Bot: {response.get('message', 'No response')}")
            else:
                print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n\nBot: Сесія завершена.")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")


if __name__ == "__main__":
    interactive_chat()
