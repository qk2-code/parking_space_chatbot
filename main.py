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
from main_langgraph import ParkingChatbotOrchestrator

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
        self.orchestrator = ParkingChatbotOrchestrator(self)
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

        template = """You are a parking assistant for the Smart City Park service.
Use the provided context to answer questions about parking.
If the answer is not in the context, say: 'I do not have information on this matter. Please contact our support team.'
Be polite and provide comprehensive information. Respond in the same language in which the question was asked.

context: {context}
question: {question}

answer:"""

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
            "message": f"""✅ Reservation was created successfully!
            
            Reservation number: #{reservation_id}{ReservationEntityExtractor.format_reservation_summary(entities)}
            Our admin will review your request. You'll receive a confirmation within 30 minutes."""
        }

    def chat(self, user_input: str) -> str:
        """Main chatbot interaction method (now using LangGraph)"""
        return self.orchestrator.process(user_input)

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
                print("Bot: Thanks for the interactions! Goodbye")
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
            print("\n\nBot: The session has ended.")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")


if __name__ == "__main__":
    interactive_chat()
