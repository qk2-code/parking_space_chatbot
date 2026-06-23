from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import logging
import sqlite3

from admin_agent import send_reservation_notification_sync
from database import ReservationManager, AuditLog
from entity_extraction import ReservationEntityExtractor
from guardrails import ResponseGuard

logger = logging.getLogger(__name__)


class ChatbotState(TypedDict):
    """State schema for the parking chatbot workflow"""
    user_input: str
    validation: dict
    rag_response: str
    is_reservation: bool
    is_database_query: bool
    entities: dict
    admin_decision: str
    reservation_id: int | None
    final_response: str
    contains_pii: bool
    error: str | None
    database_result: str


class ParkingChatbotOrchestrator:
    """LangGraph-based orchestrator for parking chatbot workflow"""

    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.graph = self._build_graph()
        logger.info("✓ Orchestrator initialized with LangGraph")

    def _build_graph(self):
        """Build the state machine workflow"""
        workflow = StateGraph(ChatbotState)

        # Define nodes
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("extract_reservation", self._extract_reservation_node)
        workflow.add_node("database_query", self._database_query_node)
        workflow.add_node("rag_query", self._rag_query_node)
        workflow.add_node("admin_approval", self._admin_approval_node)
        workflow.add_node("record_data", self._record_data_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define edges
        workflow.add_edge(START, "validate_input")

        # Validation routing
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_validation,
            {
                "continue": "extract_reservation",
                "error": "handle_error"
            }
        )

        # Check if reservation request or database query
        workflow.add_conditional_edges(
            "extract_reservation",
            self._should_escalate_or_query_db,
            {
                "escalate": "admin_approval",
                "database": "database_query",
                "rag": "rag_query"
            }
        )

        # Admin decision routing
        workflow.add_conditional_edges(
            "admin_approval",
            self._process_admin_decision,
            {
                "approve": "record_data",
                "reject": "generate_response",
                "pending": "generate_response"
            }
        )

        # Record data and generate response
        workflow.add_edge("record_data", "generate_response")
        workflow.add_edge("database_query", "generate_response")
        workflow.add_edge("rag_query", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _validate_input_node(self, state: ChatbotState) -> dict:
        """Node 1: Validate user input for safety and relevance"""
        logger.info(f"[VALIDATE] Processing input: {state['user_input'][:50]}...")

        is_valid, validation = self.chatbot.validate_input(state["user_input"])
        contains_pii = validation.get("contains_pii", False)

        return {
            "validation": validation,
            "contains_pii": contains_pii,
            "error": None if is_valid else validation.get("warning_message")
        }

    def _should_continue_validation(self, state: ChatbotState) -> str:
        """Routing: Check if input is valid"""
        if state["error"]:
            logger.warning(f"[ROUTE] Invalid input detected: {state['error']}")
            return "error"
        return "continue"

    def _extract_reservation_node(self, state: ChatbotState) -> dict:
        """Node 2: Extract reservation entities if applicable"""
        user_input = state["user_input"]

        # Check if this is a reservation request
        is_reservation = any(
            word in user_input.lower()
            for word in ["броню", "бронюю", "booking", "резерв", "reserve", "place"]
        )

        logger.info(f"[EXTRACT] Reservation detected: {is_reservation}")

        if is_reservation:
            entities = ReservationEntityExtractor.extract_all_entities(user_input)
            is_valid, errors = ReservationEntityExtractor.validate_entities(entities)

            return {
                "is_reservation": True,
                "is_database_query": False,
                "entities": entities,
                "error": None if is_valid else "\n".join([f"• {e}" for e in errors])
            }

        return {
            "is_reservation": False,
            "is_database_query": False,
            "entities": {}
        }

    def _should_escalate_or_query_db(self, state: ChatbotState) -> str:
        """Routing: Check if reservation, database query, or RAG query"""
        if state["is_reservation"]:
            logger.info("[ROUTE] Escalating reservation to admin approval")
            return "escalate"
        if self._is_database_query(state["user_input"]):
            logger.info("[ROUTE] Routing to database query")
            return "database"

        logger.info("[ROUTE] Routing to RAG system for information query")
        return "rag"

    def _is_database_query(self, user_input: str) -> bool:
        """Check if user is asking for database/reservation data"""
        keywords = [
            "show", "display", "list", "get", "print",
            "database", "reservations", "all reservations",
            "бронювання", "бронирование", "броньку", "брони"
        ]
        return any(keyword in user_input.lower() for keyword in keywords)

    def _database_query_node(self, state: ChatbotState) -> dict:
        """Node 3a: Query database for reservation data"""
        logger.info("[DATABASE] Fetching reservations from database...")

        try:
            result = self._get_reservations_from_db()
            logger.info("[DATABASE] ✓ Reservations fetched successfully")

            return {
                "is_database_query": True,
                "database_result": result
            }
        except Exception as e:
            logger.error(f"[DATABASE] Error: {e}")
            return {
                "is_database_query": True,
                "error": f"Database query failed: {str(e)}",
                "database_result": ""
            }

    def _get_reservations_from_db(self) -> str:
        """Fetch all reservations from database"""
        try:
            conn = sqlite3.connect('parking_data.db')
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, customer_name, license_plate, reservation_date, 
                       reservation_time, duration_hours, status 
                FROM reservations 
                ORDER BY reservation_date DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "No reservations found in the database."

            result = "📋 **Current Reservations:**\n\n"
            for row in rows:
                res_id, customer, plate, date, time, duration, status = row
                result += f"• **ID {res_id}**: {customer} | Plate: {plate}\n"
                result += f"  📅 {date} at {time} | Duration: {duration}h | Status: {status}\n\n"

            return result

        except sqlite3.OperationalError as e:
            logger.error(f"[DATABASE] Operational error: {e}")
            raise
        except Exception as e:
            logger.error(f"[DATABASE] Unexpected error: {e}")
            raise

    def _rag_query_node(self, state: ChatbotState) -> dict:
        """Node 3b: Query RAG system for static information"""
        logger.info("[RAG] Querying vector database...")

        try:
            bot_response = self.chatbot.rag_chain.invoke(state["user_input"])
            logger.info("[RAG] ✓ Response generated successfully")
            return {"rag_response": bot_response}
        except Exception as e:
            logger.error(f"[RAG] Error: {e}")
            return {"error": "RAG processing failed", "rag_response": ""}

    def _admin_approval_node(self, state: ChatbotState) -> dict:
        """Node 4: Send reservation to admin for approval (Human-in-the-Loop)"""
        logger.info(f"[ADMIN] Requesting approval for reservation")

        if state["error"]:
            return {
                "admin_decision": "rejected",
                "final_response": f"Cannot process reservation: {state['error']}"
            }

        entities = state["entities"]
        customer_name = " ".join(entities.get("customer_names", ["Unknown"]))
        license_plate = entities.get("license_plate", "Unknown")
        reservation_date = entities.get("date", "Unknown")
        reservation_time = entities.get("time", "Unknown")
        duration = entities.get("duration_hours", 0)

        try:
            reservation_id = ReservationManager.create_reservation(
                customer_name=customer_name,
                license_plate=license_plate,
                reservation_date=reservation_date,
                reservation_time=reservation_time,
                duration_hours=duration
            )

            if not reservation_id:
                return {
                    "admin_decision": "duplicate",
                    "final_response": "A reservation for this vehicle already exists."
                }

            send_reservation_notification_sync(reservation_id)
            logger.info(f"[ADMIN] Reservation #{reservation_id} sent for approval")

            return {
                "admin_decision": "pending",
                "reservation_id": reservation_id,
                "final_response": f"""⏳ Reservation № {reservation_id} is waiting for admin approval.
You'll receive a confirmation within 30 minutes."""
            }

        except Exception as e:
            logger.error(f"[ADMIN] Approval process error: {e}")
            return {
                "admin_decision": "error",
                "error": str(e),
                "final_response": "Unable to process reservation at this time."
            }

    def _process_admin_decision(self, state: ChatbotState) -> str:
        """Routing: Handle admin decision"""
        decision = state.get("admin_decision", "pending")
        logger.info(f"[ROUTE] Admin decision: {decision}")
        return decision if decision in ["approve", "reject", "pending"] else "reject"

    def _record_data_node(self, state: ChatbotState) -> dict:
        """Node 5: Record reservation data to database after approval"""
        logger.info(f"[RECORD] Recording reservation #{state.get('reservation_id')}")

        reservation_id = state.get("reservation_id")
        entities = state["entities"]

        try:
            reservation = ReservationManager.get_reservation(reservation_id)
            if reservation.status == "confirmed":
                AuditLog.log_interaction(
                action="reservation_confirmed",
                user_input=state["user_input"],
                bot_response=f"Reservation #{reservation_id} confirmed",
                contains_pii=state.get("contains_pii", False)
                )

                logger.info(f"[RECORD] ✓ Reservation #{reservation_id} confirmed and recorded")

            return {
                "final_response": f"""✅ The reservation was completed successfully!

Reservation number: #{reservation_id}{ReservationEntityExtractor.format_reservation_summary(entities)}
Thanks for using Smart City Park!"""
            }

        except Exception as e:
            logger.error(f"[RECORD] Data recording error: {e}")
            return {"error": str(e), "final_response": "Error recording reservation."}

    def _generate_response_node(self, state: ChatbotState) -> dict:
        """Node 6: Generate and sanitize final response"""
        logger.info("[RESPONSE] Generating final response")

        if state.get("error"):
            final_response = state.get("final_response", f"Error: {state['error']}")
        elif state.get("is_database_query") and state.get("database_result"):
            final_response = state["database_result"]
        elif state.get("rag_response"):
            final_response = ResponseGuard.sanitize_response(
                state["rag_response"],
                mask_pii=False,
                include_disclaimers=False
            )
        else:
            final_response = state.get("final_response", "No response generated")

        AuditLog.log_interaction(
            action="interaction_completed",
            user_input=state["user_input"],
            bot_response=final_response,
            contains_pii=state.get("contains_pii", False)
        )

        logger.info("[RESPONSE] ✓ Response ready for delivery")

        return {"final_response": final_response}

    def _handle_error_node(self, state: ChatbotState) -> dict:
        """Node 7: Handle validation errors gracefully"""
        logger.error(f"[ERROR] Input validation failed: {state.get('error')}")

        AuditLog.log_interaction(
            action="invalid_input",
            user_input=state["user_input"],
            bot_response="Invalid input detected",
            contains_pii=state.get("contains_pii", False)
        )

        return {
            "final_response": state.get("error", "I'm unable to process your request.")
        }

    def process(self, user_input: str) -> str:
        """Execute the workflow for a user input"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[WORKFLOW] Starting new interaction")
        logger.info(f"{'=' * 60}")

        initial_state: ChatbotState = {
            "user_input": user_input,
            "validation": {},
            "rag_response": "",
            "is_reservation": False,
            "is_database_query": False,
            "entities": {},
            "admin_decision": "",
            "reservation_id": None,
            "final_response": "",
            "contains_pii": False,
            "error": None,
            "database_result": ""
        }

        try:
            result = self.graph.invoke(initial_state)
            final_response = result.get("final_response", "No response generated")

            logger.info(f"[WORKFLOW] ✓ Interaction completed")
            logger.info(f"{'=' * 60}\n")

            return final_response

        except Exception as e:
            logger.error(f"[WORKFLOW] Critical error: {e}")
            return "System error occurred. Please try again later."
