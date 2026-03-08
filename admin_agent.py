import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from database import ReservationManager, ParkingReservation
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdminAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.admin_chat_id = os.getenv("ADMIN_TELEGRAM_ID")
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.application = None

        if not self.bot_token or not self.admin_chat_id:
            logger.error("TELEGRAM_BOT_TOKEN and ADMIN_TELEGRAM_ID must be set in .env")
            return

        self.message_chain = self._setup_message_chain()

    def _setup_message_chain(self):
        template = """You are an assistant that formats reservation requests for administrators.

Generate a clear, professional message for the administrator about a new parking reservation request.

Reservation details:
- ID: {reservation_id}
- Customer: {customer_name}
- License Plate: {license_plate}
- Date: {date}
- Time: {time}
- Duration: {duration} hours

Please approve or reject this reservation.

Format the message professionally."""

        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

    async def send_reservation_to_admin(self, reservation: ParkingReservation):
        if not self.application:
            logger.error("Telegram application not initialized")
            return

        try:
            # Generate message using LangChain
            message_text = await self.message_chain.ainvoke({
                "reservation_id": reservation.id,
                "customer_name": reservation.customer_name,
                "license_plate": reservation.license_plate,
                "date": reservation.reservation_date,
                "time": reservation.reservation_time,
                "duration": reservation.duration_hours
            })

            keyboard = [
                [InlineKeyboardButton("✅ Approve", callback_data=f"approve_{reservation.id}")],
                [InlineKeyboardButton("❌ Reject", callback_data=f"reject_{reservation.id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await self.application.bot.send_message(
                chat_id=self.admin_chat_id,
                text=message_text,
                reply_markup=reply_markup
            )

            logger.info(f"Sent reservation {reservation.id} to admin")

        except Exception as e:
            logger.error(f"Failed to send reservation to admin: {e}")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        data = query.data
        action, reservation_id_str = data.split('_')
        reservation_id = int(reservation_id_str)

        if action == "approve":
            status = "confirmed"
            response_text = f"✅ Reservation #{reservation_id} has been approved."
        elif action == "reject":
            status = "cancelled"
            response_text = f"❌ Reservation #{reservation_id} has been rejected."
        else:
            return

        # Update reservation status
        ReservationManager.update_reservation_status(reservation_id, status)

        # Edit the message to remove buttons and show result
        await query.edit_message_text(text=response_text)

        logger.info(f"Admin {action}d reservation {reservation_id}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Admin panel for Parking Chatbot. Use /pending to see pending reservations.")

    async def pending_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pending = ReservationManager.get_pending_reservations()

        if not pending:
            await update.message.reply_text("No pending reservations.")
            return

        text = "Pending Reservations:\n\n"
        for res in pending:
            text += f"ID: {res.id}\nCustomer: {res.customer_name}\nPlate: {res.license_plate}\nDate: {res.reservation_date} {res.reservation_time}\nDuration: {res.duration_hours}h\n\n"

        await update.message.reply_text(text)

    def setup_bot(self):
        if not self.bot_token:
            return

        try:
            self.application = Application.builder().token(self.bot_token).build()

            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("pending", self.pending_command))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))

            logger.info("Telegram bot application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.application = None

    async def run_bot(self):
        if not self.application:
            logger.error("Bot not set up")
            return

        await self.application.initialize()
        await self.application.start()
        logger.info("Admin Telegram bot started")

        # Keep running
        await self.application.updater.start_polling()
        await asyncio.sleep(float('inf'))

# Global instance
admin_agent = AdminAgent()

def init_admin_agent():
    admin_agent.setup_bot()

async def send_reservation_notification(reservation_id: int):
    reservation = ReservationManager.get_reservation(reservation_id)
    if reservation:
        await admin_agent.send_reservation_to_admin(reservation)

def send_reservation_notification_sync(reservation_id: int):
    asyncio.run(send_reservation_notification(reservation_id))

def run_admin_bot():
    if admin_agent.application:
        asyncio.run(admin_agent.run_bot())
    else:
        logger.error("Admin bot not initialized")


if __name__ == "__main__":
    init_admin_agent()
    run_admin_bot()
