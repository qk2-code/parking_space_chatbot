parking space chatbot

this is a chatbot for smart city park parking spaces.

features:
- answers questions about parking using rag and vector database
- manages reservations with sql database
- includes guard rails for safety
- admin agent for reservation approval via Telegram

requirements:
- python 3.x
- see requirements.txt for dependencies
- telegram bot token and admin chat ID

setup:
1. pip install -r requirements.txt
2. create a .env file with:
   - GROQ_API_KEY=your_groq_api_key
   - TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   - ADMIN_TELEGRAM_ID=your_admin_chat_id
3. Run the main chatbot: python main.py
4. in a separate terminal, run the admin bot: python admin_bot.py
