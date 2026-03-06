import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('main').setLevel(logging.WARNING)

from main import ParkingChatbot

chatbot = ParkingChatbot()

def chat_and_print_response(chat_input: str):
    response = chatbot.chat(chat_input)
    print(f"User: {chat_input}")
    print(f"Bot: {response}\n")

chat_and_print_response('Які тарифи на паркування?')
chat_and_print_response('Які години роботи паркінгу?')
chat_and_print_response('Привіт!')