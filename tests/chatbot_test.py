import pytest
from main import ParkingChatbot

@pytest.fixture(scope="module")
def chatbot():
    return ParkingChatbot()

@pytest.mark.parametrize("user_input", [
    "Які тарифи на паркування?",
    "Які години роботи паркінгу?",
    "Привіт!"
])
def test_chatbot_responses(chatbot, user_input):
    response = chatbot.chat(user_input)
    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0
