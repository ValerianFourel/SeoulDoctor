from models import State
from typing import List, Optional, Dict, Any, Tuple


def get_greeting_message(language: str) -> str:
    """Greeting for NEW_SEARCH branch."""
    if "English" in language:
        return "Let's start fresh! What kind of medical facility are you looking for?"
    else:
        return "새로 시작하겠습니다! 어떤 의료 시설을 찾으시나요?"


def get_reset_confirmation(language: str) -> str:
    """Confirmation message for reset/quit."""
    if "English" in language:
        return "Got it. Starting over. What kind of medical facility are you looking for?"
    else:
        return "알겠습니다. 다시 시작하겠습니다. 어떤 의료 시설을 찾으시나요?"


def generate_change_acknowledgment(state: State, extracted: Dict) -> str:
    """Acknowledge the change request."""
    lang = state.language_pref
    
    if "English" in lang:
        if extracted.get('specialty'):
            return f"Got it, looking for {extracted['specialty']} instead."
        elif extracted.get('location'):
            return f"Understood, searching in {extracted['location']} now."
        else:
            return "I've updated your search criteria."
    else:
        if extracted.get('specialty'):
            return f"알겠습니다, {extracted['specialty']}로 변경했습니다."
        elif extracted.get('location'):
            return f"네, {extracted['location']} 지역으로 검색하겠습니다."
        else:
            return "검색 조건을 업데이트했습니다."


def ask_for_missing_info(state: State) -> str:
    """Ask for missing specialty or location."""
    lang = state.language_pref if state.language_pref else "Korean"
    
    if not state.specialty:
        if "English" in lang:
            return "What type of medical facility do you need? (e.g., dentist, dermatologist)"
        else:
            return "어떤 의료 시설을 찾으시나요? (예: 치과, 피부과)"
    
    if not state.location and not state.latitude:
        if "English" in lang:
            return "Which area in Seoul are you looking in? (e.g., Gangnam, Songpa)"
        else:
            return "서울 어느 지역을 찾으시나요? (예: 강남, 송파)"
    
    return "Tell me more about what you're looking for."


def ask_for_specialty_clarification(state: State) -> str:
    """Ask for clarification when specialty confidence is low."""
    lang = state.language_pref
    
    if "English" in lang:
        return f"Just to confirm, you're looking for a {state.specialty}? Please specify the exact type of facility."
    else:
        return f"{state.specialty}를 찾으시는 게 맞나요? 정확한 시설 종류를 알려주세요."


def generate_chit_chat_response(message: str, state: State) -> str:
    """Simple conversational responses."""
    lang = state.language_pref if state.language_pref else "Korean"
    message_lower = message.lower()
    
    # Greetings
    if any(word in message_lower for word in ["hello", "hi", "안녕"]):
        if "English" in lang:
            return "Hi! I can help you find medical facilities in Seoul. What are you looking for?"
        else:
            return "안녕하세요! 서울의 의료 시설을 찾아드리겠습니다. 무엇을 도와드릴까요?"
    
    # Thanks
    if any(word in message_lower for word in ["thanks", "thank you", "감사", "고마워"]):
        if "English" in lang:
            return "You're welcome! Let me know if you need anything else."
        else:
            return "천만에요! 다른 도움이 필요하시면 언제든지 말씀해주세요."
    
    # Satisfaction
    if any(word in message_lower for word in ["perfect", "good", "great", "괜찮", "좋아"]):
        if "English" in lang:
            return "Great! I'm glad I could help. Anything else you need?"
        else:
            return "좋습니다! 도움이 되어 기쁩니다. 다른 것이 필요하신가요?"
    
    # Default
    if "English" in lang:
        return "How can I help you today?"
    else:
        return "무엇을 도와드릴까요?"


def generate_recovery_prompt(state: State) -> str:
    """Help message when user is stuck (stagnation check)."""
    lang = state.language_pref if state.language_pref else "Korean"
    
    if "English" in lang:
        return """It seems we're having trouble finding what you need. Let me help:

1. Type "dentist in Gangnam" for a simple search
2. Say "reset" to start over
3. Tell me: What hurts? Where are you located?"""
    else:
        return """원하시는 것을 찾는 데 어려움이 있는 것 같습니다:

1. "강남 치과"처럼 간단히 말씀해주세요
2. "처음부터"라고 하시면 다시 시작합니다
3. 증상과 위치를 알려주세요"""

def format_response(response: str) -> str:
    # 1. Handle bolded numbered headers: **1. Title**
    response = re.sub(
        r'\*\*(\d+\.\s*.*?)\*\*',
        r'\n\1\n',
        response
    )

    # 2. Handle remaining bold text: **Title**
    response = re.sub(
        r'\*\*(.*?)\*\*',
        r'\1\n',
        response
    )

    # 3. Ensure newline before numbered items not already on new line
    response = re.sub(
        r'(?<!\n)(\s*)(\d+\.)',
        r'\n\2',
        response
    )

    # 4. Cleanup excessive newlines
    response = re.sub(r'\n{3,}', '\n\n', response)

    return response.strip()

