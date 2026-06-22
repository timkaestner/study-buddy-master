# Dieser Code ist geschrieben Anlehung an
# https://mohdmus99.medium.com/strategies-and-techniques-for-managing-the-size-of-the-context-window-when-using-llm-large-3c2dbc5dcc3a
from typing import List
import tiktoken
from langchain_core.messages import AnyMessage, SystemMessage


class MessageHandler:
    def __init__(self, model, max_tokens):
        self.max_tokens = max_tokens
        self.conversation: List[AnyMessage] = []
        self.total_tokens = 0
        self.model = model
        self.encoding = self.get_encoding(model)


    def get_encoding(self, model: str):
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback für neue / unbekannte OpenAI-Modelle
            return tiktoken.get_encoding("o200k_base")
            # Alternative für ältere Modelle:
            # return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))


    def add_message(self, message):
        message_content = message.content
        message_tokens = self.count_tokens(message_content)
        self.conversation.append(message)
        self.total_tokens += message_tokens

        # Entfernen der ältesten Nachricht, falls das Tokenlimit überschritten wurde
        while self.total_tokens > self.max_tokens:
            idx = next(
                (i for i, msg in enumerate(self.conversation)
                 if not isinstance(msg, SystemMessage)),
                None
            )

            if idx is None:
                # Only system messages remain
                break

            removed_message = self.conversation.pop(idx)
            self.total_tokens -= self.count_tokens(removed_message.content)

    def get_conversation(self):
        return self.conversation
