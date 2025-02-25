import os
from chatterbot import ChatBot as CB
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import nlp_utils, preprocess_input, spell_check
from utils.match_intent import match_intent


class ChatBot:
    def __init__(self, name="MyBot"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(current_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        db_path = os.path.join(tmp_dir, "db.sqlite3")
        database_uri = f"sqlite:///{db_path}"

        self.bot = CB(name, database_uri=database_uri)
        self.trainer = ChatterBotCorpusTrainer(self.bot)
        self.conversation_history = []

    def train_bot(self):
        self.trainer.train("chatterbot.corpus.english")

        self.trainer.train("data/custom_corpus.yml")

    def get_response(self, message):
        self.conversation_history.append({"user": message})

        message = preprocess_input.preprocess_input(message)

        self.conversation_history.append({"input_preprocess": message})

        message = spell_check.correct_spelling(message)

        self.conversation_history.append({"spell_check": message})

        nlp_utils.analyze_text(message)

        custom_response = match_intent(message)
        if custom_response:
            response = custom_response
        else:
            response = self.bot.get_response(message)

        self.conversation_history.append({"bot": str(response)})

        return response

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.train_bot()
