from bot import ChatBot

def main():
    chatbot = ChatBot()
    print("Training chatbot... Please wait.")
    chatbot.train_bot()
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            print("\nConversation History:\n")
            for entry in chatbot.conversation_history:
                if 'user' in entry:
                    print("You:", entry['user'])
                if 'input_preprocess' in entry:
                    print("You (pre processed):", entry['input_preprocess'])
                if 'spell_check' in entry:
                    print("You (spell checked):", entry['spell_check'])
                if 'bot' in entry:
                    print("Bot:", entry['bot'], "\n")
            break
        response = chatbot.get_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
