import numpy as np

class ChatBot:

    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def start_chat(self):
        user_response = input()

        if user_response in self.negative_responses:
            print("Ok, goodbye")
            return

        self.chat(user_response)

    def chat(self, reply):
        while not self.end_chat(reply):
            reply = input(self.generate_response(reply))

    def generate_response(self, user_input):
        chatbot_response = "ok"

        return chatbot_response

    def end_chat(self, reply):
        for command in self.exit_commands:
            if command in reply:
                print("Ok, goodbye")
                return True
            
        return False