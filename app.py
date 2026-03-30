import logging
from src.planner.planner import Planner
from src.rag.retriever import SemanticRetriever
from src.agents.generator import ResponseGenerator

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomerSupportBot:
    def __init__(self):
        print("Initializing AssisMarket AI Pipeline...")
        #Load all the modules
        self.planner = Planner(sentiment_threshold=0.55) #Return plan user message, intent, sentiment
        self.retriever = SemanticRetriever("policy_vector_store.pkl")
        self.generator = ResponseGenerator()
        print("All sytems online. Models loaded into memory.\n")

    def chat(self, user_message: str):
         """The main orchestration pipeline."""

         #Planner phase detect intent ans sentiment
         plan = self.planner.route_message(user_message) #return plan a dict of user input, intent and sentiment
         intent = plan["target_intent"]
         sentiment = plan["detected_sentiment"]

         context = self.retriever.search(user_message) #return top k relevant context for the user input

         response = self.generator.generate_response(user_message,intent, sentiment, context)

         return plan, response


if __name__ == "__main__":
    bot = CustomerSupportBot()

    print("===============================================")
    print(" AssisMarker Support Agent - Terminal Interface")
    print("===============================================")
    print("Type 'exit' or 'quit' to stop\n")

    while True:
        user_input = input("👤 You: ")

        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down AI... Goodbye")
            break
    
        if not user_input.strip():
            continue

        #Run the pipeline
        plan, response = bot.chat(user_input)

        # Print the exact routing decisions (so you can show it off on your portfolio!)
        print(f"\n⚙️ [PIPELINE LOG] Intent: {plan['target_intent']} | Sentiment: {plan['detected_sentiment']}")
        
        # Print the final AI answer
        print(f"🤖 Agent: {response}\n")