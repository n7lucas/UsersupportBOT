import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

#setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseGenerator:

    def __init__(self):
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("No OpenAI api key found in .env file")
        self.client = OpenAI()

    def generate_response(self, user_message: str, intent: str, sentiment: str, retrieved_context: list) -> str:
        """
        Combines the Planner's routing data and the Retriever's knowledge 
        to generate a policy-accurate, empathetic response.
        """
        logger.info("Generating LLMK response based on context...")
        
        # Format the retrieved knowledge into a clean string for the LLM to read
        formatted_context = ""
        for i, match in enumerate(retrieved_context, 1):
            formatted_context += f"--- Policy Section: {match['metadata']} --- \n{match['content']}\n\n"


        # Build the system prompt
        system_prompt = f"""
        You are a world-class customer support agent for AssisMarket Online Retail.
        
        YOUR OBJECTIVE:
        Answer the customer's query using ONLY the provided company policy text below.
        If the policy does not contain the answer, politely state that you must transfer them to a human supervisor. Do not invent rules.
        
        SYSTEM DETECTIONS:
        - Customer's core intent: {intent}
        - Customer's emotional state: {sentiment}
        
        INSTRUCTIONS FOR TONE:
        - If the sentiment is 'Frustrated', begin with a sincere apology and be highly empathetic.
        - If the sentiment is 'Urgent', keep the response brief and action-oriented.
        - If the sentiment is 'Polite' or 'Neutral', maintain a warm, professional tone.
        
        --- COMPANY POLICY CONTEXT ---
        {formatted_context}
        """

        #Call the LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}")
            return "I apologize, but I am currently experiencing a system error. Please try again in a moment or contact support@AssisMarket.com directly."

if __name__ == "__main__":

    generator = ResponseGenerator()

    
    # Mock data from our previous tests
    test_user_msg = "How many days do I have to return an item if it's broken? I'm so annoyed it arrived like this!"
    mock_intent = "return_item"
    mock_sentiment = "Frustrated"
    mock_context = [
        {
            "metadata": "{'Section': '8. Damaged or Incorrect Orders'}",
            "content": "If a customer receives a damaged product, they must contact support within 7 days of delivery. Replacement shipments are typically processed within 2 business days."
        }
    ]
    
    final_answer = generator.generate_response(
        test_user_msg, 
        mock_intent, 
        mock_sentiment, 
        mock_context
    )
    
    print("\n================ FINAL LLM RESPONSE ================\n")
    print(final_answer)
    print("\n====================================================")