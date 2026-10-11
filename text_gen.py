import torch
from dotenv import load_dotenv
from transformers import pipeline

# Picks up HF_TOKEN from .env (optional; the default model is public)
load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SAMPLE_TICKETS = [
    {"plan": "premium", "industry": "e-commerce", "category": "deliverability", "urgency": "high", "tone": "frustrated", "message":""},
    {"plan": "free", "industry": "nonprofit", "category": "billing", "urgency": "low", "tone": "polite", "message":""},
    {"plan": "standard", "industry": "agency", "category": "api", "urgency": "medium", "tone": "terse", "message":""},
]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_generator():
    return pipeline("text-generation", model=MODEL_ID, device=pick_device())


def build_messages(ticket):
    system = (
        "You write realistic customer support tickets for an email-marketing SaaS company. "
        "Write as the customer, not as support staff."
    )
    user = f"""Write one support ticket from a customer with these details:
            - Plan: {ticket["plan"]}
            - Industry: {ticket["industry"]}
            - Category: {ticket["category"]}
            - Urgency: {ticket["urgency"]}
            - Tone: {ticket["tone"]}
            - User Message: {ticket['message']}

            Rules:
            - The tone must clearly sound {ticket["tone"]}.
            - Mention a concrete detail that fits a {ticket["industry"]} business.
            - Generate the content from the user provided content {ticket['message']}
            - Keep the body to 3-5 sentences.
            - Sign off with a realistic first name.

            Format:
            Subject: <subject line>
            Body: <body>
            """
        
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]



def generate_text(gen, ticket):
    messages = build_messages(ticket)
    output = gen(messages, max_new_tokens=256, do_sample=True, temperature=0.7)
    # The pipeline returns the whole conversation; the last message is the model's reply
    return output[0]["generated_text"][-1]["content"]


if __name__ == "__main__":
    gen = load_generator()

    # Peek under the hood once
    print(gen.tokenizer.apply_chat_template(build_messages(SAMPLE_TICKETS[0]), tokenize=False, add_generation_prompt=True))

    for ticket in SAMPLE_TICKETS:
        print("=" * 60)
        print(ticket)
        print(generate_text(gen, ticket))
