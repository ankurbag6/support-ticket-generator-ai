import gradio as gr

from text_gen import SAMPLE_TICKETS, build_messages, generate_text, load_generator

# Load the model once at startup, not on every button click
gen = load_generator()


def make_ticket(plan, industry, category, urgency, tone, message):
    ticket = {"plan": plan, "industry": industry, "category": category, "urgency": urgency, "tone": tone, "message": message}
    # Show the exact prompt the model sees, so you can debug prompt changes
    prompt = gen.tokenizer.apply_chat_template(build_messages(ticket), tokenize=False, add_generation_prompt=True)
    return generate_text(gen, ticket), prompt


demo = gr.Interface(
    fn=make_ticket,
    inputs=[
        gr.Dropdown(["free", "standard", "premium"], value="premium", label="Plan"),
        gr.Dropdown(["e-commerce", "nonprofit", "agency"], value="e-commerce", label="Industry"),
        gr.Dropdown(
            ["billing", "deliverability", "account", "list import", "campaign editor", "automation", "api", "integrations"],
            value="deliverability",
            label="Category",
        ),
        gr.Radio(["low", "medium", "high"], value="high", label="Urgency"),
        gr.Radio(["frustrated", "polite", "terse", "confused"], value="frustrated", label="Tone"),
        gr.Textbox(label="Your Message", placeholder="e.g. ADD YOUR DETAILS", lines=3),
    ],
    outputs=[
        gr.Textbox(label="Generated ticket", lines=12),
        gr.Textbox(label="Prompt sent to the model (chat template)", lines=12),
    ],
    examples=[[t["plan"], t["industry"], t["category"], t["urgency"], t["tone"], ""] for t in SAMPLE_TICKETS],
    title="Support Ticket Generator",
    description="Pick ticket attributes and generate a synthetic support ticket with a local Qwen model.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
