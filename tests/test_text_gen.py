from text_gen import SAMPLE_TICKETS, build_messages, generate_text, pick_device

TICKET = {
    "plan": "premium",
    "industry": "e-commerce",
    "category": "deliverability",
    "urgency": "high",
    "tone": "frustrated",
    "message": "Our Black Friday emails are landing in spam",
}


class FakeGenerator:
    """Stands in for the HF pipeline so tests don't download the model."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [{"generated_text": messages + [{"role": "assistant", "content": self.reply}]}]


def test_build_messages_roles():
    messages = build_messages(TICKET)
    assert [m["role"] for m in messages] == ["system", "user"]


def test_build_messages_includes_ticket_fields():
    user = build_messages(TICKET)[1]["content"]
    for value in TICKET.values():
        assert value in user


def test_sample_tickets_build_cleanly():
    for ticket in SAMPLE_TICKETS:
        assert build_messages(ticket)[1]["content"]


def test_generate_text_returns_assistant_reply():
    gen = FakeGenerator("Subject: Help\nBody: Emails in spam.")
    assert generate_text(gen, TICKET) == "Subject: Help\nBody: Emails in spam."


def test_generate_text_passes_sampling_args():
    gen = FakeGenerator("ok")
    generate_text(gen, TICKET)
    _, kwargs = gen.calls[0]
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["do_sample"] is True


def test_pick_device_is_valid():
    assert pick_device() in {"cuda", "mps", "cpu"}
