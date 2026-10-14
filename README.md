# Support Ticket Generator

Generate synthetic customer support tickets for an email-marketing SaaS using a local
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model and a Gradio UI.

Pick a plan, industry, category, urgency and tone (and optionally a short message), and the
model writes a ticket in the customer's voice. The UI also shows the exact chat-templated
prompt sent to the model, which is handy for debugging prompt changes.

> **Note:** This is a learning project, built to get hands-on with running open models
> locally through the Hugging Face ecosystem. It is not intended for production use.

Demo : https://www.youtube.com/watch?v=4Dwg2OfXaLc

## How it uses Hugging Face

The [Hugging Face](https://huggingface.co/) ecosystem does most of the heavy lifting:

- **[Transformers](https://huggingface.co/docs/transformers/index)**: the `pipeline("text-generation", ...)`
  helper in `text_gen.py` downloads the model and tokenizer, moves them to the right device
  and handles generation in a single call.
- **Chat templates**: the pipeline takes a list of `{"role": ..., "content": ...}` messages and
  formats them with the model's own chat template. `tokenizer.apply_chat_template(...)` is used
  in the UI to show that formatted prompt.
- **[Hugging Face Hub](https://huggingface.co/docs/hub/index)**: the model weights for
  `Qwen/Qwen2.5-1.5B-Instruct` are pulled from the Hub on first run and cached locally.
  An `HF_TOKEN` is optional for this public model.
- **[Gradio](https://www.gradio.app/)**: the web UI library maintained by Hugging Face.

## Project layout

```
app.py               Gradio UI
text_gen.py          Model loading, prompt building, generation
tests/               Unit tests (use a fake generator; no model download)
requirements.txt     Runtime dependencies
requirements-dev.txt Runtime + test dependencies
.env.example         Template for optional environment variables
```

## Setup

Requires Python 3.10+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # optional: add HF_TOKEN for faster/authenticated downloads
```

The model (~3 GB) downloads from Hugging Face on first run. It runs on CUDA, Apple Silicon
(MPS) or CPU, picked automatically.

## Run

```bash
python app.py          # UI at http://127.0.0.1:7860
python text_gen.py     # CLI: generate tickets for the built-in samples
```

## Test

```bash
pytest
```

## Help articles

**Hugging Face Transformers**
- [Pipelines tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial): using `pipeline()` for inference
- [Chat templates](https://huggingface.co/docs/transformers/chat_templating): how message lists become model prompts
- [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies): `max_new_tokens`, `do_sample`, `temperature` and more
- [Installation](https://huggingface.co/docs/transformers/installation): setup and cache locations

**Hugging Face Hub**
- [User access tokens](https://huggingface.co/docs/hub/security-tokens): creating an `HF_TOKEN`
- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): the model used here

**Gradio**
- [Quickstart](https://www.gradio.app/guides/quickstart): building a UI with `gr.Interface`

**PyTorch**
- [MPS backend](https://pytorch.org/docs/stable/notes/mps.html): running on Apple Silicon GPUs
