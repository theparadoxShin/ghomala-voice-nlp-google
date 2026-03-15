# SFT — Supervised Fine-Tuning Explained

## What is SFT?

```
You show the model:  "Question: Comment dit-on bonjour en Ghomala'?"
                     "Answer: En Ghomala', bonjour se dit àkə̀"

The model learns:    "When asked about Ghomala' greetings → respond like this"
```

**How it works:**
1. You give labeled pairs: (input → expected output) in JSONL format
2. The model adjusts its weights to match your examples
3. After training, it generates similar responses for new inputs

**When to use:** Teaching NEW KNOWLEDGE (Ghomala' vocabulary, grammar, cultural context)

**Our data:** ~3,000-5,000 conversation pairs from MAFAND-MT + dictionary + manual entries

---

## Vertex AI JSONL Format

Each line in our training JSONL follows the Gemini SFT format:

```json
{
  "systemInstruction": {
    "role": "system",
    "parts": [{"text": "Tu es NAM SA', un agent de préservation du Ghomala'..."}]
  },
  "contents": [
    {"role": "user", "parts": [{"text": "Comment dit-on bonjour en Ghomala' ?"}]},
    {"role": "model", "parts": [{"text": "En Ghomala', bonjour se dit àkə̀..."}]}
  ]
}
```

**Key differences from other platforms:**
- Role is `"model"` (not `"assistant"`)
- System prompt goes in `"systemInstruction"` (not `"system"`)
- Each message has `"parts"` array with `"text"` objects

---

## Our Strategy

```
Step 1: SFT → Teach the model Ghomala'
        Base model: Gemini 2.5 Flash
        Input: 3,000+ conversation pairs in JSONL
        Result: Model understands Ghomala' vocabulary and culture

Step 2: Evaluate
        Use Vertex AI Model Evaluation to check quality
        Test with real Ghomala' prompts

Step 3: Deploy
        The fine-tuned model gets an endpoint
        Backend uses it for text chat and translation
```

## Supported Models for SFT

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| Gemini 2.5 Flash-Lite | Fastest | Good | Lowest |
| **Gemini 2.5 Flash** | Fast | **Very Good** | **Moderate** |
| Gemini 2.5 Pro | Slower | Best | Highest |

We use **Gemini 2.5 Flash** — best balance of quality and cost for our use case.

---

## Pipeline Summary

```
01_download_datasets.py    → Download from HuggingFace
02_transform_to_jsonl.py   → Convert to Vertex AI SFT JSONL
02_2_validate_jsonl.py     → Validate format before upload
03_upload_to_gcs.py        → Upload to GCS bucket
04_launch_fine_tuning.py   → Launch SFT job on Vertex AI
```
