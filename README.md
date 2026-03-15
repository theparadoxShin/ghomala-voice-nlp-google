# NAM SA' — The Sun Has Risen ☀️

> AI-powered live voice agent for preserving the Ghomala' language using Google ADK, Gemini Live API & Vertex AI fine-tuning

An elder-like AI tutor that teaches and preserves **Ghomala' (Ghɔ́málá')**, a Bamiléké language spoken by ~1 million people in western Cameroon. The app supports **real-time voice conversations**, text translation, proverbs, and cultural teaching — powered by Google's Agent Development Kit (ADK) and Gemini models.

---

## 🏗️ Repo Structure

```
nam-sa-google/
├── data/                              # 📊 Dataset & fine-tuning pipeline
│   ├── scripts/
│   │   ├── 00_extract_dictionary_from_pdf.py  # Extract dictionary from PDF
│   │   ├── 01_download_datasets.py            # Download Masakhane datasets
│   │   ├── 02_transform_to_jsonl.py           # Convert to Vertex AI SFT JSONL format
│   │   ├── 02_2_validate_jsonl.py             # Validate JSONL for Gemini SFT
│   │   ├── 03_upload_to_gcs.py                # Upload train/val.jsonl to GCS
│   │   └── 04_launch_fine_tuning.py           # Launch SFT job on Vertex AI
│   ├── raw/                                   # Raw downloaded datasets
│   ├── processed/                             # Final JSONL files
│   └── dictionary/
│       └── ghomala_dictionary.json            # Curated Ghomala' dictionary entries
│
├── backend/                           # 🖥️ Backend (Cloud Run)
│   ├── nam_sa_agent/                          # Google ADK agent package
│   │   ├── __init__.py                        # ADK package init
│   │   ├── agent.py                           # Root agent + tools (dictionary, pronunciation, culture)
│   │   └── .env                               # Environment config (API key or Vertex AI)
│   ├── src/
│   │   └── main.py                            # FastAPI server (REST + WebSocket for mobile)
│   ├── Dockerfile                             # Python 3.11, uvicorn on port 8080
│   └── requirements.txt                       # google-adk, google-genai, fastapi, etc.
│
├── mobile/                            # 📱 React Native / Expo app
│   ├── App.js                                 # Navigation stack (Home → Conversation)
│   ├── src/
│   │   ├── screens/
│   │   │   ├── HomeScreen.js                  # Landing: 4 mode cards + voice CTA + quick phrases
│   │   │   └── ConversationScreen.js          # Chat UI: voice waveform, text input, message bubbles
│   │   ├── services/
│   │   │   └── api.js                         # REST (/api/chat, /api/translate) + WebSocket (/ws/voice)
│   │   └── theme/
│   │       └── index.js                       # Ghomala' cultural design system (maroon, gold, green)
│   ├── app.json                               # Expo config
│   └── package.json                           # Expo 50, React Native 0.73
│
├── docs/                              # 📚 Documentation
│   └── fine-tuning/
│       ├── gcp_setup.md                       # GCP IAM & API setup for SFT
│       └── SFT_explained.md                   # SFT strategy explained
│
├── deploy.sh                          # One-command deploy to Cloud Run
├── cloudbuild.yaml                    # CI/CD: Cloud Build → Cloud Run
├── requirements.txt                   # Data pipeline deps (datasets, google-cloud-*)
└── README.md
```

---

## 🚀 Quick Start

### 1. Fine-Tuning Pipeline (data → model)

```bash
# Install data pipeline dependencies
pip install -r requirements.txt

# Download Masakhane/Ghomala datasets from HuggingFace
cd data/scripts
python 01_download_datasets.py

# Transform to Vertex AI Gemini SFT JSONL format
python 02_transform_to_jsonl.py              # → capped at 20,000 samples
python 02_transform_to_jsonl.py --no-limit   # → all samples

# Validate the JSONL format
python 02_2_validate_jsonl.py

# Upload to GCS bucket
python 03_upload_to_gcs.py

# Launch SFT fine-tuning on Vertex AI
python 04_launch_fine_tuning.py --mode sft
```

### 2. Backend (FastAPI + ADK Agent)

```bash
cd backend

# Install backend dependencies
pip install -r requirements.txt

# Configure: copy .env and fill in your API key or Vertex AI project
cp nam_sa_agent/.env.example nam_sa_agent/.env

# Run locally (FastAPI server for mobile app)
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

# Or run the ADK agent directly (built-in Web UI)
adk web nam_sa_agent

# Or deploy to Cloud Run
./deploy.sh YOUR_PROJECT_ID
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Text conversation (Gemini 2.5 Flash, fine-tuned) |
| `POST` | `/api/translate` | Translation (FR ↔ Ghomala' ↔ EN) |
| `WS` | `/ws/voice` | Bidirectional voice streaming (Gemini Live API) |

### 3. Mobile App (Expo)

```bash
cd mobile
npm install
npx expo start
```

**Modes:** Tutor (📖) · Conversation (💬) · Proverbs (🌿) · Translate (🔄)

---

## 🧠 Training Strategy

| Phase | Method | Data | Goal |
|-------|--------|------|------|
| 1 | **SFT** | ~3,000-5,000 conversation pairs | Teach Ghomala' vocabulary & grammar |

**Data format (Vertex AI Gemini SFT):**
```json
{
  "systemInstruction": {"role": "system", "parts": [{"text": "Tu es NAM SA'..."}]},
  "contents": [
    {"role": "user", "parts": [{"text": "Comment dit-on bonjour en Ghomala' ?"}]},
    {"role": "model", "parts": [{"text": "En Ghomala', bonjour se dit àkə̀..."}]}
  ]
}
```

**Supported base models:** Gemini 2.5 Flash, Gemini 2.5 Flash-Lite, Gemini 2.5 Pro

See [docs/fine-tuning/SFT_explained.md](docs/fine-tuning/SFT_explained.md) for details.

---

## ⚙️ Architecture

```
┌─────────────┐         ┌──────────────────┐        ┌──────────────────────────────────┐
│  Mobile App │──REST──►│                  │──────►│  Gemini 2.5 Flash (fine-tuned)    │
│  (Expo)     │         │  FastAPI          │        │  Text chat & translation         │
│             │──WS────►│  on Cloud Run     │        └──────────────────────────────────┘
│  PCM 16kHz  │◄═══════►│  (port 8080)      │        ┌──────────────────────────────────┐
│  streaming  │ binary  │                  │◄══════►│  ADK Runner + Gemini Live API     │
│  audio      │ audio   │  ADK Runner       │ bidi  │  Native audio streaming            │
└─────────────┘ frames  │  run_live()       │        │  + automatic tool calling         │
                        │                  │        └──────────────────────────────────┘
                        │                  │        ┌──────────────────────────────────┐
                        │                  │──────►│  ADK Agent Tools                  │
                        │                  │        │  Dictionary / Pronunciation /     │
                        └──────────────────┘        │  Cultural Context                │
                                │                   └──────────────────────────────────┘
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
        │  ADK Agent   │ │  ADK       │ │  GCS        │
        │  (root_agent)│ │  Sessions  │ │  Training   │
        └──────────────┘ └────────────┘ └─────────────┘
```

### Voice Pipeline (ADK + Gemini Live API)

ADK's **Gemini Live API Toolkit** handles the entire voice streaming pipeline. The `Runner.run_live()` method establishes a persistent Live API connection, manages `LiveRequestQueue` for upstream audio, and yields `Event` objects with audio/text responses. Tool calling (dictionary lookup, pronunciation, cultural context) is executed **automatically** by ADK during the streaming session — no manual WebSocket or tool orchestration code needed.

### Key Services

- **ADK (Agent Development Kit)**: Core framework — `Runner.run_live()` + `LiveRequestQueue` for bidirectional voice streaming with automatic tool calling
- **Gemini Live API** (native audio model): Real-time speech-to-speech via ADK's Gemini Live API Toolkit
- **Gemini 2.5 Flash** (fine-tuned via Vertex AI SFT): text chat & translation with Ghomala' knowledge
- **Cloud Run**: Managed container hosting for WebSocket connections (auto-scaling)
- **GCS**: Training data storage for Vertex AI fine-tuning

---

## 📋 Hackathon
- **Competition:** Google Cloud — Live Agent Challenge
- **Category:** Live Agents 🗣️
- **Requirements:** Gemini Live API or ADK, hosted on Google Cloud

## 📜 License
MIT
