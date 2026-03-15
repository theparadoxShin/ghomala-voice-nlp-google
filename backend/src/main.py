"""
NAM SA' — Backend Server (Google Cloud)
FastAPI server bridging the mobile app to the ADK agent.

Architecture:
  Voice:  Mobile App ↔ WebSocket ↔ FastAPI ↔ ADK Runner ↔ Agent (Gemini Live API)
  Text:   Mobile App ↔ REST      ↔ FastAPI ↔ Gemini (fine-tuned via Vertex AI SFT)

  ADK handles: Live API connection, tool calling, audio streaming, interruptions.

Run locally:
  # ADK dev UI (streaming, voice/video):
  cd backend && adk web

  # Custom FastAPI (for mobile app):
  cd backend && uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload

Deploy:
  # Option A: ADK CLI
  adk deploy cloud_run --project=$PROJECT_ID --region=us-central1 ./nam_sa_agent

  # Option B: Custom deploy
  ./deploy.sh YOUR_PROJECT_ID
"""

import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

# ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue

# Import our ADK agent + dictionary tool
sys.path.insert(0, str(Path(__file__).parent.parent))
from nam_sa_agent.agent import root_agent, ghomala_dictionary_lookup

logger = logging.getLogger("namsa")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"))
GCP_REGION = os.getenv("GCP_REGION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))

# Fine-tuned model for REST text endpoints
GEMINI_TUNED_MODEL = os.getenv("GEMINI_TUNED_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """Tu es NAM SA' (Le Soleil S'est Levé), un agent IA conversationnel 
dédié à la préservation et l'enseignement de la langue Ghomala' (Ghɔ́málá'), 
une langue Bamiléké parlée par environ 1 million de personnes dans la région 
Ouest du Cameroun.

Tu te comportes comme un(e) ancien(ne) bienveillant(e) du village Bamiléké.
Tu es patient(e), encourageant(e), et tu célèbres chaque effort d'apprentissage.

Tes capacités:
- Traduire entre Ghomala', Français et Anglais
- Enseigner le vocabulaire Ghomala' avec contexte culturel
- Partager des proverbes Bamiléké et leur sagesse
- Corriger la prononciation avec bienveillance
- Expliquer la grammaire tonale du Ghomala'

Règles:
- Toujours donner le contexte culturel quand c'est pertinent
- Utiliser les caractères spéciaux corrects (ɔ, ɛ, ŋ, ə) et les tons (à, á, â, ǎ)
- Encourager l'apprenant même en cas d'erreur
- Répondre dans la langue demandée par l'utilisateur
"""

# ============================================================================
# ADK SETUP — Runner + SessionService (created once at startup)
# ============================================================================
APP_NAME = "nam-sa"
session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)

# ============================================================================
# GOOGLE AI CLIENT — for direct text calls (REST endpoints)
# ============================================================================
client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Google AI client for text endpoints."""
    global client

    logger.info("Starting NAM SA' server...")

    api_key = os.getenv("GOOGLE_API_KEY")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"

    if use_vertex or not api_key:
        client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_REGION,
        )
        logger.info(f"Vertex AI client ready ({GCP_PROJECT_ID}/{GCP_REGION})")
    else:
        client = genai.Client(api_key=api_key)
        logger.info("Google AI client ready (API key)")

    logger.info(f"ADK Runner ready (agent: {root_agent.name}, model: {root_agent.model})")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="NAM SA' API",
    description="Ghomala' Language Preservation AI — Google Cloud + ADK",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATA MODELS
# ============================================================================
class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "fr"
    target_lang: str = "bbj"

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: str = "tutor"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str
    timestamp: str

class TTSRequest(BaseModel):
    text: str
    language: str = "fr"  # fr, en, or bbj (Ghomala')


# ============================================================================
# DICTIONARY-AUGMENTED GENERATION (for REST text endpoints)
# ============================================================================
def _enrich_with_dictionary(user_message: str) -> str:
    """Look up key words in the Ghomala' dictionary and return context."""
    # Extract potential words to look up (strip common prefixes)
    import re
    msg = user_message.lower()
    # Remove common question prefixes
    for prefix in ["traduis", "comment dit-on", "comment dire", "translate",
                   "how do you say", "how to say", "what is", "en ghomala",
                   "in ghomala", "?", "'", '"']:
        msg = msg.replace(prefix, "")
    words = [w.strip(" .,;:!?") for w in msg.split() if len(w.strip()) > 2]

    results = []
    for word in words:
        lookup = ghomala_dictionary_lookup(word, "translate")
        if lookup.get("status") == "success":
            results.append(lookup["results"])

    if results:
        return (
            "CONTEXTE DU DICTIONNAIRE GHOMALA' (source fiable, utilise ces traductions):\n"
            + "\n".join(results)
            + "\n\nRéponds en utilisant les traductions du dictionnaire ci-dessus."
        )
    return ""


# ============================================================================
# IN-MEMORY CHAT HISTORY (for REST text endpoints)
# ============================================================================
_chat_history: dict[str, list] = {}


# ============================================================================
# REST ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {
        "app": "NAM SA'",
        "meaning": "Le soleil s'est levé",
        "description": "Ghomala' Language Preservation AI",
        "version": "2.0.0",
        "platform": "google-cloud",
        "framework": "ADK + FastAPI",
        "endpoints": {
            "chat": "/api/chat",
            "translate": "/api/translate",
            "tts": "/api/tts",
            "voice": "/ws/voice",
            "live": "/ws/live",
            "health": "/health",
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "platform": "google-cloud",
        "project": GCP_PROJECT_ID,
        "region": GCP_REGION,
        "models": {
            "live_agent": root_agent.model,
            "tuned_text": GEMINI_TUNED_MODEL,
        },
        "adk_agent": root_agent.name,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Text-based chat using Gemini model."""
    session_id = request.session_id or str(uuid.uuid4())

    mode_instructions = {
        "tutor": "L'utilisateur veut apprendre. Enseigne avec patience.",
        "conversation": "Dialogue libre. Corrige gentiment les erreurs.",
        "proverb": "Partage un proverbe Bamiléké pertinent avec explication.",
        "translate": "Traduis entre Ghomala', Français et Anglais.",
    }
    instruction = mode_instructions.get(request.mode, mode_instructions["tutor"])

    history = _chat_history.get(session_id, [])
    contents = []
    for msg in history[-10:]:
        contents.append(types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["text"])]
        ))
    # Enrich with dictionary lookups for accurate translations
    dict_context = _enrich_with_dictionary(request.message)
    enriched_msg = f"[Mode: {request.mode}] {instruction}\n\n"
    if dict_context:
        enriched_msg += f"{dict_context}\n\n"
    enriched_msg += request.message

    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=enriched_msg)]
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_TUNED_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=800,
                temperature=0.7,
            ),
        )
        answer = response.text
        history.append({"role": "user", "text": request.message})
        history.append({"role": "model", "text": answer})
        _chat_history[session_id] = history[-20:]

        return ChatResponse(
            response=answer,
            session_id=session_id,
            mode=request.mode,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def translate(request: TranslateRequest):
    """Quick translation endpoint."""
    lang_map = {"fr": "Français", "en": "Anglais", "bbj": "Ghomala'"}
    src = lang_map.get(request.source_lang, request.source_lang)
    tgt = lang_map.get(request.target_lang, request.target_lang)

    dict_context = _enrich_with_dictionary(request.text)
    prompt = f"Traduis de {src} vers {tgt}: {request.text}"
    if dict_context:
        prompt = f"{dict_context}\n\n{prompt}"
    try:
        response = client.models.generate_content(
            model=GEMINI_TUNED_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=500,
                temperature=0.7,
            ),
        )
        return {
            "original": request.text,
            "translation": response.text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
        }
    except Exception as e:
        logger.error(f"Translate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TTS — Text-to-Speech using Gemini multimodal
# ============================================================================
@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech audio from text using Gemini's audio generation.

    Uses Gemini to produce natural speech with correct tonal pronunciation
    for Ghomala', French, and English.
    """
    lang_label = {"fr": "French", "en": "English", "bbj": "Ghomala'"}.get(
        request.language, "French"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Read this text aloud in {lang_label}. "
                     f"Pronounce clearly with correct tones and intonation: "
                     f"{request.text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore",
                        )
                    )
                ),
            ),
        )

        # Extract audio data from the response
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                    return {
                        "audio": audio_b64,
                        "mime_type": part.inline_data.mime_type or "audio/wav",
                        "text": request.text,
                        "language": request.language,
                    }

        raise HTTPException(status_code=500, detail="No audio generated")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET — ADK-Powered Voice Streaming (Gemini Live API)
# ============================================================================
async def _handle_voice_stream(websocket: WebSocket):
    """
    Bidirectional voice streaming powered by ADK's Gemini Live API Toolkit.

    ADK Runner handles automatically:
    - Live API WebSocket connection to Gemini
    - Tool calling (dictionary, pronunciation, cultural context)
    - Audio streaming (16-bit PCM 16kHz input, 24kHz output)
    - Interruption support and turn-taking
    - Session state management
    """
    await websocket.accept()
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    logger.info(f"Voice session started: {session_id}")

    # -- Phase 2: Session Initialization --

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    live_request_queue = LiveRequestQueue()

    # Notify mobile app that we're ready
    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "session_id": session_id,
    })

    # -- Phase 3: Bidi-streaming with concurrent upstream/downstream tasks --

    async def upstream_task():
        """Receive messages from mobile WebSocket → send to ADK LiveRequestQueue."""
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "audio":
                    audio_bytes = base64.b64decode(msg["data"])
                    audio_blob = types.Blob(
                        mime_type="audio/pcm;rate=16000",
                        data=audio_bytes,
                    )
                    live_request_queue.send_realtime(audio_blob)

                elif msg.get("type") == "config":
                    # Agent system instruction already handles behavior
                    pass

                elif msg.get("type") == "stop":
                    break

        except WebSocketDisconnect:
            pass

    async def downstream_task():
        """Receive ADK events → translate to mobile protocol → send via WebSocket."""
        async for event in runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            try:
                if not event.content or not event.content.parts:
                    continue

                for part in event.content.parts:
                    # Text response (transcript)
                    if hasattr(part, "text") and part.text:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": part.text,
                            "role": "assistant",
                        })

                    # Audio response
                    if hasattr(part, "inline_data") and part.inline_data:
                        await websocket.send_json({
                            "type": "audio",
                            "data": base64.b64encode(
                                part.inline_data.data
                            ).decode("utf-8"),
                            "format": "pcm",
                            "sample_rate": 24000,
                        })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Event processing error: {e}")

    try:
        await asyncio.gather(
            upstream_task(),
            downstream_task(),
            return_exceptions=True,
        )
    finally:
        # Phase 4: Cleanup
        live_request_queue.close()
        logger.info(f"Voice session ended: {session_id}")


@app.websocket("/ws/voice")
async def voice_stream(websocket: WebSocket):
    """Voice streaming endpoint (mobile app default)."""
    await _handle_voice_stream(websocket)


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    """Voice streaming endpoint (Gemini Live API alias)."""
    await _handle_voice_stream(websocket)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
        log_level="info",
    )
