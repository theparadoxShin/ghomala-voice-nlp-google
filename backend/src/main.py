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

def _load_dict_data():
    """Load the raw dictionary list for direct lookups."""
    from nam_sa_agent.agent import _load_dictionary
    return _load_dictionary()


def _extract_best_dict_translation(text: str, source_lang: str, target_lang: str) -> str | None:
    """Find the best dictionary translation for a word/short phrase.
    Returns the translated word directly, or None if no good match."""
    import re
    dictionary = _load_dict_data()
    text_lower = text.lower().strip()

    # Skip very long phrases (dictionary is word-level)
    if len(text_lower.split()) > 4:
        return None

    if target_lang == "bbj":
        # Source is fr or en → find matching french entry → return ghomala
        scored = []
        for entry in dictionary:
            french = entry.get("french", "").lower().strip()
            ghomala = entry.get("ghomala", "").strip()
            if not ghomala or not french:
                continue
            # Skip entries where ghomala is itself a French word (bad entries)
            if ghomala.lower() == french:
                continue

            # Score matches
            first_def = french.split(",")[0].split("(")[0].strip().rstrip(".")
            # Remove category prefixes like "Vt." "N." etc.
            first_def_clean = re.sub(r'^(vt\.|vi\.|v\.|n\.|adj\.|adv\.)\s*', '', first_def).strip()

            score = 0
            if first_def_clean == text_lower:
                score = 100  # Perfect match: "remercier" == "remercier"
            elif first_def == text_lower:
                score = 100
            elif french == text_lower:
                score = 95
            elif french.startswith(text_lower + " ") or french.startswith(text_lower + "."):
                score = 90
            # Stem matching: "merci" matches "remercier", "remerciement"
            elif text_lower in first_def_clean and len(text_lower) >= 3:
                score = 70

            if score > 0:
                # Prefer verbs (Vt, Vi, V) for action words
                cat = entry.get("category", "").strip().rstrip(".")
                if cat in ("Vt", "Vi", "V"):
                    score += 5
                scored.append((score, ghomala, entry))

        if scored:
            scored.sort(key=lambda x: -x[0])
            best_ghomala = scored[0][1]
            # Clean subscript numbers (pîŋ₂ → pîŋ)
            return re.sub(r'[₀₁₂₃₄₅₆₇₈₉]+$', '', best_ghomala)

    elif source_lang == "bbj":
        # Source is ghomala → find matching ghomala entry → return french
        text_clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]+$', '', text_lower)
        for entry in dictionary:
            ghomala = entry.get("ghomala", "").lower().strip()
            ghomala_clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]+$', '', ghomala)
            if ghomala_clean == text_clean or ghomala == text_lower:
                french = entry.get("french", "").strip()
                if french:
                    # Clean: remove "Vt." prefixes, take first definition
                    french_clean = re.sub(r'^(Vt\.|Vi\.|V\.|N\.|Adj\.|Adv\.)\s*', '', french).strip()
                    first = french_clean.split(".")[0].split(",")[0].strip()
                    if first and first.lower() != ghomala_clean:
                        return first

    return None


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
    """Quick translation endpoint — dictionary-first, model fallback."""
    lang_map = {"fr": "Français", "en": "Anglais", "bbj": "Ghomala'"}
    src = lang_map.get(request.source_lang, request.source_lang)
    tgt = lang_map.get(request.target_lang, request.target_lang)

    # ── STEP 1: Try dictionary lookup first (ground truth) ──
    dictionary = ghomala_dictionary_lookup(request.text.strip(), "translate")
    if dictionary.get("status") == "success":
        dict_translation = _extract_best_dict_translation(
            request.text.strip(), request.source_lang, request.target_lang
        )
        if dict_translation:
            return {
                "original": request.text,
                "translation": dict_translation,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "source": "dictionary",
            }

    # ── STEP 2: Fallback to model (for phrases/sentences not in dictionary) ──
    dict_context = _enrich_with_dictionary(request.text)
    prompt = f"Traduis de {src} vers {tgt}: {request.text}"
    if dict_context:
        prompt = f"{dict_context}\n\n{prompt}"

    translate_instruction = (
        "Tu es un traducteur automatique Français-Anglais-Ghomala'. "
        "RÈGLE ABSOLUE: Réponds UNIQUEMENT avec la traduction. "
        "JAMAIS d'explication, JAMAIS de commentaire, JAMAIS de phrase comme "
        "'je ne peux pas' ou 'le dictionnaire ne contient pas'. "
        "Si tu ne connais pas la traduction exacte, donne ta meilleure approximation phonétique. "
        "Réponds UNIQUEMENT avec les mots traduits, rien d'autre."
    )

    # Retry with backoff for transient 429 errors
    last_err = None
    for attempt in range(3):
        try:
            if attempt > 0:
                await asyncio.sleep(1.5 * attempt)
            response = client.models.generate_content(
                model=GEMINI_TUNED_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=translate_instruction,
                    max_output_tokens=200,
                    temperature=0.1,
                ),
            )
            result_text = response.text.strip()

            # Clean model hallucinations — if response contains explanations, extract just the translation
            for noise in [
                "The provided dictionary", "Le dictionnaire", "Je ne peux pas",
                "I cannot", "I don't have", "Il n'y a pas", "Malheureusement",
                "Unfortunately", "Note:", "Remarque:",
            ]:
                if noise.lower() in result_text.lower():
                    # Try to extract a Ghomala'-like word (with diacritics) from the noise
                    import re
                    ghomala_words = re.findall(r'[A-Za-zɔɛŋəʉÀ-ÿ\u0300-\u036f]+', result_text)
                    # Filter for words with special chars typical of Ghomala'
                    special = [w for w in ghomala_words if any(c in w for c in 'ɔɛŋəʉ') or len(w) > 2]
                    if special:
                        result_text = ' '.join(special[:5])
                    else:
                        result_text = "?"
                    break

            return {
                "original": request.text,
                "translation": result_text,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "source": "model",
            }
        except Exception as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Translate 429 (attempt {attempt+1}/3): {e}")
                continue
            logger.error(f"Translate error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    logger.error(f"Translate failed after 3 retries: {last_err}")
    raise HTTPException(status_code=429, detail="Service busy, please retry")


# ============================================================================
# TTS — Text-to-Speech using Google Cloud TTS API
# ============================================================================
tts_client = None

def _get_tts_client():
    """Lazy-init Cloud TTS client."""
    global tts_client
    if tts_client is None:
        from google.cloud import texttospeech
        tts_client = texttospeech.TextToSpeechClient()
    return tts_client


# Chirp 3 HD voice mapping — multilingual, handles Unicode diacritics/tones
CHIRP3_VOICES = {
    "fr": ("fr-FR", "fr-FR-Chirp3-HD-Aoede"),
    "en": ("en-US", "en-US-Chirp3-HD-Aoede"),
    "bbj": ("fr-FR", "fr-FR-Chirp3-HD-Aoede"),  # Ghomala' → French Chirp3 (best for tonal diacritics)
}


def _build_tts_input(text: str, language: str):
    """Build SSML or plain text input for TTS.
    For Ghomala' (bbj), wraps in SSML with prosody hints for tonal pronunciation."""
    from google.cloud import texttospeech

    if language == "bbj":
        # SSML with slow prosody for tonal language learning
        ssml = (
            '<speak>'
            '<prosody rate="slow" pitch="+0st">'
            f'{_escape_ssml(text)}'
            '</prosody>'
            '</speak>'
        )
        return texttospeech.SynthesisInput(ssml=ssml)
    return texttospeech.SynthesisInput(text=text)


def _escape_ssml(text: str) -> str:
    """Escape XML special characters for SSML."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech audio using Google Cloud TTS — Chirp 3 HD.

    Uses Chirp 3 HD voices for superior multilingual pronunciation.
    Handles Unicode diacritics and tonal markers for Ghomala'.
    """
    from google.cloud import texttospeech

    lang_code, voice_name = CHIRP3_VOICES.get(request.language, CHIRP3_VOICES["fr"])

    try:
        tts = _get_tts_client()
        synthesis_input = _build_tts_input(request.text, request.language)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        )
        response = tts.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_b64 = base64.b64encode(response.audio_content).decode("utf-8")
        return {
            "audio": audio_b64,
            "mime_type": "audio/mp3",
            "text": request.text,
            "language": request.language,
        }
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET — Voice Conversation (Multimodal Gemini + Cloud TTS)
# ============================================================================

async def _handle_live_voice(websocket: WebSocket):
    """Real-time voice conversation using multimodal Gemini + Cloud TTS.

    Flow per turn:
      1. Mobile sends recorded audio over WebSocket
      2. Gemini Flash transcribes the audio
      3. Tuned model generates a response (with dictionary enrichment)
      4. Cloud TTS generates audio for the response
      5. Transcript + audio sent back to mobile
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    chat_history: list[dict] = []

    logger.info(f"Live voice session started: {session_id}")

    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "session_id": session_id,
    })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            logger.info(f"[{session_id[:8]}] Received message type: {msg.get('type')}, size: {len(data)} bytes")

            if msg.get("type") == "audio":
                audio_bytes = base64.b64decode(msg["data"])
                mime_type = msg.get("mime_type", "audio/mp4")
                logger.info(f"[{session_id[:8]}] Audio received: {len(audio_bytes)} bytes, mime: {mime_type}")

                try:
                    # ── Step 1: Transcribe audio ──
                    logger.info(f"[{session_id[:8]}] Step 1: Transcribing audio...")
                    transcribe_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[types.Content(
                            role="user",
                            parts=[
                                types.Part(inline_data=types.Blob(
                                    data=audio_bytes, mime_type=mime_type,
                                )),
                                types.Part(text=(
                                    "Transcris exactement ce que l'utilisateur dit "
                                    "dans cet audio. Donne UNIQUEMENT la transcription, "
                                    "rien d'autre. Pas de guillemets."
                                )),
                            ]
                        )],
                        config=types.GenerateContentConfig(
                            max_output_tokens=200,
                            temperature=0.1,
                        ),
                    )
                    user_text = transcribe_response.text.strip().strip('"').strip("'")
                    logger.info(f"[{session_id[:8]}] Transcription: '{user_text}'")

                    if not user_text:
                        logger.warning(f"[{session_id[:8]}] Empty transcription, skipping")
                        await websocket.send_json({"type": "turn_complete"})
                        continue

                    # Send user transcript to mobile
                    await websocket.send_json({
                        "type": "user_transcript",
                        "text": user_text,
                    })
                    chat_history.append({"role": "user", "text": user_text})

                    # ── Step 2: Generate response with dictionary enrichment ──
                    logger.info(f"[{session_id[:8]}] Step 2: Generating response...")
                    dict_context = _enrich_with_dictionary(user_text)
                    enriched = ""
                    if dict_context:
                        enriched = f"{dict_context}\n\n"
                    enriched += user_text

                    contents = []
                    for h in chat_history[-8:]:
                        contents.append(types.Content(
                            role=h["role"],
                            parts=[types.Part(text=h["text"])]
                        ))
                    # Replace last entry with enriched version
                    if contents:
                        contents[-1] = types.Content(
                            role="user",
                            parts=[types.Part(text=enriched)]
                        )

                    response = client.models.generate_content(
                        model=GEMINI_TUNED_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=500,
                            temperature=0.7,
                        ),
                    )
                    assistant_text = response.text.strip()
                    logger.info(f"[{session_id[:8]}] Response: '{assistant_text[:100]}...'")

                    # Send assistant response text
                    await websocket.send_json({
                        "type": "transcript",
                        "text": assistant_text,
                        "role": "assistant",
                    })
                    chat_history.append({"role": "model", "text": assistant_text})

                    # ── Step 3: Generate TTS audio (Chirp 3 HD) ──
                    logger.info(f"[{session_id[:8]}] Step 3: Generating TTS audio (Chirp 3 HD)...")
                    try:
                        from google.cloud import texttospeech as tts_lib
                        tts = _get_tts_client()

                        lang_code, voice_name = CHIRP3_VOICES["fr"]

                        synthesis_input = tts_lib.SynthesisInput(text=assistant_text)
                        voice = tts_lib.VoiceSelectionParams(
                            language_code=lang_code, name=voice_name,
                        )
                        audio_config = tts_lib.AudioConfig(
                            audio_encoding=tts_lib.AudioEncoding.MP3,
                        )
                        tts_response = tts.synthesize_speech(
                            input=synthesis_input, voice=voice,
                            audio_config=audio_config,
                        )

                        audio_b64 = base64.b64encode(
                            tts_response.audio_content
                        ).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio_response",
                            "data": audio_b64,
                            "format": "mp3",
                        })
                        logger.info(f"[{session_id[:8]}] Audio response sent: {len(audio_b64)} chars b64")
                    except Exception as tts_err:
                        logger.warning(f"[{session_id[:8]}] TTS in live session: {tts_err}")

                    await websocket.send_json({"type": "turn_complete"})
                    logger.info(f"[{session_id[:8]}] Turn complete")

                except Exception as e:
                    logger.error(f"[{session_id[:8]}] Live voice processing error: {e}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
                    await websocket.send_json({"type": "turn_complete"})

            elif msg.get("type") == "config":
                logger.info(f"[{session_id[:8]}] Config received: {msg}")
                pass

            elif msg.get("type") == "stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        logger.info(f"Live voice session ended: {session_id}")


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
    """Voice conversation endpoint — multimodal Gemini + Cloud TTS."""
    await _handle_live_voice(websocket)


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
