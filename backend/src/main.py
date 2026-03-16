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

    # REST endpoints always use Vertex AI (for SFT v2 endpoint access)
    # ADK Runner uses GOOGLE_API_KEY for Google AI Studio Live API
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
    )
    logger.info(f"Vertex AI client ready ({GCP_PROJECT_ID}/{GCP_REGION})")

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.info("GOOGLE_API_KEY set — ADK will use Google AI Studio for Live API")
    else:
        logger.warning("GOOGLE_API_KEY not set — ADK Live API may fail on Vertex AI")

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

            # Only clean obvious hallucinations (long explanations instead of short translations)
            if len(result_text) > 60:
                for noise in [
                    "The provided dictionary", "Le dictionnaire", "Je ne peux pas",
                    "I cannot", "I don't have", "Il n'y a pas", "Malheureusement",
                    "Unfortunately",
                ]:
                    if noise.lower() in result_text.lower():
                        # Try to extract a Ghomala'-like word from the noise
                        import re
                        ghomala_words = re.findall(r'[\w\u0250-\u02AF\u0300-\u036f]+', result_text)
                        special = [w for w in ghomala_words if any(c in w for c in 'ɔɛŋəʉ')]
                        if special:
                            result_text = ' '.join(special[:5])
                        # else keep the raw model response — better than "?"
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
# AUDIO FORMAT CONVERSION — M4A/AAC ↔ PCM (for Gemini Live API)
# ============================================================================

def _convert_audio_to_pcm(audio_bytes: bytes, mime_type: str = "audio/mp4") -> bytes:
    """Convert audio (M4A/AAC/MP3/WAV) to raw PCM 16kHz mono 16-bit.
    Required because Gemini Live API only accepts audio/pcm;rate=16000."""
    from pydub import AudioSegment
    import io

    fmt_map = {
        "audio/mp4": "mp4", "audio/m4a": "mp4", "audio/aac": "aac",
        "audio/mpeg": "mp3", "audio/mp3": "mp3",
        "audio/wav": "wav", "audio/x-wav": "wav",
        "audio/webm": "webm", "audio/ogg": "ogg",
    }
    fmt = fmt_map.get(mime_type, "mp4")

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio.raw_data


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM bytes to WAV format (adds 44-byte header)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _clean_transcript(text: str) -> str:
    """Remove markdown thinking headers from model transcripts.
    E.g. '**Initiating A Dialogue**\n\nHello!' → 'Hello!'"""
    import re
    text = re.sub(r'\*\*[^*]+\*\*\s*', '', text)
    return text.strip()


# ============================================================================
# WEBSOCKET — ADK Live Voice (Gemini Live API — Bidirectional Streaming)
# ============================================================================

async def _handle_live_voice(websocket: WebSocket):
    """Real-time bidirectional voice conversation using ADK + Gemini Live API.

    Architecture (following Google's Way Back Home Codelab pattern):
      - LiveRequestQueue buffers audio from client
      - Runner.run_live() streams audio to Gemini Live API
      - Gemini native audio model processes speech + calls tools (dictionary, SFT model)
      - Model responds with audio (voice) + transcription
      - Audio is converted to WAV and sent back to mobile client

    This replaces the old manual STT → LLM → TTS pipeline with true
    bidirectional streaming. The model handles turn detection internally.
    """
    await websocket.accept()
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    language = "fr"
    sid = session_id[:8]

    logger.info(f"ADK Live session started: {session_id}")

    # -- Phase 1: Session Setup --
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    # -- Phase 2: RunConfig for native audio model --
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    live_request_queue = LiveRequestQueue()

    await websocket.send_json({
        "type": "status",
        "status": "ready",
        "session_id": session_id,
    })

    # -- Phase 3: Concurrent bidirectional streaming --

    async def upstream_task():
        """Receive audio/config from mobile → feed to LiveRequestQueue."""
        nonlocal language
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "config":
                    language = msg.get("language", "fr")
                    logger.info(f"[{sid}] Config: lang={language}")
                    # Don't send a greeting — wait for user to speak first.
                    # The agent's system instruction handles language detection.

                elif msg.get("type") == "audio":
                    audio_bytes = base64.b64decode(msg["data"])
                    mime = msg.get("mime_type", "audio/mp4")
                    logger.info(f"[{sid}] Audio received: {len(audio_bytes)} bytes, mime={mime}")

                    try:
                        # Convert M4A/AAC → raw PCM 16kHz mono 16-bit
                        pcm_data = await asyncio.to_thread(
                            _convert_audio_to_pcm, audio_bytes, mime
                        )
                        audio_blob = types.Blob(
                            mime_type="audio/pcm;rate=16000",
                            data=pcm_data,
                        )
                        live_request_queue.send_realtime(audio_blob)
                        logger.info(f"[{sid}] PCM fed to queue: {len(pcm_data)} bytes")
                    except Exception as conv_err:
                        logger.error(f"[{sid}] Audio conversion error: {conv_err}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Audio format error",
                        })

                elif msg.get("type") == "text":
                    text = msg.get("text", "")
                    if text:
                        live_request_queue.send_content(
                            types.Content(parts=[types.Part(text=text)])
                        )

                elif msg.get("type") == "stop":
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[{sid}] Upstream error: {e}")

    async def downstream_task():
        """Receive events from Runner.run_live() → send to mobile."""
        audio_buffer = bytearray()
        transcript_text = ""

        try:
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                try:
                    # -- User input transcription --
                    input_tr = getattr(event, "input_audio_transcription", None)
                    if input_tr:
                        final = getattr(input_tr, "final_transcript", None)
                        if final:
                            logger.info(f"[{sid}] USER: {final}")
                            await websocket.send_json({
                                "type": "user_transcript",
                                "text": final,
                            })

                    # -- Model output transcription --
                    output_tr = getattr(event, "output_audio_transcription", None)
                    if output_tr:
                        final = getattr(output_tr, "final_transcript", None)
                        if final:
                            transcript_text = final
                            logger.info(f"[{sid}] NAM SA': {final}")

                    # -- Tool calls (logged for debugging) --
                    if hasattr(event, "tool_call") and event.tool_call:
                        logger.info(f"[{sid}] Tool call: {event.tool_call}")

                    # -- Audio + text content --
                    has_audio = False
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                if part.inline_data.data:
                                    audio_buffer.extend(part.inline_data.data)
                                    has_audio = True
                            if hasattr(part, "text") and part.text:
                                if not transcript_text:
                                    transcript_text = part.text

                    # -- Detect turn completion --
                    is_turn_complete = False
                    if getattr(event, "turn_complete", False):
                        is_turn_complete = True
                    if not is_turn_complete:
                        try:
                            raw = json.loads(
                                event.model_dump_json(exclude_none=True, by_alias=True)
                            )
                            sc = raw.get("serverContent", raw.get("server_content", {})) or {}
                            if sc.get("turnComplete", sc.get("turn_complete", False)):
                                is_turn_complete = True
                        except Exception:
                            pass

                    # -- Flush on turn complete --
                    if is_turn_complete:
                        if transcript_text:
                            cleaned = _clean_transcript(transcript_text)
                            if cleaned:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": cleaned,
                                })
                            transcript_text = ""

                        if audio_buffer:
                            wav_data = _pcm_to_wav(bytes(audio_buffer))
                            audio_buffer = bytearray()
                            await websocket.send_json({
                                "type": "audio_response",
                                "data": base64.b64encode(wav_data).decode(),
                                "format": "wav",
                            })

                        await websocket.send_json({"type": "turn_complete"})

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.warning(f"[{sid}] Event error: {e}")

        except Exception as e:
            logger.error(f"[{sid}] Downstream error: {e}")

    # Run both tasks concurrently (full-duplex)
    try:
        await asyncio.gather(upstream_task(), downstream_task())
    except WebSocketDisconnect:
        logger.info(f"[{sid}] Client disconnected")
    except Exception as e:
        logger.error(f"[{sid}] Session error: {e}")
    finally:
        # -- Phase 4: Cleanup --
        live_request_queue.close()
        logger.info(f"ADK Live session ended: {session_id}")


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
    """ADK Live Voice — bidirectional streaming via Gemini Live API."""
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
