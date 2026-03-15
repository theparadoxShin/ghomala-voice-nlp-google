"""
NAM SA' — ADK Agent Definition
"Le Soleil S'est Levé" — Ghomala' Language Preservation AI

This is the core agent definition using Google's Agent Development Kit (ADK).
It defines the root_agent with tools for Ghomala' dictionary lookup,
translation, cultural explanations, and pronunciation correction.

The agent uses Gemini Live API for real-time bidirectional voice streaming
and a fine-tuned Gemini Flash model (via Vertex AI SFT) as knowledge source.
"""

import json
import logging
import os
from pathlib import Path

from google.adk.agents import Agent
from google.genai import types

logger = logging.getLogger("namsa")

# ============================================================================
# CONFIGURATION
# ============================================================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

# Fine-tuned model for text/knowledge (from Vertex AI SFT)
# After fine-tuning, replace with your tuned model endpoint
GEMINI_TUNED_MODEL = os.getenv(
    "GEMINI_TUNED_MODEL",
    "gemini-2.5-flash"  # fallback to base until fine-tuned
)

# Live API model for streaming voice (set in root_agent)
# Google AI Studio: gemini-2.5-flash-native-audio-preview-12-2025
# Vertex AI:        gemini-live-2.5-flash-native-audio
GEMINI_LIVE_MODEL = os.getenv(
    "GEMINI_LIVE_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025"
)

# ============================================================================
# SYSTEM INSTRUCTION
# ============================================================================
SYSTEM_INSTRUCTION = """Tu es NAM SA' (Le Soleil S'est Levé), un agent IA conversationnel 
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
# DICTIONARY DATA (loaded once at import time)
# ============================================================================
_dictionary_data = None


def _load_dictionary() -> list:
    """Load the Ghomala' dictionary JSON file."""
    global _dictionary_data
    if _dictionary_data is not None:
        return _dictionary_data

    # Search multiple paths (dev layout vs Docker container)
    candidates = [
        Path(os.getenv("DICTIONARY_PATH", "")),
        Path(__file__).parent.parent.parent / "data" / "dictionary" / "ghomala_dictionary.json",
        Path(__file__).parent.parent / "data" / "dictionary" / "ghomala_dictionary.json",
        Path("/app/data/dictionary/ghomala_dictionary.json"),
    ]
    for dict_path in candidates:
        if dict_path.is_file():
            with open(dict_path, "r", encoding="utf-8") as f:
                _dictionary_data = json.load(f)
            logger.info(f"Dictionary loaded: {len(_dictionary_data)} entries from {dict_path}")
            return _dictionary_data

    _dictionary_data = []
    logger.warning("Dictionary not found in any search path")
    return _dictionary_data


# ============================================================================
# TOOL FUNCTIONS — ADK tools are simple Python functions
# ============================================================================

def ghomala_dictionary_lookup(query: str, lookup_type: str = "translate") -> dict:
    """Look up a word or phrase in the Ghomala' dictionary.

    Use this tool when the user asks for a translation, vocabulary word,
    cultural context, example sentence, or proverb related to a Ghomala' word.

    Args:
        query: The word or phrase to look up (in French, English, or Ghomala').
        lookup_type: Type of lookup — one of "translate", "define", "example", "proverb".

    Returns:
        dict: A dictionary with status and the lookup result.
    """
    dictionary = _load_dictionary()
    query_lower = query.lower().strip()

    # Search in dictionary
    matches = []
    for entry in dictionary:
        ghomala = entry.get("ghomala", "").lower()
        french = entry.get("french", "").lower()

        if query_lower in ghomala or query_lower in french:
            matches.append(entry)

    if matches:
        results = []
        for m in matches[:5]:  # Top 5 matches
            result = f"'{m.get('french', '')}' → Ghomala': {m.get('ghomala', '')}"
            if m.get("category"):
                result += f" ({m['category']})"
            if lookup_type == "example" and m.get("example"):
                result += f"\n  Exemple: {m['example']}"
            if m.get("cultural_note"):
                result += f"\n  Note: {m['cultural_note']}"
            results.append(result)

        return {
            "status": "success",
            "matches_found": len(matches),
            "results": "\n".join(results),
        }
    else:
        return {
            "status": "not_found",
            "message": f"Le mot '{query}' n'a pas été trouvé dans le dictionnaire local. "
                       "Je vais utiliser mes connaissances générales pour aider.",
        }


def pronunciation_helper(word: str, language: str = "ghomala") -> dict:
    """Help with pronunciation of a Ghomala' word.

    Use this tool when the user wants to know how to pronounce a word,
    or when you need to explain the tonal system of Ghomala'.

    Args:
        word: The word to get pronunciation guidance for.
        language: The language of the word — "ghomala", "french", or "english".

    Returns:
        dict: Pronunciation guidance with tonal information.
    """
    # Tone markers explanation
    tone_guide = {
        "à": "ton bas (grave)",
        "á": "ton haut (aigu)",
        "â": "ton descendant (circonflexe)",
        "ǎ": "ton montant (caron)",
    }

    special_chars = {
        "ɔ": "o ouvert (comme dans 'port')",
        "ɛ": "e ouvert (comme dans 'fête')",
        "ŋ": "n vélaire (comme 'ng' dans 'parking')",
        "ə": "schwa (e muet, comme dans 'le')",
        "ʉ": "u central (entre 'u' et 'ou')",
    }

    # Find tones in the word
    tones_found = []
    for char, desc in tone_guide.items():
        if char in word:
            tones_found.append(f"  {char} = {desc}")

    specials_found = []
    for char, desc in special_chars.items():
        if char in word:
            specials_found.append(f"  {char} = {desc}")

    result = f"Prononciation de '{word}':\n"
    if tones_found:
        result += "Tons:\n" + "\n".join(tones_found) + "\n"
    if specials_found:
        result += "Caractères spéciaux:\n" + "\n".join(specials_found) + "\n"
    if not tones_found and not specials_found:
        result += "Ce mot ne contient pas de caractères tonaux ou spéciaux marqués."

    return {"status": "success", "pronunciation_guide": result}


def cultural_context(topic: str) -> dict:
    """Provide cultural context about Bamiléké/Ghomala' traditions.

    Use this tool when the user asks about Bamiléké culture, traditions,
    ceremonies, family structure, or the cultural significance of words.

    Args:
        topic: The cultural topic to explain (e.g., "greeting", "family", "chief").

    Returns:
        dict: Cultural information related to the topic.
    """
    # Basic cultural knowledge base
    cultural_info = {
        "greeting": (
            "Dans la culture Bamiléké, les salutations sont essentielles. "
            "On salue toujours l'aîné en premier. 'Mbìə̀ Fɔ̀!' signifie 'Salut au Chef!'. "
            "Le respect des aînés se manifeste par l'inclinaison de la tête et le battement des mains."
        ),
        "family": (
            "La famille Bamiléké est étendue. 'Tá' = père, 'Má' = mère, 'Mbìŋ' = frère/sœur. "
            "Les cousins sont considérés comme des frères et tout le village participe à l'éducation."
        ),
        "chief": (
            "Le Fɔ̀ (chef) est le gardien des traditions et père de la communauté. "
            "Le système de chefferie Bamiléké est l'un des plus structurés d'Afrique."
        ),
        "marriage": (
            "Le mariage Bamiléké est une alliance entre deux familles. "
            "La dot (bride price) comprend des chèvres, huile de palme, et du vin de raphia."
        ),
        "food": (
            "La cuisine Bamiléké inclut le ndolé, le koki, le taro pilé (achu), "
            "et le nkui (sauce gluante). Le repas est un acte communautaire."
        ),
    }

    # Find matching topic
    topic_lower = topic.lower()
    for key, info in cultural_info.items():
        if key in topic_lower or topic_lower in key:
            return {"status": "success", "cultural_info": info}

    return {
        "status": "general",
        "cultural_info": (
            f"La culture Bamiléké est riche et complexe. "
            f"Je vais faire de mon mieux pour t'informer sur '{topic}' "
            f"à partir de mes connaissances générales."
        ),
    }


# ============================================================================
# ADK AGENT DEFINITION
# ============================================================================

root_agent = Agent(
    name="nam_sa_agent",
    model=GEMINI_LIVE_MODEL,
    description=(
        "NAM SA' (Le Soleil S'est Levé) — An AI agent dedicated to preserving "
        "and teaching the Ghomala' (Ghɔ́málá') language, a Bamiléké language "
        "spoken by ~1 million people in western Cameroon. Supports real-time "
        "voice conversations, translation, cultural teaching, and proverbs."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        ghomala_dictionary_lookup,
        pronunciation_helper,
        cultural_context,
    ],
)
