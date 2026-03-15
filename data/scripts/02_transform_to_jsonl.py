"""
============================================================================
Script 02: Transform raw datasets → Vertex AI JSONL for Gemini fine-tuning
============================================================================
Takes the downloaded translation datasets + your dictionary and converts
everything into Vertex AI supervised fine-tuning format.

Vertex AI Gemini SFT format for each JSONL line:
{
  "systemInstruction": {
    "role": "system",
    "parts": [{"text": "system prompt"}]
  },
  "contents": [
    {"role": "user", "parts": [{"text": "question"}]},
    {"role": "model", "parts": [{"text": "answer"}]}
  ]
}

Sources:
  - stfotso/french-ghomala-bandjoun  (~15,190 pairs)
  - stephanedonna/english_ghomala    (~7,916 pairs)
  - data/dictionary/ghomala_dictionary.json (curated entries)
  - Hand-crafted cultural conversations

Usage:
  python 02_transform_to_jsonl.py

Output:
  data/processed/train.jsonl    (90% of data — for training)
  data/processed/val.jsonl      (10% of data — for validation)
============================================================================
"""

import argparse
import json
import random
from pathlib import Path

# Paths
RAW_DIR = Path(__file__).parent.parent / "raw"
DICT_DIR = Path(__file__).parent.parent / "dictionary"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
random.seed(42)

# ============================================================================
# SYSTEM PROMPT — This defines the agent's personality
# ============================================================================
SYSTEM_PROMPT = (
    "Tu es NAM SA' (Le Soleil S'est Levé), un agent IA spécialisé dans "
    "la préservation et l'enseignement de la langue Ghomala' (Ghɔ̀málá'), "
    "une langue Bamiléké parlée dans la région Ouest du Cameroun. "
    "Tu te comportes comme un(e) ancien(ne) bienveillant(e) du village, "
    "patient(e) et encourageant(e). Tu parles Ghomala', Français et Anglais. "
    "Quand on te demande une traduction, tu donnes le mot en Ghomala' avec "
    "une explication culturelle quand c'est pertinent. Tu utilises les tons "
    "et caractères spéciaux du Ghomala' correctement (ɔ, ɛ, ŋ, etc.)."
)


def vertex_conversation(user_text: str, assistant_text: str) -> dict:
    """
    Create ONE training example in Vertex AI Gemini SFT format.

    This is the EXACT format that Vertex AI expects for supervised fine-tuning
    of Gemini models. Each call = one line in the JSONL file.

    Format reference:
    https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune_gemini/text_tune
    """
    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]},
            {"role": "model", "parts": [{"text": assistant_text}]}
        ]
    }


# ============================================================================
# TRANSFORM 1: French-Ghomala' (stfotso) → conversations
# ============================================================================
def transform_french_ghomala(raw_path: Path) -> list:
    """Convert stfotso/french-ghomala-bandjoun pairs into training conversations."""
    print("Transforming French-Ghomala' translations...")

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    for item in data:
        fr = item["french"].strip()
        bbj = item["ghomala"].strip()

        if not fr or not bbj:
            continue

        is_short = len(fr.split()) <= 5

        if is_short:
            conversations.append(vertex_conversation(
                f"Comment dit-on '{fr}' en Ghomala' ?",
                f"En Ghomala', '{fr}' se dit : {bbj}"
            ))
            conversations.append(vertex_conversation(
                f"Que veut dire '{bbj}' en Ghomala' ?",
                f"Le mot Ghomala' '{bbj}' signifie '{fr}' en français."
            ))
        else:
            conversations.append(vertex_conversation(
                f"Traduis en Ghomala' : {fr}",
                f"{bbj}"
            ))
            conversations.append(vertex_conversation(
                f"Que signifie cette phrase Ghomala' en français : {bbj}",
                f"Cette phrase Ghomala' signifie en français : {fr}"
            ))

    print(f"   ✅ Generated {len(conversations)} conversations from {len(data)} French-Ghomala' pairs")
    return conversations


# ============================================================================
# TRANSFORM 2: English-Ghomala' (stephanedonna) → conversations
# ============================================================================
def transform_english_ghomala(raw_path: Path) -> list:
    """Convert stephanedonna/english_ghomala pairs into training conversations."""
    print("Transforming English-Ghomala' translations...")

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []

    for item in data:
        en = item["english"].strip()
        bbj = item["ghomala"].strip()

        if not en or not bbj:
            continue

        is_short = len(en.split()) <= 5

        if is_short:
            conversations.append(vertex_conversation(
                f"How do you say '{en}' in Ghomala'?",
                f"In Ghomala', '{en}' is: {bbj}"
            ))
            conversations.append(vertex_conversation(
                f"What does '{bbj}' mean in English?",
                f"The Ghomala' word '{bbj}' means '{en}' in English."
            ))
        else:
            conversations.append(vertex_conversation(
                f"Translate to Ghomala': {en}",
                f"{bbj}"
            ))
            conversations.append(vertex_conversation(
                f"What does this Ghomala' text mean in English: {bbj}",
                f"This Ghomala' text means in English: {en}"
            ))

    print(f"   ✅ Generated {len(conversations)} conversations from {len(data)} English-Ghomala' pairs")
    return conversations


# ============================================================================
# TRANSFORM 3: Dictionary entries → rich vocabulary conversations
# ============================================================================
def transform_dictionary(dict_path: Path) -> list:
    """Convert the curated Ghomala' dictionary into training conversations."""
    print("Transforming dictionary entries...")

    if not dict_path.exists():
        print("   ⚠️  Dictionary file not found. Skipping.")
        return []

    with open(dict_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    conversations = []

    for entry in entries:
        ghomala = entry.get("ghomala", "").strip()
        french = entry.get("french", "").strip()
        category = entry.get("category", "")
        example = entry.get("example", "")
        cultural_note = entry.get("cultural_note", "")

        if not ghomala or not french:
            continue

        answer_parts = [f"En Ghomala', '{french}' se dit '{ghomala}'."]
        if category:
            answer_parts.append(f"C'est un mot de la catégorie '{category}'.")
        if example:
            answer_parts.append(f"Exemple d'utilisation : {example}")
        if cultural_note:
            answer_parts.append(f"Note culturelle : {cultural_note}")

        conversations.append(vertex_conversation(
            f"Comment dit-on '{french}' en Ghomala' ?",
            " ".join(answer_parts)
        ))

        conversations.append(vertex_conversation(
            f"Que veut dire '{ghomala}' en Ghomala' ?",
            f"Le mot Ghomala' '{ghomala}' signifie '{french}' en français."
        ))

        if example:
            conversations.append(vertex_conversation(
                f"Donne-moi une phrase avec le mot '{ghomala}' en Ghomala'.",
                f"Voici une phrase : {example}"
            ))

    print(f"   ✅ Generated {len(conversations)} conversations from {len(entries)} dictionary entries")
    return conversations


# ============================================================================
# TRANSFORM 4: Hand-crafted cultural conversations
# ============================================================================
def generate_cultural_conversations() -> list:
    """Hand-crafted conversations for cultural depth (highest quality examples)."""
    print("Adding hand-crafted cultural conversations...")

    conversations = [
        vertex_conversation(
            "Apprends-moi à me présenter en Ghomala'.",
            "Avec plaisir ! Pour te présenter en Ghomala', tu peux dire : "
            "'Mə̀ lá' [ton nom]' qui signifie 'Je suis [ton nom]'. "
            "Chez les Bamiléké, la présentation est un acte de respect. "
            "On commence toujours par saluer l'aîné avant de se présenter."
        ),
        vertex_conversation(
            "Comment salue-t-on un chef traditionnel en Ghomala' ?",
            "Saluer un Fɔ̀ (chef) en Ghomala' est un acte solennel. "
            "On dit 'Mbìə̀ Fɔ̀!' ce qui signifie 'Salut au Chef !'. "
            "Traditionnellement, on frappe des mains et on s'incline légèrement. "
            "Le Fɔ̀ est le gardien des traditions et le père de la communauté."
        ),
        vertex_conversation(
            "Comment dit-on les membres de la famille en Ghomala' ?",
            "La famille est au cœur de la culture Bamiléké ! Voici les termes principaux : "
            "'Tá' = père, 'Má' = mère, 'Mbìŋ' = frère/sœur. "
            "Chez les Bamiléké, la notion de famille est très étendue. "
            "Les cousins sont considérés comme des frères et sœurs, "
            "et tout le village participe à l'éducation des enfants."
        ),
        vertex_conversation(
            "Dis-moi un proverbe Ghomala' sur la sagesse.",
            "Voici un beau proverbe Bamiléké : la sagesse chez les Bamiléké se transmet "
            "de génération en génération à travers les proverbes. Chaque proverbe "
            "contient une leçon de vie profonde que les anciens partagent aux jeunes."
        ),
        vertex_conversation(
            "Comment compter de 1 à 5 en Ghomala' ?",
            "Compter en Ghomala' est une bonne base ! Voici les chiffres de 1 à 5 :\n"
            "1 = mɔ̀', 2 = pfʉ̂ə, 3 = tá', 4 = kwɔ̀', 5 = tâ'. "
            "La numération Bamiléké est fascinante car elle reflète un système "
            "basé sur le 5 et le 10."
        ),
        vertex_conversation(
            "Quelles sont les salutations courantes en Ghomala' ?",
            "Les salutations sont très importantes dans la culture Bamiléké ! "
            "Voici les principales :\n"
            "- 'À kə̀' = Bonjour (matin)\n"
            "- 'À sìə̀' = Bonsoir\n"
            "- 'Ǒ pə́ nə̀ búŋ' = Comment vas-tu ?\n"
            "On salue toujours les aînés en premier, c'est une marque de respect fondamentale."
        ),
        vertex_conversation(
            "What does 'Nam Sa' mean?",
            "NAM SA' means 'The Sun Has Risen' in Ghomala'. "
            "It symbolizes a new beginning, hope, and the enlightenment that comes with "
            "preserving our ancestral language. Just as the sun brings light to the world, "
            "our goal is to bring the Ghomala' language to new generations."
        ),
        vertex_conversation(
            "Explique-moi le système tonal du Ghomala'.",
            "Le Ghomala' est une langue tonale, ce qui signifie que la hauteur de la voix "
            "change le sens d'un mot ! Il y a 4 tons :\n"
            "- Ton bas (grave) : marqué 'à' → voix descend\n"
            "- Ton haut (aigu) : marqué 'á' → voix monte\n"
            "- Ton descendant : marqué 'â' → voix part haute et descend\n"
            "- Ton montant : marqué 'ǎ' → voix part basse et monte\n"
            "Par exemple, 'pá' (donner) et 'pà' (danse) sont des mots différents !"
        ),
    ]

    print(f"   ✅ Added {len(conversations)} cultural conversations")
    return conversations


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Transform Ghomala' data to Vertex AI JSONL")
    parser.add_argument("--no-limit", action="store_true", help="Don't limit dataset size")
    args = parser.parse_args()

    print("NAM SA' — Vertex AI JSONL Transform Pipeline")
    print("=" * 60)
    print("Target format: Vertex AI Gemini SFT (systemInstruction + contents)")
    print()

    all_conversations = []

    # 1. French-Ghomala'
    fr_path = RAW_DIR / "french_ghomala_bandjoun.json"
    if fr_path.exists():
        all_conversations.extend(transform_french_ghomala(fr_path))
    else:
        print(f"   ⚠️  {fr_path} not found. Run 01_download_datasets.py first.")

    # 2. English-Ghomala'
    en_path = RAW_DIR / "english_ghomala.json"
    if en_path.exists():
        all_conversations.extend(transform_english_ghomala(en_path))
    else:
        print(f"   ⚠️  {en_path} not found. Run 01_download_datasets.py first.")

    # 3. Dictionary
    dict_path = DICT_DIR / "ghomala_dictionary.json"
    all_conversations.extend(transform_dictionary(dict_path))

    # 4. Cultural conversations
    all_conversations.extend(generate_cultural_conversations())

    # Shuffle
    random.shuffle(all_conversations)

    # Vertex AI supports up to 10M text-only examples, but we cap for cost
    MAX_SAMPLES = len(all_conversations) if args.no_limit else 20000
    if len(all_conversations) > MAX_SAMPLES:
        all_conversations = all_conversations[:MAX_SAMPLES]
        print(f"\n   📏 Capped to {MAX_SAMPLES} samples")

    # Split: 90% train, 10% validation
    split_idx = int(len(all_conversations) * 0.9)
    train_data = all_conversations[:split_idx]
    val_data = all_conversations[split_idx:]

    # Write JSONL files
    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("TRANSFORM SUMMARY")
    print("=" * 60)
    print(f"   Format:      Vertex AI Gemini SFT (systemInstruction + contents)")
    print(f"   Train:       {len(train_data):>6} conversations → {train_path}")
    print(f"   Validation:  {len(val_data):>6} conversations → {val_path}")
    print(f"   Total:       {len(all_conversations):>6}")
    print(f"\n   Next step: python 03_upload_to_gcs.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
