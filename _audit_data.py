import json

# Check raw data integrity
with open('data/raw/french_ghomala_bandjoun.json', 'r', encoding='utf-8') as f:
    fr = json.load(f)
with open('data/raw/english_ghomala.json', 'r', encoding='utf-8') as f:
    en = json.load(f)
with open('data/dictionary/ghomala_dictionary.json', 'r', encoding='utf-8') as f:
    di = json.load(f)

print('RAW DATA:')
print(f'  french_ghomala: {len(fr)} pairs')
print(f'  english_ghomala: {len(en)} pairs')
print(f'  dictionary: {len(di)} entries')
print()

# Count expected samples with no limit
fr_valid = [(x['french'].strip(), x['ghomala'].strip()) for x in fr if x['french'].strip() and x['ghomala'].strip()]
en_valid = [(x['english'].strip(), x['ghomala'].strip()) for x in en if x['english'].strip() and x['ghomala'].strip()]

fr_short = sum(1 for f, g in fr_valid if len(f.split()) <= 5)
fr_long = sum(1 for f, g in fr_valid if len(f.split()) > 5)
en_short = sum(1 for e, g in en_valid if len(e.split()) <= 5)
en_long = sum(1 for e, g in en_valid if len(e.split()) > 5)

di_valid = [x for x in di if x.get('ghomala', '').strip() and x.get('french', '').strip()]
di_with_example = sum(1 for x in di_valid if x.get('example'))

fr_total = fr_short * 2 + fr_long * 2
en_total = en_short * 2 + en_long * 2
di_total = len(di_valid) * 2 + di_with_example
cultural = 8

total = fr_total + en_total + di_total + cultural

print('EXPECTED CONVERSATIONS (--no-limit):')
print(f'  French-Ghomala: {fr_short} short*2 + {fr_long} long*2 = {fr_total}')
print(f'  English-Ghomala: {en_short} short*2 + {en_long} long*2 = {en_total}')
print(f'  Dictionary: {len(di_valid)}*2 + {di_with_example} examples = {di_total}')
print(f'  Cultural: {cultural}')
print(f'  TOTAL: {total}')
print(f'  Train (90%): {int(total * 0.9)}')
print(f'  Val (10%): {total - int(total * 0.9)}')
print()

# Verify raw data: spot check that translations are NOT fabricated
print('SAMPLE VERIFICATION (French-Ghomala):')
for x in fr[:5]:
    f_text = x['french'][:80]
    g_text = x['ghomala'][:80]
    print(f'  FR: {f_text}')
    print(f'  GH: {g_text}')
    print()

print('SAMPLE VERIFICATION (English-Ghomala):')
for x in en[:5]:
    e_text = x['english'][:80]
    g_text = x['ghomala'][:80]
    print(f'  EN: {e_text}')
    print(f'  GH: {g_text}')
    print()

# Check for empty or suspicious entries
fr_empty_gh = sum(1 for x in fr if not x['ghomala'].strip())
fr_empty_fr = sum(1 for x in fr if not x['french'].strip())
en_empty_gh = sum(1 for x in en if not x['ghomala'].strip())
en_empty_en = sum(1 for x in en if not x['english'].strip())
fr_same = sum(1 for x in fr if x['french'].strip().lower() == x['ghomala'].strip().lower())
en_same = sum(1 for x in en if x['english'].strip().lower() == x['ghomala'].strip().lower())

print('DATA QUALITY:')
print(f'  FR empty ghomala: {fr_empty_gh}, empty french: {fr_empty_fr}, same text: {fr_same}')
print(f'  EN empty ghomala: {en_empty_gh}, empty english: {en_empty_en}, same text: {en_same}')
