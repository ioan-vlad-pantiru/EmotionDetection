# Cum să obțineți NRC EmoLex în limba română

## Problema

Pipeline-ul a copiat din greșeală fișierul englezesc (`emolex_en.txt`) în locul celui românesc. Pentru a reconstrui corect RoEmoLex, aveți nevoie de **NRC EmoLex în limba română**.

## Opțiuni pentru obținerea lexiconului românesc

### Opțiunea 1: Descărcare manuală de la NRC

1. Accesați: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
2. Descărcați versiunea "OneFilePerLanguage"
3. Găsiți fișierul `Romanian.txt`
4. Plasați-l în:
   ```
   /Users/Andreea/Documents/NLP/emotion-detection/data/raw/nrc_emolex_ro.txt
   ```

### Opțiunea 2: Folosirea RoEmoLex original

Dacă aveți deja fișierul RoEmoLex original (format CSV/TSV):

1. Plasați fișierul în:
   ```
   /Users/Andreea/Documents/NLP/emotion-detection/data/raw/roemolex.csv
   ```
   sau
   ```
   /Users/Andreea/Documents/NLP/emotion-detection/data/raw/roemolex.tsv
   ```

2. Scriptul va detecta automat fișierul

### Opțiunea 3: Descărcare de la HuggingFace

Încercați să descărcați direct:

```python
from datasets import load_dataset

# Încercați diferite nume de dataset
try:
    dataset = load_dataset("nrc_emotion_lexicon")
    # Filtrați pentru română
    ro_data = [ex for ex in dataset if ex.get("language", "").lower() == "romanian"]
except:
    pass
```

## După ce ați obținut fișierul

1. **Ștergeți fișierul greșit**:
   ```bash
   rm roemolex-reconstruction/data/raw/nrc_emolex_ro.txt
   rm roemolex-reconstruction/work/checkpoints/step1_acquire.done
   ```

2. **Plasați fișierul corect** în locația corectă

3. **Rulați din nou Step 1**:
   ```bash
   cd roemolex-reconstruction
   python step1_acquire.py --no-resume
   ```

4. **Verificați că fișierul conține cuvinte românești**:
   ```bash
   head -20 data/raw/nrc_emolex_ro.txt
   ```
   
   Ar trebui să vedeți cuvinte românești cu diacritice (ă, â, î, ș, ț), nu cuvinte englezești!

5. **Rulați întregul pipeline**:
   ```bash
   python pipeline.py --no-resume
   ```

## Verificare

După ce pipeline-ul rulează, verificați output-ul:

```bash
head -20 data/out/roemolex_recon.jsonl
```

Ar trebui să vedeți cuvinte românești, de exemplu:
- "bucurie", "tristețe", "frică", "furie", etc.
- NU "abacus", "abandon", "ability" (cuvinte englezești)

## Format așteptat

Fișierul NRC EmoLex românesc ar trebui să aibă formatul:
```
cuvânt<TAB>emoție<TAB>asociere
```

Exemplu:
```
bucurie	joy	1
tristețe	sadness	1
frică	fear	1
```

## Note importante

- **RoEmoLex original** este un lexicon diferit de NRC EmoLex românesc
- RoEmoLex original conține ~11,000 cuvinte românești
- NRC EmoLex românesc este o traducere a lexiconului NRC pentru limba română
- Pipeline-ul reconstruiește un lexicon inspirat de RoEmoLex folosind NRC EmoLex românesc ca bază



