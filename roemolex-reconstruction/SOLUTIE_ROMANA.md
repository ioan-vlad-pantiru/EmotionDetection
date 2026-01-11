# Soluție pentru obținerea lexiconului românesc

## Problema identificată

Fișierul `nrc_emolex_ro.txt` conține cuvinte **englezești** în loc de românești. Acest lucru se întâmplă pentru că scriptul a copiat din greșeală `emolex_en.txt`.

## Soluții

### Opțiunea 1: Descărcare automată (RECOMANDAT)

Am creat un script helper care încearcă să descarce automat:

```bash
cd roemolex-reconstruction
pip install requests  # dacă nu este instalat
python download_romanian_emolex.py
```

### Opțiunea 2: Descărcare manuală de la NRC

1. **Accesați**: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

2. **Descărcați**: Versiunea "OneFilePerLanguage" (ZIP)

3. **Extrageți** arhiva și găsiți fișierul `Romanian.txt`

4. **Copiați** fișierul la:
   ```bash
   cp Romanian.txt /Users/Andreea/Documents/NLP/emotion-detection/roemolex-reconstruction/data/raw/nrc_emolex_ro.txt
   ```

### Opțiunea 3: Folosirea RoEmoLex original

Dacă aveți deja fișierul **RoEmoLex original** (nu NRC EmoLex):

1. Plasați fișierul CSV/TSV la:
   ```
   data/raw/roemolex.csv
   ```
   sau
   ```
   data/raw/roemolex.tsv
   ```

2. Modificați `step1_acquire.py` să folosească acest fișier direct

### Opțiunea 4: Verificare în Downloads

Verificați dacă aveți deja fișierul în Downloads:

```bash
ls ~/Downloads/NRC-Emotion-Lexicon/OneFilePerLanguage/Romanian.txt
```

Dacă există, copiați-l:
```bash
cp ~/Downloads/NRC-Emotion-Lexicon/OneFilePerLanguage/Romanian.txt \
   roemolex-reconstruction/data/raw/nrc_emolex_ro.txt
```

## După ce ați obținut fișierul corect

1. **Verificați că este românesc**:
   ```bash
   head -20 roemolex-reconstruction/data/raw/nrc_emolex_ro.txt
   ```
   
   Ar trebui să vedeți cuvinte românești cu diacritice (ă, â, î, ș, ț), de exemplu:
   ```
   bucurie	joy	1
   tristețe	sadness	1
   frică	fear	1
   ```

2. **Rulați din nou Step 1**:
   ```bash
   cd roemolex-reconstruction
   python step1_acquire.py --no-resume
   ```

3. **Rulați întregul pipeline**:
   ```bash
   python pipeline.py --no-resume
   ```

## Verificare finală

După ce pipeline-ul rulează, verificați output-ul:

```bash
head -20 data/out/roemolex_recon.jsonl | python -m json.tool
```

Ar trebui să vedeți cuvinte românești în câmpul `"word"`, nu cuvinte englezești!

## Format așteptat

Fișierul NRC EmoLex românesc ar trebui să aibă formatul:
```
cuvânt<TAB>emoție<TAB>asociere
```

Unde:
- `cuvânt` = cuvânt românesc (ex: "bucurie", "tristețe")
- `emoție` = numele emoției în engleză (ex: "joy", "sadness", "fear")
- `asociere` = 0 sau 1 (dacă cuvântul este asociat cu emoția)

## Note importante

- **NRC EmoLex românesc** este o traducere a lexiconului NRC pentru limba română
- **RoEmoLex original** este un lexicon diferit, dezvoltat de Briciu & Lupea
- Pipeline-ul reconstruiește un lexicon inspirat de RoEmoLex folosind NRC EmoLex românesc ca bază
- Dacă nu găsiți NRC EmoLex românesc, puteți folosi direct RoEmoLex original dacă îl aveți



