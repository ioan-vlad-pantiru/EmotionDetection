# Emotion Detection System - Intuitive Explanation

## 🎯 The Big Picture

**Goal**: Given a text (like a tweet or comment), predict which emotion it expresses from 8 emotions + neutral.

**Approach**: Combine two types of "clues" about emotions:
1. **Lexicon features**: Use dictionaries that tell us which words are associated with which emotions
2. **TF-IDF features**: Learn patterns from the actual text data itself

Then train a simple classifier (like a decision boundary) to make predictions.

---

## 📊 The Complete Pipeline

```
Text Input
    ↓
[Preprocessing] → Clean and normalize text
    ↓
[Feature Extraction] → Extract two types of features
    ├─→ Lexicon Features (21 numbers)
    └─→ TF-IDF Features (up to 50,000 numbers)
    ↓
[Feature Fusion] → Combine both feature types
    ↓
[Classifier] → Predict emotion
    ↓
Emotion Label (anger, joy, sadness, etc.)
```

---

## 🔍 Step-by-Step Breakdown

### 1. **Preprocessing** (`src/utils/text.py`)

**What happens**: Clean and normalize the raw text.

**Why**: Raw text is messy - we need to standardize it.

**Process**:
- Convert to lowercase: "I'M HAPPY!" → "i'm happy!"
- Remove URLs: "Check this https://example.com" → "Check this"
- Handle hashtags: "#happy" → "happy" (keep the word, remove #)
- Normalize elongated words: "soooo happy" → "soo happy"
- Convert emojis: 😊 → "happy face"
- Find negation: "not happy" → mark "happy" as negated

**Example**:
```
Input:  "I'm SOOO happy!!! 😊 #blessed"
Output: "i'm soo happy happy face blessed"
Negation positions: [] (none)
```

---

### 2. **Lexicon Features** (`src/features/lexicon_features.py`)

**What is a lexicon?**: A dictionary that maps words to emotions.

**Example lexicon entry**:
- Word: "happy" → Emotion: joy (score: 1.0)
- Word: "terrified" → Emotion: fear (score: 1.0)
- Word: "angry" → Emotion: anger (score: 1.0)

**How we use it**:
1. Count how many emotion words appear in the text
2. Sum up the emotion "scores" 
3. Add stylistic features (exclamation marks, uppercase ratio, etc.)

**Output**: 21 numbers representing:
- 8 emotion counts (how many words for each emotion)
- 8 emotion intensity sums (total emotion strength)
- 5 stylistic features (exclamations, questions, uppercase ratio, etc.)

**Example**:
```
Text: "I'm so happy and excited! This is amazing!"
Lexicon features:
  - joy_count: 2 (happy, excited)
  - joy_intensity: 2.0
  - exclamation_count: 1
  - ... (19 more features)
```

**Why it works**: Emotion words directly signal emotions. If someone uses "happy", "joyful", "excited", they're likely expressing joy.

**Limitation**: Can't capture context, sarcasm, or complex expressions.

---

### 3. **TF-IDF Features** (`src/features/tfidf_features.py`)

**What is TF-IDF?**: A way to convert text into numbers by finding important word patterns.

**TF-IDF = Term Frequency × Inverse Document Frequency**

- **Term Frequency (TF)**: How often a word appears in THIS text
- **Inverse Document Frequency (IDF)**: How rare/common the word is across ALL texts

**Why IDF matters**: Common words like "the", "a" appear everywhere → low IDF (not informative). Rare words like "devastated" → high IDF (very informative).

**What we extract**:
1. **Word n-grams**: Single words and pairs
   - "happy birthday" → ["happy", "birthday", "happy birthday"]
2. **Character n-grams**: Character patterns (3-5 characters)
   - "happy" → ["hap", "app", "ppy", "happ", "appy"]

**Why character n-grams?**: They catch:
- Typos: "hapy" still matches "hap"
- Word variations: "happy" and "happiness" share "happ"
- Morphological patterns

**Output**: Up to 50,000 numbers (sparse vector) representing:
- Which words/patterns appear in the text
- How important each pattern is

**Example**:
```
Text: "I'm so happy"
TF-IDF features:
  - "i'm": 0.3
  - "so": 0.1
  - "happy": 0.8
  - "i'm so": 0.2
  - "so happy": 0.6
  - ... (many zeros for words not in text)
```

**Why it works**: Learns patterns from data. If "I'm so happy" appears often with joy labels, the model learns this pattern.

**Advantage**: Can capture complex patterns, context, and phrases.

---

### 4. **Feature Fusion** (`src/features/fusion.py`)

**What happens**: Combine lexicon features (21 numbers) + TF-IDF features (50,000 numbers) = 50,021 features.

**Why combine?**: 
- Lexicon: Direct emotion signals (but limited)
- TF-IDF: Pattern learning (but needs data)
- Together: Best of both worlds!

**Process**:
1. Lexicon features: Scale them (normalize) so they're on similar scale as TF-IDF
2. TF-IDF features: Already normalized, keep as-is
3. Concatenate: Put them side-by-side

**Example**:
```
Lexicon:  [2.0, 0.5, 1.0, ...] (21 numbers)
TF-IDF:   [0.3, 0.0, 0.8, ...] (50,000 numbers)
Fused:    [2.0, 0.5, 1.0, ..., 0.3, 0.0, 0.8, ...] (50,021 numbers)
```

---

### 5. **Training the Classifier** (`src/models/train.py`)

**What is a classifier?**: A function that takes features and outputs an emotion label.

**We use**: Logistic Regression (a simple linear classifier)

**How it works**:
1. **Training**: Learn weights for each feature
   - If "happy" appears → increase weight for "joy"
   - If "angry" appears → increase weight for "anger"
   - Learn thousands of such patterns

2. **Decision**: For a new text, calculate:
   ```
   joy_score = weight1 × feature1 + weight2 × feature2 + ... + bias
   anger_score = weight1' × feature1 + weight2' × feature2 + ... + bias'
   ... (for all 9 emotions)
   ```
   Pick the emotion with highest score.

**Why Logistic Regression?**:
- Simple and interpretable
- Fast to train
- Works well with many features
- No GPU needed

**Training process**:
1. Split data: 70% train, 10% validation, 20% test
2. Train on training set
3. Check performance on validation set
4. Final evaluation on test set (unseen data)

---

### 6. **Three Model Variants (Ablation Study)**

We train three models to understand what works:

#### **A. Lexicon-Only Model**
- **Features**: Only lexicon features (21 numbers)
- **Purpose**: Baseline - can emotion dictionaries alone work?
- **Result**: ~30% accuracy (not great, but shows lexicons have some signal)

#### **B. ML-Only Model** (TF-IDF)
- **Features**: Only TF-IDF features (50,000 numbers)
- **Purpose**: Can pattern learning work without lexicons?
- **Result**: ~67% accuracy (much better! Patterns matter)

#### **C. Hybrid Model** (Fusion)
- **Features**: Both lexicon + TF-IDF (50,021 numbers)
- **Purpose**: Does combining both help?
- **Result**: ~68% accuracy (slightly better - fusion helps!)

**Insight**: TF-IDF is the main driver, but adding lexicon features gives a small boost.

---

## 📈 Evaluation (`src/utils/metrics.py`)

**Metrics we compute**:

1. **Accuracy**: Overall correctness
   - "Out of 100 texts, how many did we get right?"
   - Example: 68% accuracy = 68 correct out of 100

2. **Precision**: When we predict an emotion, how often are we right?
   - "We predicted 'joy' 100 times, 75 were correct" → 75% precision

3. **Recall**: Of all actual emotions, how many did we catch?
   - "There were 100 'joy' texts, we found 80" → 80% recall

4. **F1 Score**: Balance of precision and recall
   - Harmonic mean: F1 = 2 × (precision × recall) / (precision + recall)

5. **Confusion Matrix**: Shows what we confuse with what
   - Rows = true labels, Columns = predictions
   - Diagonal = correct predictions
   - Off-diagonal = mistakes

**Example Confusion Matrix**:
```
        Predicted:
        anger joy sadness
True anger  60   5    5
     joy     3   80    2
     sadness 2   3    40
```

---

## 🔬 Error Analysis (`scripts/evaluate_all.py`)

**What we analyze**:
- Top 50 misclassified examples
- Why did we get them wrong?

**Error categories**:
1. **Negation**: "not happy" → might predict joy (wrong!)
2. **Emojis**: 😊 → might not be captured well
3. **Short text**: "lol" → not enough context
4. **Sarcasm**: "yeah right" → says one thing, means another

**Purpose**: Understand model weaknesses to improve.

---

## 📊 Results Interpretation

### English Results (from your evaluation):

**Lexicon-Only**: 30.4% accuracy
- Very low - lexicons alone aren't enough
- Best at: neutral (42% F1), joy (38% F1)
- Worst at: fear (10% F1), disgust (14% F1)

**ML-Only (TF-IDF)**: 67.2% accuracy
- Much better! Learning patterns works
- Best at: trust (72% F1), joy (78% F1)
- Worst at: disgust (29% F1) - rare emotion

**Hybrid**: 68.0% accuracy
- Slightly better than ML-only
- Best at: fear (70% F1), joy (76% F1)
- Shows fusion helps, especially for rare emotions

**Key Insights**:
1. Pattern learning (TF-IDF) is crucial
2. Lexicons help but aren't sufficient alone
3. Combining both gives best results
4. Some emotions (disgust, fear) are harder to detect
5. Common emotions (joy, neutral) are easier

---

## 🌍 Bilingual Support

**English Pipeline**:
- Dataset: GoEmotions (Reddit comments)
- Lexicon: NRC EmoLex
- Labels: 9 emotions (8 Plutchik + neutral)

**Romanian Pipeline**:
- Dataset: REDv2 (Twitter tweets)
- Lexicon: RoEmoLex (when available)
- Labels: 7 emotions (from dataset)

**Why bilingual?**: 
- Different languages express emotions differently
- Need language-specific lexicons
- Can compare approaches across languages

---

## 🎓 Key Concepts Explained Simply

### **Why Feature Fusion?**
Think of it like combining:
- **Lexicon**: A dictionary expert who knows emotion words
- **TF-IDF**: A pattern detective who finds hidden clues
- **Together**: Both experts working together = better predictions

### **Why Three Models?**
Ablation study = "What if we remove X?"
- Remove TF-IDF → Lexicon-only (baseline)
- Remove Lexicon → ML-only (pattern learning)
- Keep both → Hybrid (best)

This tells us: "TF-IDF is essential, lexicon is helpful bonus"

### **Why Logistic Regression?**
Simple = better for:
- Understanding what matters
- Fast training
- No GPU needed
- Reproducible results

Deep learning would be overkill for this task.

---

## 🔄 Complete Workflow Example

**Input Text**: "I'm so excited about this! Can't wait!"

**Step 1 - Preprocessing**:
```
"i'm so excited about this can't wait"
```

**Step 2 - Lexicon Features**:
```
joy_count: 1 (excited)
joy_intensity: 1.0
anticipation_count: 1 (wait)
anticipation_intensity: 1.0
exclamation_count: 1
...
```

**Step 3 - TF-IDF Features**:
```
"excited": 0.7
"wait": 0.5
"excited about": 0.6
...
```

**Step 4 - Fusion**:
```
[lexicon features (21) + TF-IDF features (50,000)] = 50,021 features
```

**Step 5 - Classification**:
```
joy_score: 0.85
anticipation_score: 0.72
anger_score: 0.12
...
→ Predict: JOY (highest score)
```

**Output**: "joy" ✅

---

## 📁 File Structure Explained

```
src/
├── utils/
│   ├── text.py          → Preprocessing (clean text)
│   ├── metrics.py       → Evaluation metrics
│   └── io.py            → Save/load models
│
├── lexicons/
│   ├── emolex_en.py     → English emotion dictionary loader
│   └── roemolex.py      → Romanian emotion dictionary loader
│
├── datasets/
│   ├── goemotions.py    → Load English dataset
│   ├── red_ro.py        → Load Romanian dataset
│   └── mapping.py       → Map dataset labels to our 8 emotions
│
├── features/
│   ├── lexicon_features.py → Extract emotion word features
│   ├── tfidf_features.py   → Extract pattern features
│   └── fusion.py           → Combine both feature types
│
└── models/
    ├── train.py         → Train classifier
    └── infer.py         → Make predictions
```

---

## 🎯 Summary: Why This Approach Works

1. **Lexicon Features**: Direct emotion signals from word meanings
2. **TF-IDF Features**: Learn patterns from actual data
3. **Fusion**: Combine both for best results
4. **Simple Classifier**: Fast, interpretable, effective
5. **Evaluation**: Understand what works and what doesn't

**The Magic**: By combining explicit knowledge (lexicons) with learned patterns (TF-IDF), we get a system that:
- Understands emotion words (lexicon)
- Learns context and phrases (TF-IDF)
- Makes accurate predictions (68% accuracy)

This is a **practical, reproducible** approach that works well without needing deep learning or GPUs!

