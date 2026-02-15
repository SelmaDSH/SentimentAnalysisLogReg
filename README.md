# Sentiment Analysis on Twitter Data

A step-by-step implementation of sentiment classification using logistic regression built from scratch.

**Final Accuracy: 99.65%**

---

## What This Project Does

Predicts whether a tweet expresses positive or negative sentiment.

**Example:**
- Input: `"I love this movie!"` → Output: **POSITIVE** 
- Input: `"This is terrible"` → Output: **NEGATIVE** 

---

## The Code - Step by Step

### 1️⃣ Import Libraries

Load all necessary Python packages:
- **NumPy** - for mathematical operations
- **Pandas** - for data handling
- **Matplotlib/Seaborn** - for visualizations
- **NLTK** - for natural language processing
- **scikit-learn** - for evaluation metrics only

### 2️⃣ Load the Tweet Dataset

- Downloaded **10,000 tweets** from NLTK Twitter Samples
  - 5,000 positive tweets
  - 5,000 negative tweets
- Split the data:
  - **Training set**: 8,000 tweets (first 4,000 of each)
  - **Test set**: 2,000 tweets (last 1,000 of each)
- Created labels: `1` = positive, `0` = negative

### 3️⃣ Preprocess the Tweets

Built a function to clean tweets:

**Before:** `"#FollowFriday @France_Inte for being top engaged :)"`

**Steps:**
1. Remove URLs (`https://...`)
2. Remove @mentions (`@France_Inte`)
3. Remove # symbols
4. Tokenize (split into words)
5. Remove stopwords ("for", "being")
6. Stem words ("engaged" → "engag")

**After:** `['followfriday', 'top', 'engag', ':)']`

### 4️⃣ Build the Frequency Dictionary

Created a dictionary that counts how many times each word appears in positive vs negative tweets.

**Example:**
```
('love', 1.0): 523  ← "love" appeared 523 times in positive tweets
('love', 0.0): 18   ← "love" appeared 18 times in negative tweets
('hate', 1.0): 12   ← "hate" appeared 12 times in positive tweets
('hate', 0.0): 456  ← "hate" appeared 456 times in negative tweets
```

**Result:** 11,390 unique (word, label) pairs

### 5️⃣ Extract Features

Converted each tweet into 3 numbers:

**For tweet:** `"I love this movie"`

**Output:** `[1, 800, 270]`

Where:
- `x[0] = 1` (bias term - always 1)
- `x[1] = 800` (sum of positive word frequencies)
- `x[2] = 270` (sum of negative word frequencies)

Since 800 > 270, this tweet leans positive!

### 6️⃣ Logistic Regression Functions

Built three core functions from scratch:

**a) sigmoid(z)** - Converts any number to a probability (0 to 1)
```python
sigmoid(0) = 0.5
sigmoid(10) = 0.999 (almost 1)
sigmoid(-10) = 0.001 (almost 0)
```

**b) compute_cost(X, Y, theta)** - Measures how wrong our predictions are
- Lower cost = better predictions
- Uses binary cross-entropy formula

**c) gradient_descent(X, Y, theta, alpha, num_iters)** - Trains the model
- Runs 1,500 iterations
- Each iteration adjusts weights to reduce errors
- Learning rate: 1e-9

### 7️⃣ Train the Model

- Started with weights = [0, 0, 0]
- Ran gradient descent for 1,500 iterations
- Cost decreased from 0.693 → 0.225
- Final learned weights:
  - θ₀ = 5.97e-08 (bias)
  - θ₁ = 0.000538 (positive weight)
  - θ₂ = -0.000559 (negative weight)

**Visualization:** Plotted training cost over iterations (shows learning progress)

### 8️⃣ Make Predictions

Built a function to predict sentiment of any tweet:

**Test Results:**
```
"I love this!" → 0.5273 (Positive)
"I hate this!" → 0.4949 (Negative)
"I am happy :)" → 0.8413 (Positive)
"I am sad :(" → 0.1085 (Negative)
```

### 9️⃣ Test Accuracy on Test Set

Evaluated model on 2,000 unseen tweets:
- **Accuracy: 99.65%**
- **Correct: 1,993 out of 2,000**
- **Wrong: Only 7 tweets**

**Confusion Matrix:**
```
           Predicted
         Neg    Pos
Actual Neg 1000    0
       Pos    7  993
```

**Classification Report:** Perfect 1.00 scores for precision, recall, and F1

### 🔟 Error Analysis

Analyzed the 7 misclassified tweets:

**Common error patterns:**
1. Sarcasm - "Not sure it would be good thing"
2. Emoticons with opposite context - ": )" in negative tweets
3. Ambiguous short phrases - "pats jay : ("

**Finding:** Model struggles with context-dependent sentiment and sarcasm

### 1️⃣1️⃣ Predict on New Tweets

Tested model on custom tweets:
```
"I really enjoyed the movie! It was awesome :)"
→ 😊 POSITIVE (83.37%)

"This is terrible, I hate it"
→ 😞 NEGATIVE (50.63%)

"Wow it's amazing!"
→ 😊 POSITIVE (50.52%)

"The worst day ever"
→ 😊 POSITIVE (50.71%) [Should be negative - shows model limitation]
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Training samples | 8,000 tweets |
| Test samples | 2,000 tweets |
| Test accuracy | 99.65% |
| Errors | 7 / 2,000 |
| Training iterations | 1,500 |
| Final cost | 0.225 |

---

## How to Run
```bash
# Install packages
pip install numpy pandas matplotlib seaborn nltk scikit-learn jupyter

# Download NLTK data
python -c "import nltk; nltk.download('twitter_samples'); nltk.download('stopwords')"

# Run the notebook
jupyter notebook sentiment_analysis.ipynb
```

Then run all cells in order (1 through 11).

---

## Technologies Used

- **Python 3.14** - Programming language
- **NumPy** - Matrix operations and math
- **NLTK** - Text preprocessing (tokenization, stemming, stopwords)
- **Matplotlib & Seaborn** - Data visualization
- **Pandas** - Data handling
- **scikit-learn** - Evaluation metrics only (not for modeling)

---

## What I Learned

1. **Text Preprocessing** - How to clean and prepare text data
2. **Feature Engineering** - Converting words to numerical features
3. **Logistic Regression** - How the algorithm works mathematically
4. **Gradient Descent** - How models learn from data
5. **Model Evaluation** - Accuracy, precision, recall, confusion matrix
6. **Error Analysis** - Understanding model limitations

---

## Why This Project

Implementation of a sentiment analysis model from scratch to understand the fundamentals of binary classification in NLP and the underlying mathematical and algorithmic principles.

---

## Limitations & Future Work

**Current limitations:**
- Struggles with sarcasm
- Doesn't handle negation well ("not good")
- Binary classification only (no neutral)

**Future improvements:**
- Add neutral sentiment category
- Handle negation patterns
- Try deep learning (LSTM, BERT)
- Build web interface for live predictions

---

## Project Structure
```
sentiment-analysis/
├── sentiment_analysis.ipynb    # Main code (cells 1-11)
├── README.md                   # This file
└── requirements.txt            # Python packages
```

---

## **requirements.txt**
```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
jupyter
