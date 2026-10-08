# 🚀 Sentiment Analysis with RNN

**Enough with Tabular Data: My First RNN for NLP**

A 5-class sentiment analysis model built from scratch with **PyTorch**, trained on tweets. The goal was not to chase the highest accuracy, but to understand what happens when raw text travels through an NLP pipeline and into a neural network.

---

## 📊 Dataset

A publicly available sentiment analysis dataset with two main columns:

| Column | Description |
|---|---|
| `OriginalTweet` | Tweet text (input) |
| `Sentiment` | Sentiment label (target) |

**Classes (5):** Positive · Negative · Neutral · Extremely Positive · Extremely Negative

🔗 Dataset: [PASTE DATASET LINK HERE]

---

## 🔄 Pipeline

1. **Cleaning:** removed URLs, mentions, special symbols and extra spaces; normalized the text
2. **Tokenization and lemmatization:** reduced vocabulary size while keeping meaning
3. **Vocabulary:** built from lemmatized tokens, plus `<pad>` and `<unk>`
4. **Token IDs:** tokens counted with `collections.Counter` and mapped to unique IDs
5. **Sequence padding:** sequences padded from the **left**, since the model uses the final hidden state for prediction

---

## 🧠 Model

```
Text → Token IDs → Embedding → RNN → Final Hidden State → Linear Layer → 5 Classes
```

| Setting | Value |
|---|---|
| Embedding dimension | 100 |
| Hidden size | 128 |
| RNN layers | 1 |
| Output classes | 5 |
| Dropout | 0.4 |

**Why left padding?** With left padding, the real tokens sit at the end of the sequence, so the final hidden state is computed right after reading the actual words rather than after a run of `<pad>` tokens.

---

## 📈 Result

**~65% test accuracy.** It is a baseline, not the finish line. Low accuracy and long sequences were the main challenges, and they showed how much harder NLP can be compared to tabular ML.

---

## 📚 What I Learned

- Tokenization and lemmatization
- Vocabulary creation and token-to-ID mapping
- Sequence padding
- Word embeddings
- RNN hidden states and sequence modeling
- Dropout and overfitting
- Building neural networks with PyTorch

---

## 🔭 What's Next

- Pre-trained **GloVe** embeddings
- **Early stopping**
- Different embedding and hidden dimensions
- Learning-rate and hyperparameter tuning
- Better handling of long sequences
- Other techniques to improve generalization

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Pandas` · `NLTK` · `Scikit-learn`

---

## 🔗 Links

- 👨🏻‍💻 LinkedIn: https://www.linkedin.com/in/kovihrushikesh/
- 📊 Dataset : https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification