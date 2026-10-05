# Arabic dialect identifier (18 countries)

Guesses which of 18 Arab countries a tweet's author is from. Fine-tuned AraBERTv0.2-Twitter on the QADI dataset.

**Live demo:** 
**Model:** https://arabic-dialect-classifier-fqpappsg4auawwsvh4smk24.streamlit.app/

## Results (test set, evaluated once)
| Model | Macro F1 |
|---|---|
| TF-IDF + LinearSVC (untuned baseline) | 0.524 |
| AraBERTv0.2-Twitter, fine-tuned 4 epochs | **0.631** (accuracy 0.655) |

With my own grouping of the 18 countries into 5 regions, region accuracy is 0.846.
The QADI paper reports 0.606 on its own official test set (about 3,300 tweets). My test split
(8,981 tweets) is a different one, so this is context, not a like-for-like comparison.

## What I tried (validation macro F1)
| Experiment | Validation macro F1 |
|---|---|
| TF-IDF baseline | 0.544 |
| AraBERTv2 + my own text cleaning, 4 epochs | 0.585 |
| AraBERTv0.2-Twitter, epochs 1 / 2 / 3 / 4 | 0.594 / 0.638 / 0.645 / 0.643 |

Epoch 3 was the best on validation. I kept the final epoch-4 model, because a difference of 0.002 is small.

## What I found
- Region is right for 84.6% of tweets, country for 65.5%. The biggest mix-up is Jordan and
  Palestine (104 and 99 tweets). Six of the eight lowest-scoring countries are Gulf countries
  (counting Yemen as Gulf).
- Iraq: 59% of its 123 mistakes go to Gulf countries, and 64% of tweets wrongly predicted as
  Iraqi are Gulf tweets. Of 20 Iraqi tweets sent to the Gulf, 15 sounded Gulf to me and 5 had a
  clear Iraqi clue. Either the dialects overlap, or the labels are noisy: QADI labels come from
  account descriptions, and its authors estimate 91.5% label accuracy.
- Debugging: my first Farasa preprocessing shifted the cleaned text by two rows against the
  labels. I caught it by checking that mentions in the raw text match the placeholders in the
  cleaned text (0.997 on all splits after the fix) and switched to the Twitter model.

## Limitations
- Small, subjective error analysis; the 5-region grouping is my own choice.
- I did not check whether the same accounts appear in both train and test (QADI has about
  140 accounts per country). If they do, scores may look better than on new users.
- Scores are model outputs, not calibrated probabilities.
- No clear license found for the training data. Research and education use only.

## Run it locally
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py     # downloads the model from the Hugging Face Hub
```

## Files
- `01_data_exploration.ipynb`: data exploration and the TF-IDF baseline
- `arabertv2_first_attempt.ipynb`: first run with AraBERTv2 (Colab)
- `arabert_twitter_final.ipynb`: final AraBERT-Twitter run (Kaggle)
- `app.py`: Streamlit demo
- `upload_model.py`: uploads the model to the Hugging Face Hub
- `confusion_matrix.png`: test-set confusion matrix
