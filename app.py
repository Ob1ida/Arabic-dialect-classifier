import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor

MODEL_ID = "Majellan/arabic-dialect-classifier"   # later: "YOUR_USERNAME/arabic-dialect-classifier"
BASE_NAME = "aubmindlab/bert-base-arabertv02-twitter"

COUNTRY = {'OM':'Oman','SD':'Sudan','SA':'Saudi Arabia','KW':'Kuwait','QA':'Qatar','LB':'Lebanon',
           'JO':'Jordan','SY':'Syria','IQ':'Iraq','MA':'Morocco','EG':'Egypt','PL':'Palestine',
           'YE':'Yemen','BH':'Bahrain','DZ':'Algeria','AE':'UAE','TN':'Tunisia','LY':'Libya'}

@st.cache_resource
def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    prep = ArabertPreprocessor(model_name=BASE_NAME)
    return tok, model, prep

tok, model, prep = load()

st.title("Arabic dialect identifier")
st.write("Paste an Arabic tweet. The model guesses which of 18 countries its author is from.")

text = st.text_area("Tweet", height=120)

if st.button("Predict") and text.strip():
    clean = prep.preprocess(text)
    if not clean.strip():
        st.warning("Nothing is left after cleaning (only mentions, links or symbols). Try a tweet with text.")
    else:
        enc = tok(clean, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        top = torch.topk(probs, 5)
        rows = [(COUNTRY[model.config.id2label[i.item()]], round(p.item(), 3))
                for p, i in zip(top.values, top.indices)]
        df = pd.DataFrame(rows, columns=["Country", "Score"])
        st.subheader(f"Best guess: {df.iloc[0]['Country']}")
        st.bar_chart(df.set_index("Country"))

with st.expander("About this model"):
    st.write(
        "AraBERTv0.2-Twitter, fine-tuned on about 440k tweets from the QADI dataset. "
        "Test macro F1 on 18 countries: 0.631 (untuned TF-IDF baseline: 0.524). "
        "It picks the right region for 84.6% of test tweets. QADI labels come from users' "
        "account descriptions, not from the tweets, so some labels are noisy, and neighbouring "
        "dialects (for example Jordan and Palestine, or the Gulf countries) are often confused. "
        "Scores are model outputs, not certainty."
    )