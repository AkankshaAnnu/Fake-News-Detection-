#!/usr/bin/env python
# coding: utf-8

# In[14]:


#get_ipython().system('pip install streamlit')
#get_ipython().system('pip install scikit-learn')
#get_ipython().system('pip install pandas')
#get_ipython().system('pip install requests')
#get_ipython().system('pip install pyngrok')
## Upgrade newspaper3k and install the required extra
#get_ipython().system('pip install --upgrade newspaper3k[lxml_html_clean]')
#get_ipython().system('pip install --upgrade lxml_html_clean')


# In[15]:


import pickle
import pandas as pd
import requests
import re
from pathlib import Path
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")


# In[16]:


# -----------------------
# Function to clean text
# -----------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # Remove punctuation and digits
    text = re.sub(r'\s+', ' ', text)              # Normalize spaces
    return text.strip().lower()


# In[17]:


# -----------------------
# 1. Train or Load Model
# -----------------------
model_dir = Path("fake_news_model")
model_file = model_dir / "model.pkl"
vec_file = model_dir / "vectorizer.pkl"

# Replace these paths with your CSV files
fake_path = r"C:\Users\murar\OneDrive\Desktop\Fake.csv"
real_path = r"C:\Users\murar\OneDrive\Desktop\Real.csv"

if not model_file.exists() or not vec_file.exists():
    st.info("Training model for the first time... please wait.")
    
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    if "text" not in fake_df.columns or "text" not in real_df.columns:
        st.error("CSV files must contain a 'text' column.")
        st.stop()

    # Combine title + text
    fake_df["content"] = (fake_df.get("title","").astype(str) + " " + fake_df["text"].astype(str)).apply(clean_text)
    real_df["content"] = (real_df.get("title","").astype(str) + " " + real_df["text"].astype(str)).apply(clean_text)
    
    fake_df["label"] = 1
    real_df["label"] = 0

    df = pd.concat([fake_df[["content","label"]], real_df[["content","label"]]], ignore_index=True)
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"].values, df["label"].values,
        test_size=0.25, random_state=42, stratify=df["label"].values
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(max_iter=3000, C=2, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_tfidf))

    if acc < 0.80:
        st.warning(f"Accuracy {acc*100:.2f}% is below 80%, switching to SVM...")
        model = LinearSVC()
        model.fit(X_train_tfidf, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_tfidf))
        st.success(f"SVM trained successfully! Accuracy: {acc*100:.2f}%")
    else:
        st.success(f"Logistic Regression trained successfully! Accuracy: {acc*100:.2f}%")

    model_dir.mkdir(exist_ok=True)
    with open(vec_file,"wb") as f: pickle.dump(vectorizer, f)
    with open(model_file,"wb") as f: pickle.dump(model, f)

else:
    with open(vec_file,"rb") as f: vectorizer = pickle.load(f)
    with open(model_file,"rb") as f: model = pickle.load(f)


# In[18]:


# -----------------------
# 2. ML Prediction
# -----------------------
def predict_news(text):
    X_vec = vectorizer.transform([clean_text(text)])
    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(X_vec)[0]
        pred = model.predict(X_vec)[0]
        return {"prediction": "Fake" if pred==1 else "Real",
                "real_score": float(prob[0]),
                "fake_score": float(prob[1])}
    else:
        pred = model.predict(X_vec)[0]
        return {"prediction": "Fake" if pred==1 else "Real",
                "real_score": 1.0 if pred==0 else 0.0,
                "fake_score": 1.0 if pred==1 else 0.0}


# In[19]:


# -----------------------
# 3. Verify news with GNews API
# -----------------------
def verify_with_gnews(api_key, query):
    url = "https://gnews.io/api/v4/search"
    params = {"q": query, "token": api_key, "lang":"en", "max":3}
    try:
        response = requests.get(url, params=params).json()
        return bool(response.get("articles"))
    except:
        return False

SECOND_API_KEY = "c268d145437a5053a62247fd65677990"  # replace with your API key


# In[20]:


# -----------------------
# 4. Hybrid Prediction
# -----------------------
def hybrid_prediction(text):
    ml_result = predict_news(text)
    verified = verify_with_gnews(SECOND_API_KEY, text[:100])
    
    if verified and ml_result["prediction"]=="Real":
        confidence = max(ml_result["real_score"], 0.80)
    elif not verified and ml_result["prediction"]=="Real":
        confidence = min(ml_result["real_score"], 0.60)
    else:
        confidence = ml_result["fake_score"] if ml_result["prediction"]=="Fake" else ml_result["real_score"]
    
    return {"prediction": ml_result["prediction"],
            "real_score": confidence if ml_result["prediction"]=="Real" else 1-confidence,
            "fake_score": confidence if ml_result["prediction"]=="Fake" else 1-confidence,
            "verified": verified}


# In[21]:


# -----------------------
# 5. Fetch Latest News
# -----------------------
def fetch_latest_news(api_key, country="in", limit=5):
    url = "https://newsapi.org/v2/top-headlines"
    params = {"apiKey": api_key, "country": country, "pageSize": limit}
    response = requests.get(url, params=params).json()

    if response.get("status")!="ok" or not response.get("articles"):
        url = "https://newsapi.org/v2/everything"
        params = {"apiKey": api_key, "q":"news", "language":"en", "pageSize":limit, "sortBy":"publishedAt"}
        response = requests.get(url, params=params).json()

    if response.get("status")!="ok":
        st.error(f"News API error: {response.get('message','Unknown error')}")
        return []

    articles = []
    for a in response.get("articles", []):
        title = a.get("title","").strip()
        desc = a.get("description","").strip()
        img = a.get("urlToImage","")
        full_text = f"{title} {desc}".strip()
        if title or desc:
            articles.append({"headline": full_text, "image": img})
    return articles[:limit]


# In[22]:


# -----------------------
# 6. Extract text from URL
# -----------------------
def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return (article.title or "") + " " + (article.text or "")
    except Exception:
        return url


# In[23]:


# -----------------------
# 7. Streamlit UI
# -----------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding:20px'>
    <h1>📰 Fake News Detection</h1>
    <p style='font-size:18px;'>AI-powered detector (ML + GNews verification)</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", ["Check News", "Live Headlines"])

if menu=="Check News":
    st.subheader("🔍 Analyze Any News Content or URL")
    user_input = st.text_area("Paste news text or link here:", height=180, placeholder="Enter text or paste a link...")
    if st.button("Analyze News") and user_input.strip():
        with st.spinner("Analyzing with AI and verifying with GNews..."):
            text = extract_text_from_url(user_input) if user_input.startswith("http") else user_input
            result = hybrid_prediction(text)

        icon = "✅" if result['prediction']=="Real" else "❌"
        color = "#2ecc71" if result['prediction']=="Real" else "#e74c3c"
        verify_icon = "🔗" if result["verified"] else "⚠"
        verify_text = "Verified by trusted sources" if result["verified"] else "Not found in trusted sources"

        st.markdown(f"""
        <div style='padding:15px; border-radius:10px; background-color:{color}; color:white;'>
            <h3>{icon} Prediction: {result['prediction']} News {verify_icon}</h3>
            <p><b>Real Probability:</b> {result['real_score']*100:.1f}%<br>
               <b>Fake Probability:</b> {result['fake_score']*100:.1f}%</p>
            <p>{verify_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text or URL.")

elif menu=="Live Headlines":
    st.subheader("🗞 Live News Headlines (India)")
    API_KEY = "cf9d1a74cf564c5daf851dca5b256769"  # replace with your NewsAPI key
    with st.spinner("Fetching top headlines..."):
        articles = fetch_latest_news(API_KEY, country="in", limit=5)

    if not articles:
        st.warning("No headlines available at the moment.")
    else:
        for a in articles:
            result = hybrid_prediction(a["headline"])
            color = "green" if result['prediction']=="Real" else "red"
            icon = "✅" if result['prediction']=="Real" else "❌"
            verify_icon = "🔗" if result["verified"] else "⚠"
            verify_text = "Verified by trusted sources" if result["verified"] else "Not found in trusted sources"

            with st.expander(a["headline"][:120]+"..."):
                if a["image"]:
                    st.image(a["image"], width=400, caption="News Thumbnail")
                st.markdown(f"<h4 style='color:{color};'>{icon} {result['prediction']} News {verify_icon}</h4>", unsafe_allow_html=True)
                st.write(f"**Real Probability:** {result['real_score']*100:.1f}%")
                st.write(f"**Fake Probability:** {result['fake_score']*100:.1f}%")
                st.write(verify_text)


# In[ ]:




