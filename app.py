import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. LIVE WEB CRAWL & SEARCH LAYER (Tavily Engine)
# ==========================================
def fetch_web_references(query_text, api_key):
    """
    Queries the live web via Tavily API using context keywords,
    extracting text body snippets from live websites.
    """
    if not api_key:
        return []
        
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query_text[:200], # Keep query length optimized for search engines
        "search_depth": "basic",
        "max_results": 4
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return [(item.get("content", ""), item.get("url", "")) for item in results]
    except Exception:
        pass
    return []

# ==========================================
# 2. CORE MACHINE LEARNING DISTANCE METRICS
# ==========================================
def calculate_internet_plagiarism(user_text, api_key):
    """
    Compares user text against real-time web results scraped via API.
    Returns the highest similarity score and the source URL.
    """
    if not user_text.strip():
        return 0.0, ""

    web_sources = fetch_web_references(user_text, api_key)
    
    if not web_sources:
        return 0.0, "No web records matched."

    highest_match = 0.0
    matched_url = ""

    for web_content, url in web_sources:
        if not web_content.strip():
            continue
            
        corpus = [user_text, web_content]
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_score = similarity_matrix[0][0] * 100
            
            if match_score > highest_match:
                highest_match = match_score
                matched_url = url
        except ValueError:
            pass

    return round(highest_match, 2), matched_url


# ==========================================
# 3. USER INTERFACE GRAPHICS & RENDERING
# ==========================================
st.set_page_config(page_title="Global AI Plagiarism Checker", page_icon="🌐", layout="centered")

# Custom UI CSS styles inject
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f1f5f9; }
    .stTextArea textarea { background-color: #111827 !important; color: #f1f5f9 !important; border: 1px solid #4f46e5 !important; border-radius: 8px !important; }
    .stButton>button { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; width: 100%; padding: 12px !important; }
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 0 15px rgba(124, 58, 237, 0.4); }
    </style>
""", unsafe_allow_html=True)

st.title("🌐 Live Internet Plagiarism Detector")
st.caption("Scan the entire internet in real-time. This app extracts underlying contextual signals and maps them against live web results via TF-IDF Vector Spaces.")
st.markdown("---")

# AUTOMATIC API KEY RETRIEVAL FROM STREAMLIT SECRETS SHIELD
# This avoids manual entry completely and hides your secret key safely from public GitHub view.
try:
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    TAVILY_API_KEY = None

# Main Text Form Interface
user_text = st.text_area("Paste your content here to check against the live web:", height=280, placeholder="Start typing or paste your document...")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Scan Entire Web"):
    if not TAVILY_API_KEY:
        st.error("🔑 Environment Key Missing: Please configure your TAVILY_API_KEY inside the Streamlit Cloud Settings panel.")
    elif user_text.strip() == "":
        st.warning("⚠️ Submission empty. Paste text into the box above before initializing a live scan request.")
    else:
        with st.spinner("Broadcasting queries to search index matrices and computing similarity overlap..."):
            
            # Execute internet validation routine using the automatically pulled key
            score, source_link = calculate_internet_plagiarism(user_text, TAVILY_API_KEY)
            
            st.markdown("---")
            st.subheader("📊 Global Scan Results")
            
            if score > 70:
                st.error(f"🔴 High Plagiarism Risk: **{score}%** Match across the web.")
                st.progress(int(score))
                st.markdown(f"🔗 **Primary Matching Source Found:** [{source_link}]({source_link})")
            elif 30 <= score <= 70:
                st.warning(f"🟡 Moderate Plagiarism Risk: **{score}%** Match detected.")
                st.progress(int(score))
                st.markdown(f"🔗 **Potential Source Overlap:** [{source_link}]({source_link})")
            else:
                st.success(f"🟢 Clean Document: Only **{score}%** Match found across the web.")
                st.progress(max(int(score), 1))
                st.write("Your document shows strong uniqueness scores when mapped against live index pools.")
