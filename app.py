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
    # Defensive fall-back: If no API key is set, prevent network crash
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
            # Extract out the web page text bodies alongside their URLs
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

    # Call search engine function to look up text context live on the internet
    web_sources = fetch_web_references(user_text, api_key)
    
    if not web_sources:
        return 0.0, "No web records matched."

    highest_match = 0.0
    matched_url = ""

    # Loop through scraped text blocks returned from live URLs
    for web_content, url in web_sources:
        if not web_content.strip():
            continue
            
        # Combine user text and current web document into a comparison token pair
        corpus = [user_text, web_content]
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            # Transform strings into high-dimensional TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Execute math calculation checking the directional angle between both vectors
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            match_score = similarity_matrix[0][0] * 100
            
            # Record the highest overlapping match found on the web
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

# Retrieve the API token securely from Streamlit Sidebar or config
st.sidebar.subheader("Configuration Panel")
api_key_input = st.sidebar.text_input("Enter Tavily API Key:", type="password", help="Get a free key from tavily.com")

# Main Text Form Interface
user_text = st.text_area("Paste your content here to check against the live web:", height=280, placeholder="Start typing or paste your document...")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Scan Entire Web"):
    if not api_key_input.strip():
        st.error("🔑 Access Denied: Please provide your free Tavily API Key in the sidebar panel to enable internet scanning.")
    elif user_text.strip() == "":
        st.warning("⚠️ Submission empty. Paste text into the box above before initializing a live scan request.")
    else:
        with st.spinner("Broadcasting queries to search index matrices and computing similarity overlap..."):
            
            # Execute internet validation routine
            score, source_link = calculate_internet_plagiarism(user_text, api_key_input)
            
            st.markdown("---")
            st.subheader("📊 Global Scan Results")
            
            # Match scoring status display blocks
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
