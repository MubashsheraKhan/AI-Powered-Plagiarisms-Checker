import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CORE MACHINE LEARNING LOGIC
# ==========================================
def check_plagiarism(text1, text2):
    """
    Takes two text inputs, converts them into TF-IDF vectors,
    and returns their mutual Cosine Similarity score.
    """
    # Combine documents into a temporary corpus array
    corpus = [text1, text2]
    
    # Initialize the TfidfVectorizer to eliminate meaningless common filler words ('english')
    # and convert raw textual patterns into mathematical feature vectors.
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # fit_transform learns the vocabulary vocabulary dictionary and returns a term-document matrix
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute the pairwise cosine similarity metric between the two generated vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Extract the percentage value out of the similarity array matrix
    similarity_percentage = similarity_matrix[0][0] * 100
    return round(similarity_percentage, 2)


# ==========================================
# 2. STREAMLIT UI/UX DESIGN (Web3-Inspired Aesthetic)
# ==========================================
# Configure webpage meta settings
st.set_page_config(page_title="AI Plagiarism Checker", page_icon="🕵️‍♂️", layout="wide")

# Inject Custom CSS styles to create a clean, modern card layout with interactive elements
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stTextArea textarea { background-color: #1e293b !important; color: #f8fafc !important; border: 1px solid #38bdf8 !important; border-radius: 8px !important; }
    .stButton>button { background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; padding: 10px 24px !important; transition: transform 0.2s ease; }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)

# Main Title Header Banner
st.title("🕵️‍♂️ AI-Powered Plagiarism Checker")
st.caption("An advanced Machine Learning system utilizing TF-IDF and Cosine Similarity to detect similarity metrics between documents.")
st.markdown("---")

# Layout creation: Divide page into two distinct horizontal input panels
col1, col2 = st.columns(2)

with col1:
    st.subheader("Document A (Original / Source)")
    doc_a = st.text_area("Paste text or source material here:", height=300, placeholder="Type or paste text content...")

with col2:
    st.subheader("Document B (Suspected / Student Submission)")
    doc_b = st.text_area("Paste the text you want to screen here:", height=300, placeholder="Type or paste text content...")

st.markdown("<br>", unsafe_allow_html=True)

# Trigger Assessment Calculation Process upon Button Selection
if st.button("Analyze Documents"):
    # Defensive programming: Ensure both components possess text records before calculations
    if doc_a.strip() == "" or doc_b.strip() == "":
        st.warning("⚠️ Please provide text within both input panels to initiate processing.")
    else:
        with st.spinner("Processing text vectors and analyzing similarity structures..."):
            # Execute machine learning logic
            score = check_plagiarism(doc_a, doc_b)
            
            st.markdown("---")
            st.subheader("📊 Analysis Report Summary")
            
            # Highlight results based on structural similarity risk thresholds
            if score > 70:
                st.error(f"🔴 High Plagiarism Risk Detected: **{score}%** Match")
                st.progress(int(score))
            elif 30 <= score <= 70:
                st.warning(f"🟡 Moderate Plagiarism Risk Detected: **{score}%** Match")
                st.progress(int(score))
            else:
                st.success(f"🟢 Low/No Plagiarism Risk Detected: **{score}%** Match")
                st.progress(max(int(score), 1))
