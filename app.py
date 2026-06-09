import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. REFERENCE DATABASE (CORPUS)
# ==========================================
# This acts as our local database repository. In a commercial application, 
# this would connect to a live database or web-crawler API.
DATABASE_CORPUS = [
    "Artificial intelligence and machine learning are transforming the modern tech landscape through automation.",
    "Natural Language Processing allows computers to understand, interpret, and manipulate human language structures.",
    "To compute cosine similarity, text documents must first be transformed into numerical vectors using TF-IDF matrices.",
    "Streamlit is an open-source Python library that makes it easy to create custom web apps for machine learning and data science.",
    "Python is a high-level, interpreted programming language known for its clear syntax and massive ecosystem of libraries.",
    "Data science combines statistical methods, data analysis algorithms, and technology to extract insights from structured data."
]

# ==========================================
# 2. CORE MACHINE LEARNING DETECTION ENGINE
# ==========================================
def calculate_plagiarism(user_text):
    """
    Compares the single user text input against the entire reference database.
    Returns the highest matching percentage found within the corpus database.
    """
    # Defensive check: if the input is empty or just spaces, return 0% immediately
    if not user_text.strip():
        return 0.0

    highest_match = 0.0
    
    # We iterate through each document entry stored inside our local database corpus
    for source_doc in DATABASE_CORPUS:
        # Create a temporary list holding the user's text and the current database document
        temp_corpus = [user_text, source_doc]
        
        # Initialize vectorizer to tokenize terms and strip out standard English filler words ('the', 'is', 'at')
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            # Transform text data arrays into mathematical TF-IDF feature matrices
            tfidf_matrix = vectorizer.fit_transform(temp_corpus)
            
            # Run a dot product calculation to evaluate the Cosine Similarity metric
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            # Extract the raw float match score from the multi-dimensional array
            match_score = similarity_matrix[0][0] * 100
            
            # Track and save only the highest matched value found across the database
            if match_score > highest_match:
                highest_match = match_score
        except ValueError:
            # Handles edge cases where user input contains only stop-words or unrecognizable symbols
            pass
            
    return round(highest_match, 2)


# ==========================================
# 3. INTERACTIVE RESPONSIVE DASHBOARD UI
# ==========================================
# Page configuration styling properties
st.set_page_config(page_title="AI Plagiarism Detector", page_icon="🕵️‍♂️", layout="centered")

# Injecting Custom CSS styling properties directly into the HTML wrapper markup
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stTextArea textarea { background-color: #1e293b !important; color: #f8fafc !important; border: 1px solid #38bdf8 !important; border-radius: 10px !important; }
    .stButton>button { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; width: 100%; padding: 12px 0px !important; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 5px 15px rgba(99, 102, 241, 0.4); }
    </style>
""", unsafe_allow_html=True)

# Application Heading Content
st.title("🕵️‍♂️ AI-Powered Plagiarism Detector")
st.caption("Paste your document below to analyze its mathematical and structural matching metrics against our reference database repository.")
st.markdown("---")

# Single Unified Input Text Area Component
user_input = st.text_area(
    "Enter or paste your text content below:", 
    height=250, 
    placeholder="Type or paste your paragraphs here to check for plagiarism levels..."
)

st.markdown("<br>", unsafe_allow_html=True)

# Trigger Action processing upon button interaction click event
if st.button("Scan Text For Plagiarism"):
    if user_input.strip() == "":
        st.warning("⚠️ Action required: Please paste some text content first before initiating an index scan.")
    else:
        with st.spinner("Tokenizing document string matrices and performing similarity matches..."):
            # Call the ML calculations engine
            plagiarism_percentage = calculate_plagiarism(user_input)
            
            st.markdown("---")
            st.subheader("📊 Plagiarism Risk Assessment Summary")
            
            # Render descriptive UI warnings conditionally based on mathematical risk metrics
            if plagiarism_percentage > 70:
                st.error(f"🔴 High Plagiarism Match Risk: **{plagiarism_percentage}%** Similarity Detected.")
                st.progress(int(plagiarism_percentage))
                st.markdown("> **Note:** The text shares structural alignments with materials found inside our source repository database.")
            elif 35 <= plagiarism_percentage <= 70:
                st.warning(f"🟡 Moderate Plagiarism Match Risk: **{plagiarism_percentage}%** Similarity Detected.")
                st.progress(int(plagiarism_percentage))
                st.markdown("> **Note:** Paraphrasing or matching contextual sentence fragments detected during scanning routines.")
            else:
                st.success(f"🟢 Low/No Plagiarism Match Risk: **{plagiarism_percentage}%** Similarity Detected.")
                st.progress(max(int(plagiarism_percentage), 1))
                st.markdown("> **Note:** Your document shows high uniqueness metrics when indexed against our corpus references.")
