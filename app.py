import streamlit as st
import pdfplumber
import docx
import io
import spacy
from spacy.matcher import Matcher
from skills import SKILL_KEYWORDS
from sentence_transformers import SentenceTransformer, util  # --- NEW ---

# --- Page Setup ---
st.set_page_config(
    page_title="SkillSync",
    page_icon="🤖"
)

# --- Load NLP Model (spaCy) & Create Matcher ---
@st.cache_resource
def load_nlp_resources():
    """Loads the spaCy model and builds the skill matcher."""
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    
    patterns = []
    for skill in SKILL_KEYWORDS:
        pattern = [{"LOWER": token} for token in skill.split()]
        patterns.append(pattern)
        
    matcher.add("SKILL_MATCHER", patterns)
    return nlp, matcher

# Load the resources
nlp, matcher = load_nlp_resources()

# --- NEW: Load Semantic Model (SentenceTransformer) ---
@st.cache_resource
def load_semantic_model():
    """Loads the SentenceTransformer model."""
    # 'all-MiniLM-L6-v2' is a small, fast, and very good model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Load the semantic model
semantic_model = load_semantic_model()


# --- Helper Functions for Text Extraction ---

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    text = ""
    doc = docx.Document(file)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# --- Skill Extraction Function (spaCy) ---
def extract_skills(text, nlp, matcher):
    """Extract skills from text using the spaCy matcher."""
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.title())
        
    return found_skills

# --- NEW: Semantic Score Function ---
def get_semantic_score(text1, text2, model):
    """Calculates semantic similarity score between two texts."""
    try:
        # 1. Encode the texts into vectors (embeddings)
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        
        # 2. Calculate Cosine Similarity
        # This gives a score between -1 and 1 (usually 0 to 1 for text)
        cosine_score = util.cos_sim(embedding1, embedding2)
        
        # 3. Convert to a 0-100 percentage
        score = (cosine_score.item() * 100)
        # Ensure score is not negative (it shouldn't be, but as a safeguard)
        return max(score, 0)
    
    except Exception as e:
        st.error(f"Error in semantic calculation: {e}")
        return 0.0

# --- Main App Interface ---
st.title("SkillSync - AI Resume Analyzer 🤖")
st.write("Welcome! Let's find the *true* match between a resume and a job description.")

# --- 1. File Uploader ---
uploaded_resume = st.file_uploader(
    "1. Upload Your Resume",
    type=['pdf', 'docx'],
    help="Please upload your resume in PDF or DOCX format."
)

# --- 2. Text Area ---
jd_text = st.text_area(
    "2. Paste the Job Description",
    height=300,
    placeholder="Paste the entire job description here..."
)

# --- 3. Analyze Button & FINAL Logic ---
if st.button("Analyze 🚀"):
    
    if uploaded_resume is not None and jd_text:
        
        # --- Step 1: Extract Text ---
        resume_text = ""
        try:
            if uploaded_resume.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_resume)
            elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = extract_text_from_docx(uploaded_resume)
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()
        
        # --- Run both analyses ---
        st.info("Analyzing... This may take a moment.")

        # --- Analysis 1: Keyword Matching (Our old logic) ---
        resume_skills = extract_skills(resume_text, nlp, matcher)
        jd_skills = extract_skills(jd_text, nlp, matcher)
        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills.difference(resume_skills)
        bonus_skills = resume_skills.difference(jd_skills)
        
        keyword_score = 0.0
        if jd_skills:
            keyword_score = (len(matched_skills) / len(jd_skills)) * 100
        
        # --- Analysis 2: Semantic Matching (Our new logic) ---
        semantic_score = get_semantic_score(resume_text, jd_text, semantic_model)
        
        st.success("Analysis Complete!")

        # --- Display Report ---
        st.markdown("---")
        
        # --- NEW: Display Semantic Score ---
        st.subheader(f"🧠 Semantic Match Score: {semantic_score:.2f}%")
        st.progress(int(semantic_score))
        st.write("This score measures the *contextual meaning* and relevance of the entire resume against the job description, powered by a Deep Learning model.")
        
        st.markdown("---")

        # --- OLD: Display Keyword Score ---
        st.subheader(f"🔑 Keyword Match Score: {keyword_score:.2f}%")
        st.progress(int(keyword_score))
        st.write("This score shows the percentage of *exact keywords* from the job description that were found in the resume.")
        
        
        # Use columns for the keyword breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Matched Skills")
            if matched_skills:
                st.write(f"_{', '.join(matched_skills)}_")
            else:
                st.info("No matching keywords found.")
        
        with col2:
            st.subheader("❌ Missing Skills")
            if missing_skills:
                st.write(f"_{', '.join(missing_skills)}_")
            else:
                st.info("All required keywords are present!")
        
        st.subheader("💡 Bonus Skills (in Resume)")
        if bonus_skills:
            st.write(f"_{', '.join(bonus_skills)}_")
        else:
            st.info("No bonus keywords were found.")

        # Hide the raw text in expanders
        with st.expander("Show Raw Extracted Text"):
            st.subheader("Resume Text")
            st.text(resume_text)
            st.subheader("Job Description Text")
            st.text(jd_text)
        
    else:
        st.warning("Please upload a resume AND paste a job description.")