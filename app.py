import streamlit as st
import pdfplumber
import docx
import io
import spacy
from spacy.matcher import Matcher
from skills import SKILL_KEYWORDS

# --- Page Setup ---
st.set_page_config(
    page_title="SkillSync MVP",
    page_icon="🤖"
)

# --- Load NLP Model & Create Matcher ---
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

def extract_skills(text, nlp, matcher):
    """Extract skills from text using the spaCy matcher."""
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.title())
        
    return found_skills

# --- Main App Interface ---
st.title("SkillSync - AI Resume Analyzer 🤖")
st.write("Welcome! Upload a resume and paste a job description to see the magic.")

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

        # --- Step 2: Extract Skills ---
        st.info("Analyzing text and extracting skills...")
        
        resume_skills = extract_skills(resume_text, nlp, matcher)
        jd_skills = extract_skills(jd_text, nlp, matcher)
        
        st.success("Analysis Complete!")

        # --- FINAL UPDATED SECTION: Calculate Score & Display Report ---
        
        # 1. Calculate Matches
        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills.difference(resume_skills)
        bonus_skills = resume_skills.difference(jd_skills)
        
        # 2. Calculate Score
        score = 0.0
        if jd_skills: # Avoid division by zero
            score = (len(matched_skills) / len(jd_skills)) * 100
        
        # 3. Display the Report
        st.markdown("---")
        st.subheader(f"📈 Compatibility Score: {score:.2f}%")
        
        # Use columns for a clean side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Matched Skills")
            if matched_skills:
                st.write(f"_{', '.join(matched_skills)}_")
            else:
                st.info("No matching skills found.")
        
        with col2:
            st.subheader("❌ Missing Skills")
            if missing_skills:
                st.write(f"_{', '.join(missing_skills)}_")
            else:
                st.info("All required skills are present!")
        
        st.subheader("💡 Bonus Skills (in Resume)")
        if bonus_skills:
            st.write(f"_{', '.join(bonus_skills)}_")
        else:
            st.info("No bonus skills (not required by JD) were found.")

        # Hide the raw text in expanders
        with st.expander("Show Raw Extracted Text"):
            st.subheader("Resume Text")
            st.text(resume_text)
            st.subheader("Job Description Text")
            st.text(jd_text)
        
    else:
        st.warning("Please upload a resume AND paste a job description.")