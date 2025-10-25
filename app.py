import streamlit as st
import pdfplumber
import docx
import io
import spacy
from spacy.matcher import Matcher
from skills import SKILL_KEYWORDS
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# --- Page Setup ---
st.set_page_config(
    page_title="SkillSync",
    page_icon="🤖"
)

# --- 1. Initialize Session State (Our App's "Memory") ---
# This runs only once at the very beginning.
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.resume_skills = set()
    st.session_state.jd_skills = set()
    st.session_state.matched_skills = set()
    st.session_state.missing_skills = set()
    st.session_state.bonus_skills = set()
    st.session_state.keyword_score = 0.0
    st.session_state.semantic_score = 0.0
    st.session_state.resume_text = ""
    st.session_state.jd_text = ""


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

# --- Load Semantic Model (SentenceTransformer) ---
@st.cache_resource
def load_semantic_model():
    """Loads the SentenceTransformer model."""
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

# --- Semantic Score Function ---
def get_semantic_score(text1, text2, model):
    """Calculates semantic similarity score between two texts."""
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        cosine_score = util.cos_sim(embedding1, embedding2)
        score = (cosine_score.item() * 100)
        return max(score, 0)
    except Exception as e:
        st.error(f"Error in semantic calculation: {e}")
        return 0.0
    
# --- AI Recommendation Function (using Google Gemini) ---

def get_ai_recommendations(missing_skills, api_key):
    """Generates learning and resume advice based on missing skills using Google Gemini."""
    try:
        # Configure the generative AI client
        genai.configure(api_key=api_key)

        # Use the latest valid Gemini model name
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        skills_str = ", ".join(missing_skills)

        prompt = f"""
        You are an expert career coach and technical recruiter.
        A candidate is applying for a job, but their resume is missing the following required skills: {skills_str}.

        Please provide a concise, actionable plan for them in two parts. Use markdown for formatting:

        1. **Learning Path:** Recommend 1-2 specific, high-quality online courses (e.g., Coursera, Udemy, freeCodeCamp) to learn these skills.
        2. **Resume Enhancement:** Write 3 example resume bullet points that this person could add to a 'Projects' section *after* they learn these skills.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"An error occurred while contacting the AI. Check your API key or model name. Error: {e}")
        return None


# --- Main App Interface ---
st.title("SkillSync - AI Resume Analyzer 🤖")
st.write("Welcome! Let's find the *true* match between a resume and a job description.")

uploaded_resume = st.file_uploader(
    "1. Upload Your Resume",
    type=['pdf', 'docx'],
    help="Please upload your resume in PDF or DOCX format."
)

jd_text = st.text_area(
    "2. Paste the Job Description",
    height=300,
    placeholder="Paste the entire job description here..."
)

api_key = st.text_input(
    "Enter Your Google AI API Key (100% Free)",
    type="password",
    help="Get your free key from Google AI Studio (aistudio.google.com)"
)

# --- 2. Analyze Button & Logic (The "Saver") ---
if st.button("Analyze 🚀"):

    if uploaded_resume is not None and jd_text:

        with st.spinner("Analyzing... This may take a moment."):

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

            # --- Analysis 1: Keyword Matching ---
            resume_skills = extract_skills(resume_text, nlp, matcher)
            jd_skills = extract_skills(jd_text, nlp, matcher)

            # --- Analysis 2: Semantic Matching ---
            semantic_score = get_semantic_score(resume_text, jd_text, semantic_model)

            # --- Save EVERYTHING to Session State ---
            st.session_state.resume_skills = resume_skills
            st.session_state.jd_skills = jd_skills
            st.session_state.matched_skills = resume_skills.intersection(jd_skills)
            st.session_state.missing_skills = jd_skills.difference(resume_skills)
            st.session_state.bonus_skills = resume_skills.difference(jd_skills) # Bug fix: Should be resume_skills.difference(jd_skills)

            st.session_state.keyword_score = 0.0
            if st.session_state.jd_skills:
                st.session_state.keyword_score = (len(st.session_state.matched_skills) / len(st.session_state.jd_skills)) * 100

            st.session_state.semantic_score = semantic_score
            st.session_state.resume_text = resume_text
            st.session_state.jd_text = jd_text

            # --- Set our "flag" to True ---
            st.session_state.analysis_complete = True

        st.success("Analysis Complete!")

    else:
        st.warning("Please upload a resume AND paste a job description.")


# --- 3. NEW: DISPLAY BLOCK (The "Reader") ---
# This block is now OUTSIDE the "Analyze" button.
# It checks our "memory" on every rerun.
if st.session_state.analysis_complete:

    st.markdown("---")
    st.subheader(f"🧠 Semantic Match Score: {st.session_state.semantic_score:.2f}%")
    st.progress(int(st.session_state.semantic_score))
    st.write("This score measures the *contextual meaning* and relevance of the entire resume against the job description.")

    st.markdown("---")
    st.subheader(f"🔑 Keyword Match Score: {st.session_state.keyword_score:.2f}%")
    st.progress(int(st.session_state.keyword_score))
    st.write("This score shows the percentage of *exact keywords* from the job description that were found in the resume.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Matched Skills")
        if st.session_state.matched_skills:
            st.write(f"_{', '.join(st.session_state.matched_skills)}_")
        else:
            st.info("No matching keywords found.")

    with col2:
        st.subheader("❌ Missing Skills")
        if st.session_state.jd_skills:
            if st.session_state.missing_skills:
                st.write(f"_{', '.join(st.session_state.missing_skills)}_")
            else:
                st.info("All required keywords are present!")
        else:
            st.warning("No keywords were found in the Job Description.")

    st.subheader("💡 Bonus Skills (in Resume)")
    # --- Bug Fix: Correctly calculate bonus skills ---
    # Need to read from session state where it was correctly saved
    if st.session_state.bonus_skills:
        st.write(f"_{', '.join(st.session_state.bonus_skills)}_")
    else:
        st.info("No bonus keywords were found.")

    # --- AI Recommendation Section ---
    st.markdown("---")
    st.subheader("🚀 Get Your AI-Powered Career Plan")

    if not st.session_state.jd_skills:
        st.info("No keywords were found in the job description, so no AI recommendations can be generated.")

    elif st.session_state.missing_skills:
        # This button is now inside the "memory" block, so it will work!
        if st.button("Get AI Recommendations 💡"):
            if not api_key:
                st.error("Please enter your Google AI API key above to get recommendations.")
            else:
                with st.spinner("🤖 Consulting the AI career coach..."):
                    recommendations = get_ai_recommendations(st.session_state.missing_skills, api_key)
                    # Check if recommendations were generated or if None was returned due to an error/block
                    if recommendations:
                        st.markdown(recommendations)
                    # If recommendations is None, the error message is already handled inside get_ai_recommendations

    else:
        st.success("Great news! Your resume matches all the keywords. No recommendations needed.")

    with st.expander("Show Raw Extracted Text"):
        st.subheader("Resume Text")
        st.text(st.session_state.resume_text)
        st.subheader("Job Description Text")
        st.text(st.session_state.jd_text)