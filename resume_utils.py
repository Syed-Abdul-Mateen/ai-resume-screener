import PyPDF2
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define a simple clean_text function without spaCy
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces and stopwords
    tokens = text.split()
    stopwords = {
        'the', 'and', 'is', 'of', 'to', 'in', 'a', 'on', 'for', 'with', 'as', 'by',
        'be', 'that', 'this', 'it', 'at', 'from', 'or', 'are', 'was', 'we', 'you',
        'they', 'he', 'she', 'his', 'her', 'their', 'our', 'my', 'an'
    }
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text


def parse_resumes(resume_folder):
    """
    Parse all resumes in folder. Supports .pdf and .txt files.
    Returns list of dicts: {'name', 'raw_text', 'cleaned_text', 'role'}
    """
    resumes = []

    # Role mapping based on resume filenames
    role_mapping = {
        "developer": "Developer",
        "full stack": "Full Stack Developer",
        "frontend": "Frontend Developer",
        "backend": "Backend Developer",
        "qa": "QA Engineer",
        "ux": "UX Designer",
        "data analyst": "Data Analyst",
        "cybersecurity": "Cybersecurity Analyst",
        "cloud": "Cloud Engineer",
        "business analyst": "Business Analyst",
        "content writer": "Content Writer",
        "marketing": "Digital Marketing Specialist",
        "technical writer": "Content Writer"
    }

    for filename in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, filename)

        if filename.endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            continue  # Skip unsupported files

        cleaned_text = clean_text(raw_text)

        # Detect role from filename
        name_lower = filename.lower()
        role = "Other"
        for key in role_mapping:
            if key in name_lower:
                role = role_mapping[key]
                break

        resumes.append({
            'name': filename,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'role': role
        })

    return resumes


def compute_similarity(job_desc, resumes):
    """
    Compute cosine similarity between job description and resumes.
    Also returns matched keywords from job description.
    """
    job_cleaned = clean_text(job_desc)
    resume_texts = [r['cleaned_text'] for r in resumes]

    tfidf = TfidfVectorizer().fit_transform([job_cleaned] + resume_texts)
    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_cleaned] + resume_texts)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense().tolist()

    job_vector = dense[0]
    top_keywords = sorted(zip(feature_names, job_vector), key=lambda x: x[1], reverse=True)[:15]
    top_keyword_set = set(kw[0] for kw in top_keywords if kw[1] > 0)

    results = []
    for i, score in enumerate(cosine_similarities):
        matched = [kw for kw in top_keyword_set if kw in resumes[i]['cleaned_text']]
        results.append({
            'Resume': resumes[i]['name'],
            'Score': round(score * 100, 2),
            'Matched Keywords': ", ".join(matched) if matched else "None"
        })

    return pd.DataFrame(results).sort_values(by='Score', ascending=False)