import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """
    Use spaCy to clean and lemmatize text.
    Removes stopwords, punctuation, and performs lemmatization.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
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
    Parse all resumes in a given folder (supports .pdf and .txt).
    Returns list of dicts: {'name', 'raw_text', 'cleaned_text', 'role'}
    """
    resumes = []

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
            continue  # Skip unsupported file types

        cleaned_text = clean_text(raw_text)

        # Assign role based on filename
        role = "Other"
        name_lower = filename.lower()
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

    # Compute TF-IDF and Cosine Similarity
    tfidf = TfidfVectorizer().fit_transform([job_cleaned] + resume_texts)
    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Get top keywords from job description
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([job_cleaned] + resume_texts)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense().tolist()

    job_vector = dense[0]
    top_keywords = sorted(zip(feature_names, job_vector), key=lambda x: x[1], reverse=True)[:15]
    top_keywords_list = [kw[0] for kw in top_keywords if kw[1] > 0]

    results = []
    for i, score in enumerate(cosine_similarities):
        matched_keywords = [kw for kw in top_keywords_list if kw in resumes[i]['cleaned_text']]
        results.append({
            'Resume': resumes[i]['name'],
            'Score': round(score * 100, 2),
            'Matched Keywords': ", ".join(matched_keywords) if matched_keywords else "None",
            'Role': resumes[i]['role']
        })

    return pd.DataFrame(results).sort_values(by='Score', ascending=False)