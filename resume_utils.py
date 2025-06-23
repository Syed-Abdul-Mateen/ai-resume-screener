import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def parse_resumes(resume_folder):
    resumes = []
    for filename in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, filename)
        if filename.endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            continue
        cleaned_text = clean_text(raw_text)
        resumes.append({
            'name': filename,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text
        })
    return resumes

def compute_similarity(job_desc, resumes):
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