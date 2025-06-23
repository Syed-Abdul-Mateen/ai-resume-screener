# AI Resume Screener

 **An AI-powered resume screener built with Python, Streamlit, spaCy, and Scikit-learn**

This app allows hiring teams or recruiters to:
- Upload multiple resumes (PDF/TXT)
- Paste a job description
- Automatically rank resumes by match score using NLP and Cosine Similarity
- Filter candidates by role (e.g., Developer, QA, UX, Data Analyst, etc.)
- Export results to CSV or PDF

##  Features

✅ **Resume Parsing**: Supports `.pdf` and `.txt` files  
✅ **Text Cleaning**: Uses `spaCy` for lemmatization and stopword removal  
✅ **Keyword Matching**: Highlights top matching keywords from job description  
✅ **Role Filtering**: Sort resumes by role (e.g., Full Stack Developer, Business Analyst)  
✅ **Export Options**: Download results as `CSV` or `PDF`  
✅ **Dark UI Theme**: Modern dark mode design with responsive layout  

##  How It Works

1. **Upload Resumes**: Drag & drop or browse resumes in PDF or TXT format.
2. **Paste Job Description**: Enter the job description or requirements.
3. **Screen Resumes**: The app parses each resume and compares it with the job description using **TF-IDF + Cosine Similarity**.
4. **View Results**: Get ranked results showing:
   - Match Score (% match)
   - Matched Keywords
   - Role-based filtering
5. **Export Report**: Download a detailed report in CSV or PDF format

##  Requirements

Make sure to install these dependencies:

```bash
pip install streamlit spacy scikit-learn pandas PyPDF2 matplotlib fpdf
python -m spacy download en_core_web_sm
