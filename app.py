import streamlit as st
import os
import pandas as pd
from resume_utils import parse_resumes, compute_similarity, extract_text_from_pdf, clean_text
from fpdf import FPDF

# Set page config
st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stButton button {
        background: linear-gradient(to right, #6C63FF, #3E2FD5);
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 12px 20px;
        border: none;
        width: 100%;
        transition: all 0.3s ease-in-out;
    }
    .stTextInput textarea, .stTextInput input {
        background-color: #1a1f28;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .resume-preview {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        color: #dcdcdc;
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .metric-card {
        background: linear-gradient(145deg, #1f2128, #12141c);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin: 10px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #6C63FF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("AI Resume Screener")
    st.markdown("Upload resumes and match them against a job description.")
    st.markdown("---")
    st.markdown("Created by Syed Abdul Mateen")

# Main Content
st.title("AI Resume Screener")
st.markdown("Upload resumes and paste a job description to find the best matches.")

# Upload Resumes
uploaded_files = st.file_uploader(
    "Upload PDF or TXT Resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="resume_uploader"
)

# Job Description Input
job_description = st.text_area(
    "Paste the Job Description",
    height=300,
    placeholder="Paste or type the full job description..."
)

# Settings
col1, col2 = st.columns(2)
with col1:
    top_n = st.slider("Show Top N Resumes", min_value=1, max_value=20, value=5)
with col2:
    filter_role = st.selectbox(
        "Filter by Role",
        ["All"] + [
            "Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer",
            "QA Engineer", "UX Designer", "Data Analyst", "Cybersecurity Analyst",
            "Cloud Engineer", "Business Analyst", "Content Writer", "Digital Marketing Specialist"
        ]
    )

if st.button("Screen Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        temp_dir = "temp_resumes"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

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
            "marketing": "Digital Marketing Specialist"
        }

        resume_files = [f for f in os.listdir(temp_dir) if f.endswith(('.pdf', '.txt'))]
        resumes = []

        for filename in resume_files:
            file_path = os.path.join(temp_dir, filename)
            if filename.endswith('.pdf'):
                raw_text = extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
            cleaned_text = clean_text(raw_text)
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

        df_results = compute_similarity(job_description, resumes)

        result_list = []
        for i, row in df_results.iterrows():
            matched_resume = next((r for r in resumes if r['name'] == row['Resume']), None)
            result_list.append({
                'Resume': row['Resume'],
                'Role': matched_resume['role'] if matched_resume else 'Unknown',
                'Score': row['Score'],
                'Matched Keywords': row['Matched Keywords'],
                'Full Resume Text': matched_resume['raw_text'] if matched_resume else 'N/A'
            })
        df_results = pd.DataFrame(result_list)

        if filter_role != "All":
            df_results = df_results[df_results['Role'] == filter_role]
        df_results = df_results.head(top_n)

        st.subheader("Match Summary")
        if not df_results.empty:
            cols = st.columns(len(df_results))
            for i, row in df_results.iterrows():
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{row['Score']}%</div>
                        <div style="font-size:14px;">{row['Resume']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No resumes matched the criteria.")

        st.subheader("Top Matching Resumes")
        st.dataframe(df_results.style.background_gradient(cmap='Blues'))

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detailed Report as CSV",
            data=csv,
            file_name='resume_screening_report.csv',
            mime='text/csv',
        )

        def generate_pdf_report(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="AI Resume Screener - Detailed Report", ln=True, align='C')

            for _, row in df.iterrows():
                pdf.ln(10)
                pdf.set_font("Arial", 'B', size=10)
                pdf.cell(200, 6, txt=f"Resume: {row['Resume']} ({row['Role']})", ln=True)
                pdf.set_font("Arial", size=9)
                pdf.cell(200, 6, txt=f"Score: {row['Score']}%", ln=True)
                pdf.cell(200, 6, txt=f"Matched Keywords: {row['Matched Keywords']}", ln=True)
                pdf.ln(4)
                pdf.set_font("Arial", size=8)
                try:
                    pdf.multi_cell(0, 5, txt=row['Full Resume Text'])
                except:
                    pdf.multi_cell(0, 5, txt="Unable to display resume content.")

            return pdf.output(dest='S').encode('latin-1')

        if st.checkbox("Generate PDF Report"):
            try:
                pdf_bytes = generate_pdf_report(df_results)
                st.download_button(
                    label="Download Detailed Report as PDF",
                    data=pdf_bytes,
                    file_name="resume_screening_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error("Failed to generate PDF report.")
                st.exception(e)

        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)