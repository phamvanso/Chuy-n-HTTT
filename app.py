from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import spacy
import random
import pdfplumber
import os
import nltk
from nltk.corpus import wordnet

# Tải dữ liệu NLTK cần thiết
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.secret_key = 'super_secret_key_697'  # cho flash messages

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load spaCy model tiếng Anh
try:
    nlp = spacy.load('en_core_web_sm')
    print("✅ Loaded spaCy model: en_core_web_sm")
except OSError:
    print("❌ Model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def extract_text(file_path, extension):
    """Extract text from PDF, TXT, DOCX"""
    text = ""
    try:
        if extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        elif extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        elif extension in ['.doc', '.docx']:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

        return text.strip()

    except Exception as e:
        print(f"❌ Error extracting text from {file_path}: {e}")
        return ""


def get_synonyms(word):
    """Get English synonyms from WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and ' ' not in synonym:
                synonyms.add(synonym)
    return list(synonyms)


def generate_mcqs(text, num_questions=5):
    if not text or not nlp:
        print("❌ No text or model loaded.")
        return []

    doc = nlp(text)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if 15 <= len(sent.text.strip()) <= 250  # Nới rộng để lấy nhiều câu hơn
    ]

    print(f"🔍 Found {len(sentences)} valid sentences.")

    if not sentences:
        print("❌ No valid sentences after filtering.")
        return []

    mcqs = []
    random.shuffle(sentences)  # Trộn để đa dạng

    for sentence in sentences:
        if len(mcqs) >= num_questions:
            break

        sent_doc = nlp(sentence)
        nouns = [
            token.text for token in sent_doc
            if token.pos_ in ["NOUN", "PROPN"]
            and len(token.text) > 1
            and token.text.isalpha()
        ]

        if not nouns:
            continue

        subject = random.choice(nouns)
        question_stem = sentence.replace(subject, "________", 1)

        if len(question_stem) < 30:
            continue

        distractors = get_synonyms(subject)

        # Fallback distractors từ văn bản
        if len(distractors) < 3:
            candidates = [
                t.text for t in doc
                if t.pos_ in ["NOUN", "PROPN"]
                and t.text.lower() != subject.lower()
                and t.text not in distractors
            ]
            random.shuffle(candidates)
            distractors.extend(candidates[:3 - len(distractors)])

        distractors = list(set(distractors))[:3]
        while len(distractors) < 3:
            distractors.append("Other")

        choices = [subject] + distractors
        random.shuffle(choices)

        correct_letter = chr(65 + choices.index(subject))

        mcqs.append((question_stem, choices, correct_letter))
        print(f"✅ Generated MCQ: {question_stem[:60]}... (correct: {correct_letter})")

    print(f"📊 Total MCQs generated: {len(mcqs)} / requested: {num_questions}")
    return mcqs


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/how')
def howto():
    return render_template('howto.html')


@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        flash("No file part.", "error")
        return redirect(url_for('upload_page'))

    file = request.files['pdf_file']
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for('upload_page'))

    extension = os.path.splitext(file.filename)[1].lower()
    allowed = {'.pdf', '.txt', '.docx'}

    if extension not in allowed:
        flash("Only PDF, TXT, DOCX are supported.", "error")
        return redirect(url_for('upload_page'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        text = extract_text(file_path, extension)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up

    if not text:
        flash("Could not extract text (empty or invalid file).", "error")
        return redirect(url_for('upload_page'))

    try:
        num_questions = int(request.form.get('num_questions', 5))
    except ValueError:
        num_questions = 5

    mcqs = generate_mcqs(text, num_questions)

    if not mcqs:
        flash("No questions generated (text too short or no suitable content).", "error")
        return redirect(url_for('upload_page'))

    # Thêm index cho mỗi MCQ để khớp với JS trong questions.html
    mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]

    return render_template('questions.html', mcqs=mcqs_with_index, enumerate=enumerate, chr=chr)


if __name__ == '__main__':
    app.run(debug=True)