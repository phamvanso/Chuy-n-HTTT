from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import spacy
import random
import pdfplumber
import os
import nltk
import uuid
from nltk.corpus import wordnet

# Tải dữ liệu NLTK cần thiết
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.secret_key = 'super_secret_key_697'  # Khóa bí mật cho phiên làm việc

UPLOAD_FOLDER = 'uploads'
TEMP_TEXT_FOLDER = 'temp_texts'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Giới hạn file tối đa 50MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(TEMP_TEXT_FOLDER):
    os.makedirs(TEMP_TEXT_FOLDER)

# Tải mô hình xử lý ngôn ngữ spaCy
try:
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000
    print("✅ Đã tải mô hình xử lý ngôn ngữ thành công")
except OSError:
    print("❌ Lỗi: Không tìm thấy mô hình ngôn ngữ. Vui lòng chạy lệnh: python -m spacy download en_core_web_sm")
    nlp = None

# Giới hạn độ dài văn bản để tránh quá tải bộ nhớ
MAX_TEXT_LENGTH = 500000


def extract_text(file_path, extension):
    """Trích xuất văn bản từ file PDF, TXT, DOCX"""
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

        elif extension == '.docx':
            from docx import Document
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

        return text.strip()

    except Exception as e:
        print(f"❌ Lỗi khi trích xuất văn bản từ {file_path}: {e}")
        return ""


def get_synonyms(word):
    """Lấy từ đồng nghĩa tiếng Anh từ WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and ' ' not in synonym:
                synonyms.add(synonym)
    return list(synonyms)


def generate_mcqs(text, num_questions=5):
    if not text or not nlp:
        print("❌ Không có văn bản đầu vào hoặc mô hình chưa được tải.")
        return []

    doc = nlp(text)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if 15 <= len(sent.text.strip()) <= 300
    ]

    print(f"🔍 Tìm thấy {len(sentences)} câu hợp lệ.")

    if not sentences:
        print("❌ Không có câu hợp lệ sau khi lọc.")
        return []

    mcqs = []
    used_sentences = set()
    attempts = 0
    max_attempts = len(sentences) * 3

    while len(mcqs) < num_questions and attempts < max_attempts:
        attempts += 1
        
        sentence = random.choice(sentences)
        sent_doc = nlp(sentence)
        
        nouns = [
            token.text for token in sent_doc
            if token.pos_ in ["NOUN", "PROPN"]
            and len(token.text) > 2
            and token.text.isalpha()
        ]

        if not nouns:
            continue

        subject = random.choice(nouns)
        
        question_key = f"{sentence}_{subject}"
        if question_key in used_sentences:
            continue
        used_sentences.add(question_key)
        
        question_stem = sentence.replace(subject, "________", 1)

        if len(question_stem) < 25:
            continue

        distractors = get_synonyms(subject)

        if len(distractors) < 3:
            candidates = [
                t.text for t in doc
                if t.pos_ in ["NOUN", "PROPN"]
                and t.text.lower() != subject.lower()
                and t.text not in distractors
                and len(t.text) > 2
            ]
            random.shuffle(candidates)
            distractors.extend(candidates[:3 - len(distractors)])

        distractors = list(set(distractors))[:3]
        while len(distractors) < 3:
            distractors.append("Khác")

        choices = [subject] + distractors
        random.shuffle(choices)

        correct_letter = chr(65 + choices.index(subject))

        mcqs.append((question_stem, choices, correct_letter))
        print(f"✅ Đã tạo câu hỏi trắc nghiệm: {question_stem[:60]}... (đáp án đúng: {correct_letter})")

    print(f"📊 Tổng số câu hỏi đã tạo: {len(mcqs)} / yêu cầu: {num_questions}")
    
    if len(mcqs) < num_questions:
        print(f"⚠️ Chỉ tạo được {len(mcqs)} câu do văn bản không đủ nội dung phù hợp.")
    
    return mcqs


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Xử lý lỗi khi file tải lên quá lớn"""
    flash("File quá lớn! Vui lòng chọn file có dung lượng dưới 50MB.", "error")
    return redirect(url_for('upload_page'))


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
        flash("Không tìm thấy trường tệp trong yêu cầu.", "error")
        return redirect(url_for('upload_page'))

    file = request.files['pdf_file']
    if file.filename == '':
        flash("Bạn chưa chọn tệp.", "error")
        return redirect(url_for('upload_page'))

    extension = os.path.splitext(file.filename)[1].lower()
    allowed = {'.pdf', '.txt', '.docx'}

    if extension not in allowed:
        flash("Chỉ hỗ trợ các định dạng PDF, TXT, DOCX.", "error")
        return redirect(url_for('upload_page'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        text = extract_text(file_path, extension)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)  # Xóa file tạm

    if not text:
        flash("Không thể trích xuất văn bản (tệp rỗng hoặc không hợp lệ).", "error")
        return redirect(url_for('upload_page'))

    try:
        num_questions = int(request.form.get('num_questions', 5))
    except ValueError:
        num_questions = 5

    # Cắt văn bản nếu quá dài
    text_truncated = False
    original_length = len(text)
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        text_truncated = True

    # Lưu văn bản vào file tạm thay vì session (tránh cookie quá lớn)
    text_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_TEXT_FOLDER, f"{text_id}.txt")
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Chỉ lưu ID vào session
    session['text_id'] = text_id
    session['num_questions'] = num_questions
    session['filename'] = file.filename
    
    # Thống kê văn bản
    word_count = len(text.split())
    char_count = len(text)
    
    # Hiển thị preview (chỉ 10000 ký tự đầu)
    preview_text = text[:10000] + ("..." if len(text) > 10000 else "")
    
    return render_template('preview.html', 
                           text=preview_text,
                           full_char_count=original_length,
                           filename=file.filename,
                           word_count=word_count,
                           char_count=char_count,
                           num_questions=num_questions,
                           text_truncated=text_truncated)


@app.route('/generate', methods=['POST'])
def generate():
    """Xử lý tạo câu hỏi trắc nghiệm từ văn bản đã tải lên"""
    text_id = session.get('text_id')
    num_questions = session.get('num_questions', 5)
    
    if not text_id:
        flash("Không tìm thấy văn bản. Vui lòng tải lên lại.", "error")
        return redirect(url_for('upload_page'))
    
    # Đọc văn bản từ file tạm
    temp_file_path = os.path.join(TEMP_TEXT_FOLDER, f"{text_id}.txt")
    if not os.path.exists(temp_file_path):
        flash("Phiên làm việc đã hết hạn. Vui lòng tải lên lại.", "error")
        return redirect(url_for('upload_page'))
    
    with open(temp_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    mcqs = generate_mcqs(text, num_questions)
    
    # Dọn dẹp: xóa file tạm và dữ liệu phiên làm việc
    try:
        os.remove(temp_file_path)
    except Exception:
        pass
    session.pop('text_id', None)
    session.pop('num_questions', None)
    session.pop('filename', None)

    if not mcqs:
        flash("Không tạo được câu hỏi (văn bản quá ngắn hoặc không có nội dung phù hợp).", "error")
        return redirect(url_for('upload_page'))

    # Thông báo nếu số câu hỏi ít hơn yêu cầu
    actual_count = len(mcqs)
    warning_message = None
    if actual_count < num_questions:
        warning_message = f"Chỉ tạo được {actual_count}/{num_questions} câu hỏi do văn bản không đủ nội dung phù hợp."

    # Thêm index cho mỗi MCQ để khớp với JS trong questions.html
    mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]

    return render_template('questions.html', 
                           mcqs=mcqs_with_index, 
                           enumerate=enumerate, 
                           chr=chr,
                           warning_message=warning_message,
                           requested=num_questions)


if __name__ == '__main__':
    app.run(debug=True)