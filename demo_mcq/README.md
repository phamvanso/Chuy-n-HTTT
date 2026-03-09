# 📝 Vietnamese MCQ Generator – Demo App

Web app sinh câu hỏi trắc nghiệm tiếng Việt cho giáo viên, chạy bằng **Streamlit**.

```
Đoạn văn  →  ViQAG (ViT5)  →  Q+A  →  LLM  →  Distractors  →  MCQ  →  Export
```

---

## 🚀 Chạy nhanh (3 bước)

### Bước 1 – Cài thư viện

```bash
cd demo_mcq

# PyTorch CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Các thư viện còn lại
pip install -r requirements.txt
```

### Bước 2 – Tạo file `.env`

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
```

Điền vào `.env`:

```env
VIQAG_MODEL=VietAI/vit5-base-vi-qag
HF_TOKEN=hf_...                          # huggingface.co/settings/tokens

GEMINI_API_KEY=AIzaSyC8gGKtQ9PDQ6u1pD9BDg3JOVolzS0KQYc
# OPENAI_API_KEY=sk-...
# OLLAMA_HOST=http://localhost:11434
```

Chỉ cần **một** key LLM (Gemini / OpenAI / Ollama).

### Bước 3 – Khởi động Streamlit

```bash
streamlit run app.py
```

→ Mở **http://localhost:8501**

---

## 📂 Cấu trúc

```
demo_mcq/
├── app.py            # Streamlit UI – entry point
├── generator.py      # Stage 1: ViQAG (ViT5) sinh Q-A
├── distractor.py     # Stage 2: LLM sinh distractors
├── export_utils.py   # Xuất Word (.docx) và PDF
├── requirements.txt
├── .env              # API keys (tạo từ .env.example)
└── .env.example
```

---

## 🧩 Kiến trúc

### Stage 1 – `generator.py` (ViQAG)

| Chế độ | Yêu cầu | Tốc độ |
|---|---|---|
| **HF Inference API** | HF Token miễn phí | Nhanh, không cần GPU |
| **Local model** | ~1 GB RAM | Nhanh sau lần tải đầu |

Model: `VietAI/vit5-base-vi-qag`

### Stage 2 – `distractor.py` (LLM)

| Backend | Chất lượng | Setup |
|---|---|---|
| **Gemini** (gemini-1.5-flash) | ⭐⭐⭐⭐⭐ | `GEMINI_API_KEY` – miễn phí |
| **OpenAI** (gpt-4o-mini) | ⭐⭐⭐⭐⭐ | `OPENAI_API_KEY` – trả phí |
| **Ollama** (llama3/qwen2) | ⭐⭐⭐ | Local – miễn phí |

---

## ✏️ Tính năng chỉnh sửa

- Chỉnh nội dung câu hỏi và đáp án trực tiếp
- Đổi đáp án đúng bằng một click
- Xóa / Thêm câu
- Regenerate distractors cho từng câu riêng lẻ
- Xáo thứ tự A/B/C/D
- Chọn câu để xuất (checkbox)
- Lịch sử 10 đề gần nhất

## 📤 Export

| Định dạng | Nội dung |
|---|---|
| **Word (.docx)** | Đề thi + trang đáp án riêng |
| **PDF** | Đề thi + trang đáp án riêng |
| **TXT** | Kèm ghi chú đáp án đúng |
| **JSON** | Import Quizizz / Google Forms |

---

## 🐛 Lỗi thường gặp

| Lỗi | Fix |
|---|---|
| `CUDA out of memory` | Dùng HF Inference API |
| `JSONDecodeError` | `distractor.py` tự xử lý fallback |
| `429 Too Many Requests` | Tự retry 3 lần với sleep |
| Câu hỏi tiếng Anh | Đổi model → `VietAI/vit5-base-vi-qag` |
| ViQAG sinh 0 câu | Văn bản quá ngắn, kiểm tra HF Token |


Hệ thống sinh câu hỏi trắc nghiệm (MCQ) tiếng Việt tự động bằng **ViQAG (ViT5) + LLM**.

```
Đoạn văn tiếng Việt
        │
        ▼
  ViQAG (ViT5)  ──►  Question + Answer
        │
        ▼
     LLM  ──────────►  3 Distractors (đáp án sai)
        │
        ▼
   Build MCQ  ────────►  4 options A/B/C/D
        │
        ▼
  Streamlit UI
```

---

## 🚀 Setup nhanh (< 10 phút)

### 1. Cài thư viện

```bash
# CPU-only (nhẹ hơn, đủ để demo)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Tạo file `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền:

| Biến | Giá trị | Ghi chú |
|---|---|---|
| `HF_TOKEN` | `hf_...` | Lấy tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENAI_API_KEY` | `sk-...` | **hoặc** dùng Gemini/Ollama |
| `GEMINI_API_KEY` | `AI...` | Miễn phí tier |
| `OLLAMA_HOST` | `http://localhost:11434` | Nếu cài Ollama local |

Chỉ cần **một** key LLM là đủ chạy.

### 3. Chạy

```bash
streamlit run app.py
```

Mở trình duyệt: [http://localhost:8501](http://localhost:8501)

---

## 📂 Cấu trúc

```
demo_mcq/
├── app.py           # Streamlit UI (entry point)
├── generator.py     # ViQAG – sinh Q-A từ context (ViT5)
├── distractor.py    # LLM – sinh distractors (đáp án sai)
├── requirements.txt
├── .env.example     # Template biến môi trường
└── README.md
```

---

## 🧩 Kiến trúc chi tiết

### `generator.py` – QA Generation

| Chế độ | Khi nào dùng | Yêu cầu |
|---|---|---|
| **HF Inference API** | Demo nhanh, không có GPU | HF Token miễn phí |
| **Local model** | Dùng offline / nhiều request | ~1 GB RAM |

Model mặc định: `VietAI/vit5-base-vi-qag` (fine-tuned QAG tiếng Việt).

### `distractor.py` – Distractor Generation

| Backend | Chất lượng | Setup |
|---|---|---|
| **OpenAI** (gpt-4o-mini) | ⭐⭐⭐⭐⭐ | API key (trả phí) |
| **Gemini** (gemini-1.5-flash) | ⭐⭐⭐⭐ | API key miễn phí |
| **Ollama** (llama3/qwen2) | ⭐⭐⭐ | Cài local miễn phí |

---

## 💡 Ví dụ output

```
Context:
Hà Nội là thủ đô của Việt Nam. Đây là thành phố lớn thứ hai...

Câu 1. Thủ đô của Việt Nam là gì?
  A. Huế
  B. Hà Nội  ✔
  C. Đà Nẵng
  D. Hải Phòng

Câu 2. Hà Nội là thành phố lớn thứ mấy cả nước?
  A. Thứ nhất
  B. Thứ ba
  C. Thứ hai  ✔
  D. Thứ tư
```

---

## 🐛 Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `CUDA out of memory` | Model quá lớn | Dùng HF API, hoặc thêm `device_map="auto"` |
| `Token length exceeded` | Context quá dài | Rút ngắn context (<400 từ) |
| `JSONDecodeError` | LLM trả format lạ | Đã xử lý tự động trong `distractor.py` |
| `429 Too Many Requests` | Rate limit LLM | Tự động retry với sleep |
| `NoneType error` | ViQAG không sinh được | Kiểm tra HF Token / context |
| Câu hỏi ra tiếng Anh | Sai model | Đổi sang `VietAI/vit5-base-vi-qag` |

---

## 📦 Yêu cầu hệ thống

- Python 3.9+
- RAM: ≥ 4 GB (local model) hoặc chỉ cần internet (API mode)
- GPU: Không bắt buộc (CPU đủ để demo)

---

## 📚 Tham khảo

- [ViQAG Repository](https://github.com/asahi417/lm-question-generation)
- [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base)
- [Streamlit Docs](https://docs.streamlit.io)
