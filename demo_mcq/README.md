# 📝 Vietnamese MCQ Generator – Demo App (Local)

Web app sinh câu hỏi trắc nghiệm tiếng Việt cho giáo viên, chạy **100% local** bằng Streamlit.

```
Đoạn văn  →  ViQAG (ViT5)  →  Q+A  →  Ollama (LLM)  →  Distractors  →  MCQ  →  Export
```

✅ **Không cần API key** – chạy hoàn toàn trên máy local  
✅ **Không tốn tiền** – dùng model mã nguồn mở  
✅ **Bảo mật** – dữ liệu không gửi lên cloud

---

## 🚀 Chạy nhanh (4 bước)

### Bước 1 – Cài Ollama

Tải về tại [ollama.ai](https://ollama.ai) và chạy:

```bash
ollama pull llama3
```

### Bước 2 – Cài thư viện Python

```bash
cd demo_mcq

# PyTorch CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Các thư viện còn lại
pip install -r requirements.txt
```

### Bước 3 – Tạo file `.env`

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Mac/Linux
```

Điền vào `.env` (hoặc để mặc định):

```env
VIQAG_MODEL=shnl/vit5-vinewsqa-qg-ae
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

### Bước 4 – Khởi động Streamlit

```bash
streamlit run app.py
```

→ Mở **http://localhost:8501**

---

## 📂 Cấu trúc

```
demo_mcq/
├── app.py            # Streamlit UI – entry point (Ollama-only)
├── generator.py      # Stage 1: ViQAG (ViT5) sinh Q-A
├── distractor.py     # Stage 2: Ollama LLM sinh distractors
├── export_utils.py   # Xuất Word (.docx) và PDF
├── requirements.txt
├── .env              # Config (tạo từ .env.example)
└── .env.example
```

---

## 🧩 Kiến trúc

### Stage 1 – `generator.py` (ViQAG)

- **Model**: `shnl/vit5-vinewsqa-qg-ae` (fine-tuned QAG tiếng Việt)
- **Chế độ**: Local model mode – tự động tải về máy lần đầu
- **Yêu cầu**: ~1 GB RAM, không cần GPU

### Stage 2 – `distractor.py` (Ollama LLM)

- **Backend**: Ollama (local LLM)
- **Model**: llama3, qwen2, qwen2.5, gemma, mistral
- **Chất lượng**: ⭐⭐⭐⭐ (tốt với llama3/qwen2)
- **Yêu cầu**: Cài Ollama từ [ollama.ai](https://ollama.ai)

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
| `CUDA out of memory` | Dùng CPU mode (mặc định) |
| `JSONDecodeError` | `distractor.py` tự xử lý fallback |
| Ollama không kết nối được | Kiểm tra `ollama serve` đang chạy |
| Câu hỏi tiếng Anh | Đổi model → `shnl/vit5-vinewsqa-qg-ae` |
| ViQAG sinh 0 câu | Văn bản quá ngắn (<50 từ) |

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

## 📦 Yêu cầu hệ thống

- Python 3.9+
- RAM: ≥ 4 GB (local ViT5)
- Ollama: ≥ 8 GB RAM (cho llama3)
- CPU: Đủ để demo, không cần GPU

---

## 📚 Model được dùng

- **QA Generation**: [shnl/vit5-vinewsqa-qg-ae](https://huggingface.co/shnl/vit5-vinewsqa-qg-ae)
- **Distractor Generation**: Ollama LLM (llama3, qwen2, gemma, mistral)
- **Base Model**: [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base)

---

## 🔧 Tùy chỉnh

Sửa file `.env` để thay đổi:

- `VIQAG_MODEL`: Đổi sang model QAG khác
- `OLLAMA_HOST`: Ollama remote server
- `OLLAMA_MODEL`: Đổi model LLM (llama3, qwen2, gemma)

Sửa file `app.py` để thay đổi:

- `MAX_NEW_TOKENS`: Độ dài câu hỏi
- `NUM_QUESTIONS`: Số câu sinh mặc định
- `DIFFICULTY`: Độ khó mặc định
