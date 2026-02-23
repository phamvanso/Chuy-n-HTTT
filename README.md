# Ung dung Tao Cau Hoi Trac Nghiem Tu Dong

Ung dung web su dung NLP (Xu ly Ngon ngu Tu nhien) de tu dong tao cau hoi trac nghiem tu van ban tieng Viet.

## Tinh nang

- Tai len file PDF, TXT hoac DOCX
- Tu dong trich xuat van ban tu file
- Ho tro xu ly van ban tieng Viet
- Tao cau hoi trac nghiem voi 4 dap an
- Giao dien lam bai trac nghiem truc quan
- Hien thi ket qua va diem so chi tiet

## Cai dat

### 1. Tao moi truong ao

```bash
python -m venv venv
```

### 2. Kich hoat moi truong ao

Windows:
```bash
venv\Scripts\activate
```

Linux/MacOS:
```bash
source venv/bin/activate
```

### 3. Cai dat cac thu vien

```bash
pip install -r requirements.txt
```
### 4. 

```bash
pip install underthesea
```
## Chay ung dung

```bash
python app.py
```

Mo trinh duyet va truy cap: http://127.0.0.1:5000

## Huong dan su dung

1. Nhan nut "Bat dau" tren trang chu
2. Tai len file van ban (PDF, TXT hoac DOCX)
3. Chon so luong cau hoi muon tao
4. Xem truoc van ban da trich xuat
5. Nhan "Tao cau hoi trac nghiem"
6. Tra loi cac cau hoi va xem ket qua

## Luu y

- Ung dung ho tro van ban tieng Viet
- File tai len toi da 50MB
- Van ban can co du noi dung de tao cau hoi

## Cong nghe su dung

- Flask - Web framework
- underthesea - Xu ly ngon ngu tieng Viet
- pdfplumber - Doc file PDF
- python-docx - Doc file DOCX
