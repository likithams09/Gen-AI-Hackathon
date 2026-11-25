import pdfplumber


def extract_resume_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF file object or file path."""
    text_parts = []
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise ValueError(f"Failed to read PDF resume: {exc}") from exc
    finally:
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

    return "\n".join(text_parts).strip()
