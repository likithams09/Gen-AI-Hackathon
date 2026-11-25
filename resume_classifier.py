import os
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ResumeClassifier:
    """TF-IDF + Logistic Regression resume category classifier."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
        )
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False

    def extract_pdf_text(self, pdf_path: str) -> str:
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception:
            pass
        return " ".join(text_parts)

    def train_model(self, dataset_path: str):
        texts = []
        labels = []

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if not os.path.isdir(category_path):
                continue

            for file_name in os.listdir(category_path):
                if file_name.endswith(".pdf"):
                    pdf_path = os.path.join(category_path, file_name)
                    text = self.extract_pdf_text(pdf_path)
                    if len(text) > 100:
                        texts.append(text)
                        labels.append(category)

        if not texts:
            raise ValueError(f"No valid training PDFs found in '{dataset_path}'.")

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict(self, resume_text: str):
        if not self.is_trained:
            raise RuntimeError("ResumeClassifier must be trained before calling predict().")

        X = self.vectorizer.transform([resume_text])
        prediction = self.model.predict(X)[0]
        probability = max(self.model.predict_proba(X)[0])
        return prediction, round(probability * 100, 2)
