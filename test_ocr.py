from app.ocr import extract_text
from app.llm import summarize_financial_text

def run_test(file_path):
    print(f"\nProcessing file: {file_path}")

    text = extract_text(file_path)

    print("\n=== RAW TEXT ===\n")
    print(text[:500])  # preview

    summary = summarize_financial_text(text)

    print("\n=== AI SUMMARY ===\n")
    print(summary)


if __name__ == "__main__":
    run_test("sample_docs/sample.pdf")