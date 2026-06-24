from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def summarize_financial_text(text: str) -> str:
    prompt = f"""
You are a financial document analyst.

Extract and summarize:
- Invoice Number
- Total Amount
- Vendor Name
- Date
- Key Insights

Document:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def extract_financial_fields(text: str) -> str:
    prompt = f"""
You are a financial document parser.

Return ONLY valid JSON.

Extract ONLY these fields:

{{
    "invoice_number": "",
    "vendor_name": "",
    "invoice_date": "",
    "due_date": "",
    "total_amount": ""
}}

Rules:
- Return valid JSON only
- Include commas between fields
- No markdown
- No explanations
- No extra fields
- No nested objects
- No arrays

Document:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def analyze_invoice(text: str) -> str:
    prompt = f"""
You are a financial risk analyst.

Analyze this invoice.

Return ONLY valid JSON.

Format:

{{
    "risk_score": 0,
    "risk_level": "",
    "payment_status": "",
    "key_risks": [
        ""
    ],
    "recommendation": ""
}}

Scoring Rules:
- 0-30 = LOW
- 31-70 = MEDIUM
- 71-100 = HIGH

Determine:
1. Risk Score
2. Risk Level
3. Payment Status (PAID / UNPAID / UNKNOWN)
4. Key Risks
5. Recommendation

Document:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content