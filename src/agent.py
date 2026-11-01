"""
agent.py
--------
Identifies PHI in a submitted document and writes a redacted copy.

The LLM's only job is to report the PHI it finds, as structured JSON matching
the PHIReport schema below. Python then does the find-and-replace and writes
the output file, so non-PHI text (clinical content) is never rewritten by the
model.

Usage:
    python3 src/agent.py <input_file> <output_file>
    python3 src/agent.py --lookup "<question>"   # test the RAG tool on one question
"""

import sys
import threading
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.config import Settings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

LLM_MODEL       = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_HOST        = "http://172.25.240.1:11434"  # Ollama host (Windows host IP from WSL)
DB_PATH         = "./chroma_db"
COLLECTION_NAME = "phi_regulations"
TOP_K_CHUNKS    = 5     # number of regulation chunks to retrieve per query
NUM_CTX         = 8192  # LLM context window in tokens (Ollama defaults to 4096)
OUTPUT_METHOD   = "json_schema"  # "json_schema" = Ollama constrains output to the schema
                                 # "json_mode"   = any valid JSON; shape comes from the prompt only


# ---------------------------------------------------------------------------
# Output schema — what the LLM must return
# ---------------------------------------------------------------------------

# Placeholder types, based on the 18 Safe Harbor identifiers in 45 CFR 164.514(b)(2).
# Each finding is replaced in the output as [TYPE], e.g. [NAME], [SSN].
PHIType = Literal[
    "NAME", "ADDRESS", "DATE", "AGE", "PHONE", "FAX", "EMAIL", "SSN",
    "MRN", "HEALTH_PLAN_ID", "ACCOUNT_NUMBER", "LICENSE_NUMBER",
    "VEHICLE_ID", "DEVICE_ID", "URL", "IP_ADDRESS", "BIOMETRIC", "PHOTO",
    "OTHER_ID",
]


class PHIFinding(BaseModel):
    value: str = Field(description="The exact PHI text, copied character-for-character from the document")
    type: PHIType = Field(description="The Safe Harbor identifier category")


class PHIReport(BaseModel):
    findings: list[PHIFinding] = Field(description="Every PHI value found in the document")


# ---------------------------------------------------------------------------
# System prompt — the LLM's standing instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a HIPAA compliance expert specializing in de-identification of Protected Health Information (PHI).

Your job is to find every piece of PHI in a medical document, based on the 18 Safe Harbor identifiers defined in 45 CFR § 164.514(b)(2).

Label each PHI value with exactly one of these types:
- NAME: names of people (patients, relatives, employers' contacts, physicians)
- ADDRESS: street address, city, county, or zip code (anything smaller than a state)
- DATE: any date related to the individual (birth, admission, discharge, appointment, created); report the date or date-time as written
- AGE: ages over 89
- PHONE: telephone numbers
- FAX: fax numbers
- EMAIL: email addresses
- SSN: Social Security numbers
- MRN: medical record numbers
- HEALTH_PLAN_ID: health plan member, beneficiary, or group numbers
- ACCOUNT_NUMBER: account numbers
- LICENSE_NUMBER: certificate or license numbers (including provider NPI)
- VEHICLE_ID: vehicle identifiers and license plates
- DEVICE_ID: device identifiers and serial numbers
- URL: web addresses
- IP_ADDRESS: IP addresses
- BIOMETRIC: fingerprints, voice prints, other biometrics
- PHOTO: full-face photos or comparable images
- OTHER_ID: any other unique identifying number or code (employee IDs, encounter IDs, study IDs, document IDs, confirmation numbers)

Rules:
- Copy each PHI value exactly as it appears in the document, character for character. Report only the value, never the field label.
- Report a value once even if it appears multiple times.
- Do NOT report clinical content: diagnoses, symptoms, allergies, medications, dosages, lab values, procedures, or care plans. These are not PHI.
- Do NOT report gender or a state name. These are not PHI.

ADDRESS forms to report (each part separately, exactly as written):
- Street addresses: "1427 Willow Creek Drive", "500 Medical Parkway", "3100 Cedar Avenue", "22 N. Main St.", "4800 Hwy 71 W"
- Unit or suite numbers: "Apt 4B", "Suite 210", "Unit #12", "Bldg 3"
- PO boxes and rural routes: "PO Box 1182", "P.O. Box 77", "RR 2 Box 15"
- Cities and towns: "Austin", "Round Rock", "St. Louis"
- Counties: "Travis County", "Cook Co."
- Zip codes, 5-digit or ZIP+4: "78704", "78705", "60614-2231"
- Addresses inside sentences count too: "lives on Elm Street in Pflugerville"
- Addresses of any party in the document (patient, relatives, employer, clinic, facility) are reported the same way.
- Not PHI: state names or abbreviations ("Texas", "TX"), and country ("USA").

DATE forms to report (exactly as written, including any time attached to it):
- Numeric: "03/14/1978", "3/14/78", "03-14-1978", "1978-03-14", "14.03.1978"
- Written out: "March 14, 1978", "Mar 14 1978", "14 March 1978", "March 14th"
- Month and year only: "March 1978", "03/1978"
- Date with time: "10/28/2026 08:15 AM", "11/06/2026 at 09:30 AM", "2026-10-30T14:42"
- Dates inside sentences count too: "presented on 10/28/2026", "seen again on Nov 6"
- If the same date appears both with and without a time, report each form separately, e.g. "10/28/2026 08:15 AM" and "10/28/2026".
- Not PHI: a year by itself ("1978", "in 2026"), durations ("for two days", "within 7 days"), and dosing schedules ("twice daily", "nightly").

Respond with JSON only, in this shape:
{"findings": [{"value": "<exact text>", "type": "<TYPE>"}, ...]}

Example input:
Patient: Jane Doe
DOB: 02/11/1965
Gender: Female
Address: 88 Oak Street, Springfield, IL 62704
Phone: (312) 555-0199
Diagnosis: Asthma
Medication: Albuterol 90 mcg inhaler
Account #: AC-55102

Example output:
{"findings": [
  {"value": "Jane Doe", "type": "NAME"},
  {"value": "02/11/1965", "type": "DATE"},
  {"value": "88 Oak Street", "type": "ADDRESS"},
  {"value": "Springfield", "type": "ADDRESS"},
  {"value": "62704", "type": "ADDRESS"},
  {"value": "(312) 555-0199", "type": "PHONE"},
  {"value": "AC-55102", "type": "ACCOUNT_NUMBER"}
]}

Note what the example does NOT report: "Female" (gender), "IL" (state), "Asthma" (diagnosis), and "Albuterol 90 mcg inhaler" (medication).
"""


# ---------------------------------------------------------------------------
# RAG retriever — queries Chroma for relevant regulation chunks
# ---------------------------------------------------------------------------

# Chroma's PersistentClient is not safe to construct from multiple threads at
# once. Build the retriever once, guarded by a lock, and reuse it.
_retriever = None
_retriever_lock = threading.Lock()


def get_retriever():
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            embed_fn = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=LLM_HOST)
            client = chromadb.PersistentClient(
                path=DB_PATH,
                settings=Settings(anonymized_telemetry=False),
            )
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embed_fn,
                client=client,
            )
            _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})
        return _retriever


@tool
def query_phi_regulation(question: str) -> str:
    """
    Query the HIPAA 45 CFR 164.514 knowledge base for regulation text
    relevant to a question about what counts as PHI.

    Args:
        question: A natural language question, e.g. 'Are telephone numbers PHI?'
    """
    docs = get_retriever().invoke(question)
    if not docs:
        return "No relevant regulation text found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# RAG lookup — agent answers one question using the query tool
# ---------------------------------------------------------------------------

LOOKUP_PROMPT = """You answer questions about HIPAA de-identification rules.
Always call the query_phi_regulation tool first, then answer using only the regulation text it returns.
Cite the paragraph (e.g. 164.514(b)(2)(i)(D)) when you can."""


def lookup(question: str) -> None:
    """Run the agent on a single question so the tool -> Chroma -> answer path can be checked."""
    llm = ChatOllama(model=LLM_MODEL, base_url=LLM_HOST, temperature=0, num_ctx=NUM_CTX)
    agent = create_agent(model=llm, tools=[query_phi_regulation], system_prompt=LOOKUP_PROMPT)

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[tool call] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f"[tool result] {len(message.content)} chars retrieved:")
            print("  " + message.content[:400].replace("\n", "\n  ") + "...\n")

    print("[answer]")
    print(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# PHI detection and redaction
# ---------------------------------------------------------------------------

def find_phi(document_text: str) -> PHIReport:
    """Ask the LLM for the PHI in the document, constrained to the PHIReport schema."""
    llm = ChatOllama(model=LLM_MODEL, base_url=LLM_HOST, temperature=0, num_ctx=NUM_CTX)

    # Either way, LangChain validates the reply against PHIReport.
    # The prompt describes the JSON shape, so json_mode works without the schema.
    structured_llm = llm.with_structured_output(PHIReport, method=OUTPUT_METHOD)

    return structured_llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Find all PHI in this document:\n\n{document_text}"),
    ])


def redact(document_text: str, report: PHIReport) -> tuple[str, list[PHIFinding]]:
    """
    Replace each reported PHI value with its [TYPE] placeholder.

    Returns the redacted text and the findings that were skipped because the
    value doesn't appear in the document (e.g. the model paraphrased it).
    """
    redacted = document_text
    skipped: list[PHIFinding] = []
    seen: set[str] = set()

    # Replace longer values first so a short value (e.g. "Austin") can't break
    # up a longer one that contains it (e.g. "Austin Regional Medical Center").
    for finding in sorted(report.findings, key=lambda f: len(f.value), reverse=True):
        value = finding.value.strip()
        if value in seen:
            continue  # duplicate finding; already replaced every occurrence
        seen.add(value)
        if not value or value not in document_text:
            skipped.append(finding)  # model reported text that isn't in the document
            continue
        redacted = redacted.replace(value, f"[{finding.type}]")

    return redacted, skipped


def run(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    output_file = Path(output_path)
    document_text = input_file.read_text(encoding="utf-8")

    print(f"Processing: {input_file}")
    print(f"Output:     {output_file}\n")

    report = find_phi(document_text)
    redacted_text, skipped = redact(document_text, report)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(redacted_text, encoding="utf-8")

    print(f"Found {len(report.findings)} PHI values:")
    for finding in report.findings:
        print(f"  [{finding.type}] {finding.value}")

    if skipped:
        print(f"\nSkipped {len(skipped)} values not found verbatim in the document:")
        for finding in skipped:
            print(f"  [{finding.type}] {finding.value!r}")

    print(f"\nRedacted document written to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--lookup":
        lookup(sys.argv[2])
    elif len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 src/agent.py <input_file> <output_file>", file=sys.stderr)
        print('       python3 src/agent.py --lookup "<question>"', file=sys.stderr)
        sys.exit(1)
