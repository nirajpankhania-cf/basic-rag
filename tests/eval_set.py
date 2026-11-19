"""Fixed evaluation set: questions with known source chunks and known answers.

Each case has:
  id               short name used in test output
  question         what the employee asks
  expected_chunks  chunk ids that contain the answer; retrieval passes if
                   ANY of them is in the retrieved top-k
  key_facts        facts the generated answer must contain. Every fact must
                   be present; each fact is a list of acceptable phrasings
                   (e.g. "one hour" or "1 hour").

The set mixes natural-language questions (which favour vector search) with
section-code questions (which need keyword search), across both the remote
work and expense policies.
"""

CASES = [
    {
        "id": "home_office_stipend",
        "question": "How much is the home office stipend?",
        "expected_chunks": ["remote_work_policy::REM-4"],
        "key_facts": [["600"]],
    },
    {
        "id": "core_hours",
        "question": "What are the core hours everyone must be available?",
        "expected_chunks": ["remote_work_policy::REM-3"],
        "key_facts": [
            ["10:00", "10 a.m", "10 am", "10am"],
            ["3:00", "3 p.m", "3 pm", "3pm"],
        ],
    },
    {
        "id": "international_remote_days",
        "question": "How many days per year can I work from another country?",
        "expected_chunks": ["remote_work_policy::REM-6"],
        "key_facts": [["20"]],
    },
    {
        "id": "lost_laptop",
        "question": "How quickly must I report a lost or stolen laptop?",
        "expected_chunks": ["remote_work_policy::REM-5"],
        "key_facts": [["one hour", "1 hour", "an hour"]],
    },
    {
        "id": "in_person_gatherings",
        "question": "How often are remote employees expected to come on site?",
        "expected_chunks": ["remote_work_policy::REM-8"],
        "key_facts": [["two", "2", "twice"]],
    },
    {
        "id": "rem9_section_code",
        "question": "What does REM-9 say about ending a remote work arrangement?",
        "expected_chunks": ["remote_work_policy::REM-9"],
        "key_facts": [["30"]],
    },
    {
        "id": "expense_deadline",
        "question": "What is the deadline for submitting an expense claim?",
        "expected_chunks": ["expense_policy::EXP-2"],
        "key_facts": [["30"]],
    },
    {
        "id": "hotel_limit",
        "question": "What is the nightly hotel limit in a standard city?",
        "expected_chunks": ["expense_policy::EXP-3"],
        "key_facts": [["220"]],
    },
    {
        "id": "meal_limit",
        "question": "How much can I claim for meals per day when travelling?",
        "expected_chunks": ["expense_policy::EXP-4"],
        "key_facts": [["75"]],
    },
    {
        "id": "large_claim_approval",
        "question": "Who has to approve an expense claim above 2,500 dollars?",
        "expected_chunks": ["expense_policy::EXP-6"],
        "key_facts": [["department head"], ["finance director"]],
    },
    {
        "id": "exp7_section_code",
        "question": "What does EXP-7 say about a lost or stolen card?",
        "expected_chunks": ["expense_policy::EXP-7"],
        "key_facts": [["24"]],
    },
    {
        "id": "development_budget",
        "question": "What is the annual professional development budget?",
        "expected_chunks": ["expense_policy::EXP-8"],
        "key_facts": [["1500"]],
    },
    # Bare section-code questions: nothing for the embedding to latch on to
    # except the code itself, so these depend on keyword search.
    {
        "id": "exp4_bare_code",
        "question": "What does EXP-4 say?",
        "expected_chunks": ["expense_policy::EXP-4"],
        "key_facts": [["75"]],
    },
    {
        "id": "rem6_bare_code",
        "question": "Summarize section REM-6",
        "expected_chunks": ["remote_work_policy::REM-6"],
        "key_facts": [["30"], ["20"]],
    },
]

# Minimum scores for the aggregate tests to pass.
RECALL_THRESHOLD = 0.9
ACCURACY_THRESHOLD = 0.8
