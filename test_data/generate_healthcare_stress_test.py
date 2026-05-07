"""
Generate a healthcare stress-test CSV for the Ingestion Quality Agent.

Intentional data quality issues injected:
  - Emails: invalid formats, missing @, spaces
  - Dates: future dates, null, garbage strings
  - Currency (total_cost): N/A, --, MISSING, negatives (mixed-type numeric)
  - Categoricals: typos, unknown values outside expected set
  - Nulls: scattered across multiple columns
  - Duplicates: repeated patient_ids
  - Phone: some garbage
  - Age: out-of-range (negative, >150), strings mixed in
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

NUM_ROWS = 800
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "healthcare_patient_records.csv")

# ── Valid value pools ─────────────────────────────────────────────────────────
DEPARTMENTS = ["Cardiology", "Neurology", "Orthopedics", "Oncology", "Pediatrics",
               "Dermatology", "Radiology", "Emergency"]
DIAGNOSIS = ["Hypertension", "Diabetes Type 2", "Fracture", "Migraine", "Asthma",
             "Bronchitis", "Anemia", "COVID-19", "Flu", "Pneumonia"]
INSURANCE = ["BlueCross", "Aetna", "Cigna", "UnitedHealth", "Medicare", "Medicaid", "None"]
GENDER = ["Male", "Female", "Non-Binary"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
DISCHARGE_STATUS = ["Discharged", "Transferred", "Deceased", "Under Observation"]

FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
               "Linda", "David", "Elizabeth", "Aisha", "Raj", "Yuki", "Carlos", "Fatima"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Martinez", "Patel", "Kim", "Tanaka", "Chen", "Ali"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hospital.org", "health.net"]

# ── Corruption pools ─────────────────────────────────────────────────────────
CORRUPT_EMAILS = ["notanemail", "user@@double.com", "missing-at-sign.com",
                  "spaces in@email.com", "@nodomain", "user@", "", "N/A",
                  "test@.com", "a@b"]
CORRUPT_DATES = ["not-a-date", "31/13/2024", "TBD", "PENDING", "", "N/A",
                 "2099-01-01", "2030-12-25", "00/00/0000"]
CORRUPT_COSTS = ["N/A", "--", "MISSING", "null", "TBD", "n/a", "none",
                 "free", "UNKNOWN", "???"]
CORRUPT_DEPARTMENTS = ["Cardiolgy", "neuro", "ORTHO", "oncollogy", "Unknown",
                       "XYZ", "N/A", ""]
CORRUPT_DIAGNOSIS = ["Hypertnsion", "diabtes", "UNKNOWN", "TBD", "N/A", "???",
                     "something_wrong", ""]


def generate_valid_email(first, last):
    return f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}"


def generate_phone():
    return f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"


def generate_date(start_year=2022, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def generate_row(row_id):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)

    row = {
        "patient_id": f"PAT-{row_id:05d}",
        "patient_name": f"{first} {last}",
        "email": generate_valid_email(first, last),
        "phone": generate_phone(),
        "age": random.randint(1, 95),
        "gender": random.choice(GENDER),
        "blood_type": random.choice(BLOOD_TYPES),
        "admission_date": generate_date(),
        "department": random.choice(DEPARTMENTS),
        "diagnosis": random.choice(DIAGNOSIS),
        "total_cost": round(random.uniform(500, 50000), 2),
        "insurance_provider": random.choice(INSURANCE),
        "discharge_status": random.choice(DISCHARGE_STATUS),
    }
    return row


rows = [generate_row(i) for i in range(1, NUM_ROWS + 1)]

# ── Inject corruptions ───────────────────────────────────────────────────────
corruption_log = {
    "bad_emails": 0, "bad_dates": 0, "bad_costs": 0, "bad_departments": 0,
    "bad_diagnosis": 0, "null_injections": 0, "duplicate_ids": 0,
    "bad_ages": 0, "future_dates": 0, "bad_phones": 0,
}

# ~8% bad emails
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.08)):
    rows[i]["email"] = random.choice(CORRUPT_EMAILS)
    corruption_log["bad_emails"] += 1

# ~5% bad admission dates
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.05)):
    rows[i]["admission_date"] = random.choice(CORRUPT_DATES)
    corruption_log["bad_dates"] += 1

# ~6% corrupt total_cost (mixed-type: strings in a numeric column)
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.06)):
    rows[i]["total_cost"] = random.choice(CORRUPT_COSTS)
    corruption_log["bad_costs"] += 1

# ~4% negative total_cost
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.04)):
    rows[i]["total_cost"] = round(-random.uniform(100, 5000), 2)
    corruption_log["bad_costs"] += 1

# ~7% bad departments (typos, unknown)
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.07)):
    rows[i]["department"] = random.choice(CORRUPT_DEPARTMENTS)
    corruption_log["bad_departments"] += 1

# ~5% bad diagnosis
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.05)):
    rows[i]["diagnosis"] = random.choice(CORRUPT_DIAGNOSIS)
    corruption_log["bad_diagnosis"] += 1

# ~3% null injections across random columns
nullable_cols = ["email", "phone", "age", "gender", "blood_type",
                 "admission_date", "department", "diagnosis",
                 "total_cost", "insurance_provider", "discharge_status"]
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.03)):
    col = random.choice(nullable_cols)
    rows[i][col] = ""
    corruption_log["null_injections"] += 1

# ~2% duplicate patient_ids
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.02)):
    donor = random.randint(0, NUM_ROWS - 1)
    rows[i]["patient_id"] = rows[donor]["patient_id"]
    corruption_log["duplicate_ids"] += 1

# ~3% bad ages (negatives, impossibly high, strings)
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.03)):
    rows[i]["age"] = random.choice([-5, -10, 200, 350, "N/A", "unknown", ""])
    corruption_log["bad_ages"] += 1

# ~2% bad phones
for i in random.sample(range(NUM_ROWS), int(NUM_ROWS * 0.02)):
    rows[i]["phone"] = random.choice(["not-a-phone", "12345", "N/A", "", "???"])
    corruption_log["bad_phones"] += 1

# ── Write CSV ─────────────────────────────────────────────────────────────────
fieldnames = ["patient_id", "patient_name", "email", "phone", "age", "gender",
              "blood_type", "admission_date", "department", "diagnosis",
              "total_cost", "insurance_provider", "discharge_status"]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Generated: {OUTPUT_PATH}")
print(f"   Rows: {NUM_ROWS}")
print(f"   Columns: {len(fieldnames)}")
print(f"\n   Corruption summary:")
for k, v in corruption_log.items():
    print(f"     {k}: {v}")
