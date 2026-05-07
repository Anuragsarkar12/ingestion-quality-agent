# generate_data.py
# This script creates a realistic but intentionally flawed dataset
# for our agent to process.

import pandas as pd                    # For creating and manipulating data tables
import random                          # For generating random values
from faker import Faker                # For generating realistic fake names, emails, etc.
from datetime import datetime, timedelta  # For date calculations
import os                              # For creating folders

# Initialize the Faker library (en_US = US English locale)
fake = Faker('en_US')

# Set a random seed so we get the same data every time we run this
# (good for reproducibility when debugging)
random.seed(42)
fake.seed_instance(42)

def generate_clean_order():
    """Generate one valid, clean order record."""
    # Generate a date between 6 months ago and yesterday (all valid past dates)
    days_ago = random.randint(1, 180)
    order_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    return {
        'order_id': None,              # Will be set later with sequential ID
        'customer_id': random.randint(1000, 9999),
        'customer_name': fake.name(),
        'email': fake.email(),
        'product': random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard']),
        'quantity': random.randint(1, 10),
        'order_amount': round(random.uniform(10.0, 2000.0), 2),  # Between $10 and $2000
        'order_date': order_date,
        'status': random.choice(['pending', 'processing', 'shipped', 'delivered']),
        'region': random.choice(['North', 'South', 'East', 'West'])
    }

def inject_problems(records):
    """
    Deliberately corrupt some records to simulate real-world data quality issues.
    This is what our agent will detect and fix.
    """
    corrupted = records.copy()
    total = len(corrupted)
    
    # Problem 1: Negative order amounts (5% of records)
    # Scenario: A billing system bug negated some amounts
    negative_indices = random.sample(range(total), int(total * 0.05))
    for i in negative_indices:
        corrupted[i]['order_amount'] = -abs(corrupted[i]['order_amount'])
    
    # Problem 2: Future dates (3% of records)
    # Scenario: A timezone bug set dates to next year
    future_indices = random.sample(
        [i for i in range(total) if i not in negative_indices],
        int(total * 0.03)
    )
    for i in future_indices:
        days_ahead = random.randint(1, 365)
        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        corrupted[i]['order_date'] = future_date
    
    # Problem 3: Missing emails (4% of records)
    # Scenario: Some customers opted out and the field was set to empty
    null_email_indices = random.sample(
        [i for i in range(total) if i not in negative_indices + future_indices],
        int(total * 0.04)
    )
    for i in null_email_indices:
        corrupted[i]['email'] = None  # None becomes NULL in the database
    
    # Problem 4: Invalid status values (2% of records)
    # Scenario: A new system introduced non-standard status codes
    invalid_statuses = ['shipped_maybe', 'unknown', 'ERROR', 'pending_review', '']
    status_indices = random.sample(
        [i for i in range(total) if i not in negative_indices + future_indices + null_email_indices],
        int(total * 0.02)
    )
    for i in status_indices:
        corrupted[i]['status'] = random.choice(invalid_statuses)
    
    # Problem 5: Duplicate order IDs (create 10 duplicates)
    # Scenario: A retry mechanism sent the same order twice
    dup_source_indices = random.sample(range(total), 10)
    duplicates = [corrupted[i].copy() for i in dup_source_indices]
    corrupted.extend(duplicates)
    
    return corrupted

def main():
    print("🔧 Generating mock order data...")
    
    # Create folder structure
    os.makedirs('data/raw', exist_ok=True)       # exist_ok=True means don't error if folder exists
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/quarantine', exist_ok=True)
    os.makedirs('database', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('great_expectations/expectations', exist_ok=True)
    
    # Generate 500 clean records
    print("  Generating 500 clean records...")
    records = [generate_clean_order() for _ in range(500)]
    
    # Inject problems
    print("  Injecting data quality problems...")
    records = inject_problems(records)
    
    # Assign sequential order IDs (1 through N)
    for i, record in enumerate(records):
        record['order_id'] = i + 1001  # Start from 1001
    
    # Convert to a pandas DataFrame (a table-like data structure)
    df = pd.DataFrame(records)
    
    # Reorder columns for readability
    columns = ['order_id', 'customer_id', 'customer_name', 'email', 
               'product', 'quantity', 'order_amount', 'order_date', 'status', 'region']
    df = df[columns]
    
    # Save to CSV
    output_path = 'data/raw/orders_raw.csv'
    df.to_csv(output_path, index=False)  # index=False means don't save row numbers
    
    print(f"  ✅ Saved {len(df)} records to {output_path}")
    print(f"  📊 Summary:")
    print(f"     Total records: {len(df)}")
    print(f"     Null emails: {df['email'].isna().sum()}")
    print(f"     Negative amounts: {(df['order_amount'] < 0).sum()}")
    print(f"     Future dates: {(pd.to_datetime(df['order_date']) > datetime.now()).sum()}")
    print(f"     Invalid statuses: {(~df['status'].isin(['pending', 'processing', 'shipped', 'delivered'])).sum()}")
    
    # Load into SQLite database
    print("\n  Loading data into SQLite database...")
    import sqlite3
    conn = sqlite3.connect('database/orders.db')
    
    # Write the DataFrame to a SQL table named 'orders_staging'
    # if_exists='replace' means overwrite it if it already exists
    df.to_sql('orders_staging', conn, if_exists='replace', index=False)
    
    # Also create an empty 'orders_clean' table for validated records
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders_clean AS 
        SELECT * FROM orders_staging WHERE 1=0
    """)  # WHERE 1=0 means copy the structure but no rows
    
    # Create a quarantine table for bad records
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders_quarantine AS 
        SELECT *, '' as reject_reason FROM orders_staging WHERE 1=0
    """)
    
    conn.commit()  # Save changes
    conn.close()   # Close database connection
    
    print("  ✅ Database initialized at database/orders.db")
    print("\n🎉 Data setup complete! Ready to run the agent.")

if __name__ == "__main__":
    main()