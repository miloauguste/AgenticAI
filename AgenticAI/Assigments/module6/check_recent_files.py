#!/usr/bin/env python3
"""
Check recent file entries in the main database
"""

import sqlite3
import os

def check_recent_files():
    """Check recent file entries"""
    
    db_path = "./data/support_agent.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for PDF files specifically
        cursor.execute("SELECT file_name, file_type, created_at FROM file_content WHERE file_type = 'pdf' ORDER BY created_at DESC LIMIT 5")
        pdf_files = cursor.fetchall()
        
        print("Recent PDF files:")
        if pdf_files:
            for filename, ftype, created in pdf_files:
                print(f"  - {filename} ({ftype}) at {created}")
        else:
            print("  No PDF files found")
        
        # Check recent files of any type
        cursor.execute("SELECT file_name, file_type, created_at FROM file_content ORDER BY created_at DESC LIMIT 10")
        recent_files = cursor.fetchall()
        
        print("\nMost recent files (any type):")
        for filename, ftype, created in recent_files:
            print(f"  - {filename} ({ftype}) at {created}")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    check_recent_files()