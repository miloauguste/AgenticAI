#!/usr/bin/env python3
"""
Script to delete the tmpizuu1pf6_fixed_kb_document.txt file from the knowledge base
"""

import sqlite3
import sys
from pathlib import Path

def delete_kb_file(filename_pattern="tmpizuu1pf6"):
    """Delete knowledge base file by filename pattern"""
    db_path = "./data/support_agent.db"
    
    if not Path(db_path).exists():
        print(f"ERROR: Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, show what we're about to delete
        print("Searching for files to delete...")
        cursor.execute(
            'SELECT id, file_name, section_title, content_type FROM file_content WHERE file_name LIKE ?',
            (f"%{filename_pattern}%",)
        )
        results = cursor.fetchall()
        
        if not results:
            print(f"SUCCESS: No files found matching pattern '{filename_pattern}'")
            return True
        
        print(f"\nFound {len(results)} records to delete:")
        for row in results:
            print(f"  - ID: {row[0]}, File: {row[1]}, Section: {row[2]}, Type: {row[3]}")
        
        # Ask for confirmation
        response = input(f"\nAre you sure you want to delete {len(results)} records? (y/N): ")
        
        if response.lower() not in ['y', 'yes']:
            print("Deletion cancelled")
            return False
        
        # Perform the deletion
        print("\nDeleting records...")
        cursor.execute(
            'DELETE FROM file_content WHERE file_name LIKE ?',
            (f"%{filename_pattern}%",)
        )
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"SUCCESS: Deleted {deleted_count} records")
        
        # Verify deletion
        cursor.execute(
            'SELECT COUNT(*) FROM file_content WHERE file_name LIKE ?',
            (f"%{filename_pattern}%",)
        )
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            print("Verification: No matching records remain in database")
        else:
            print(f"WARNING: {remaining} matching records still exist")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function"""
    print("Knowledge Base File Deletion Tool")
    print("=====================================")
    
    # Default pattern or user input
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = "tmpizuu1pf6"
    
    print(f"Target pattern: '{pattern}'")
    
    success = delete_kb_file(pattern)
    
    if success:
        print("\nOperation completed successfully!")
    else:
        print("\nOperation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()