#!/usr/bin/env python3
"""
Fix PDF vectors by removing old ones and re-processing with correct structure
"""

import os
import sys
import shutil

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def fix_pdf_vectors():
    """Remove PDF vectors and re-process with correct structure"""
    
    try:
        from simple_vector_response import VectorStore
        from enhanced_support_agent import EnhancedSupportTriageAgent
        
        print("PDF Vector Fix")
        print("==============")
        
        # Initialize vector store
        vector_store = VectorStore()
        print(f"Initial vector count: {len(vector_store.in_memory_store)}")
        
        # Remove all PDF vectors
        pdf_vector_ids = []
        for vector_id, vector_data in vector_store.in_memory_store.items():
            metadata = vector_data.get('metadata', {})
            if metadata.get('file_type') == 'pdf':
                pdf_vector_ids.append(vector_id)
        
        print(f"Found {len(pdf_vector_ids)} PDF vectors to remove")
        
        # Remove PDF vectors
        for vector_id in pdf_vector_ids:
            del vector_store.in_memory_store[vector_id]
        
        print(f"Removed PDF vectors. New count: {len(vector_store.in_memory_store)}")
        
        # Save the cleaned vector store
        vector_store._save_persistent_data()
        print("Saved cleaned vector store")
        
        # Re-process PDF with fixed enhanced agent
        print("\nRe-processing PDF with fixed structure...")
        
        agent = EnhancedSupportTriageAgent()
        
        # Process the PDF again
        pdf_path = "./policy.pdf"
        if os.path.exists(pdf_path):
            result = agent.add_knowledge_from_file(pdf_path)
            
            if result.get('success'):
                print(f"✅ PDF re-processing successful!")
                print(f"   Sections: {result.get('sections_created', 0)}")
                print(f"   Chunks: {result.get('chunks_created', 0)}")
            else:
                print(f"❌ PDF re-processing failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ PDF file not found: {pdf_path}")
        
        # Verify the fix
        print("\nVerifying fix...")
        final_vector_count = len(agent.vector_store.in_memory_store)
        print(f"Final vector count: {final_vector_count}")
        
        # Check new PDF vector structure
        pdf_vectors = []
        for vector_id, vector_data in agent.vector_store.in_memory_store.items():
            metadata = vector_data.get('metadata', {})
            if metadata.get('file_type') == 'pdf':
                pdf_vectors.append((vector_id, vector_data))
        
        print(f"New PDF vectors: {len(pdf_vectors)}")
        
        if pdf_vectors:
            sample_vector = pdf_vectors[0][1]
            has_text = 'text' in sample_vector
            has_content = 'content' in sample_vector
            print(f"Sample PDF vector structure:")
            print(f"  Has 'text' field: {has_text}")
            print(f"  Has 'content' field: {has_content}")
            
            if has_text:
                text_content = sample_vector.get('text', '')
                print(f"  Text field length: {len(text_content)}")
                print(f"  Text preview: {text_content[:100]}...")
                
        # Test search
        print("\nTesting search...")
        try:
            search_results = agent.vector_store.search_similar("YNC E-COMMERCE", top_k=5)
            pdf_results = [r for r in search_results if r.get('metadata', {}).get('file_type') == 'pdf']
            print(f"Search results: {len(search_results)} total, {len(pdf_results)} PDF results")
            
            if pdf_results:
                best_pdf = pdf_results[0]
                print(f"✅ PDF found in search! Score: {best_pdf.get('score', 0):.3f}")
                print(f"   Content: {best_pdf.get('text', '')[:100]}...")
            else:
                print("❌ No PDF results in search")
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_pdf_vectors()