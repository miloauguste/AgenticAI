#!/usr/bin/env python3
"""
Deep debugging of PDF embedding issues
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def deep_pdf_debug():
    """Deep dive into PDF embedding issues"""
    
    try:
        from enhanced_support_agent import EnhancedSupportTriageAgent
        
        print("Deep PDF Debug")
        print("==============")
        
        # Initialize agent
        agent = EnhancedSupportTriageAgent()
        
        # Find PDF vectors
        pdf_vectors = []
        txt_vectors = []
        
        for vector_id, vector_data in agent.vector_store.in_memory_store.items():
            metadata = vector_data.get('metadata', {})
            file_type = metadata.get('file_type', 'unknown')
            
            if file_type == 'pdf':
                pdf_vectors.append((vector_id, vector_data))
            elif file_type == 'txt':
                txt_vectors.append((vector_id, vector_data))
        
        print(f"PDF vectors: {len(pdf_vectors)}")
        print(f"TXT vectors: {len(txt_vectors)}")
        
        if not pdf_vectors:
            print("ERROR: No PDF vectors found!")
            return
        
        # Check PDF embedding quality
        test_query = "YNC E-COMMERCE"
        print(f"\nTesting with query: '{test_query}'")
        
        # Generate query embedding
        query_embedding = agent.vector_store.generate_embeddings([test_query])[0]
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Check each PDF vector manually
        print(f"\nChecking PDF vectors manually:")
        pdf_scores = []
        
        for i, (vector_id, vector_data) in enumerate(pdf_vectors):
            text = vector_data.get('text', '')
            embedding = vector_data.get('embedding')
            metadata = vector_data.get('metadata', {})
            
            print(f"\nPDF {i+1}: {vector_id}")
            print(f"  File: {metadata.get('file_name', 'unknown')}")
            print(f"  Text length: {len(text)}")
            print(f"  Text preview: {text[:80]}...")
            
            if embedding is not None:
                try:
                    # Calculate similarity manually
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    # Normalize embeddings
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    doc_norm = embedding / np.linalg.norm(embedding)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_norm, doc_norm)
                    pdf_scores.append((similarity, vector_id, text[:100]))
                    
                    print(f"  Embedding shape: {embedding.shape}")
                    print(f"  Manual similarity: {similarity:.4f}")
                    
                    # Check if text contains the query
                    query_in_text = test_query.lower() in text.lower()
                    print(f"  Query in text: {query_in_text}")
                    
                except Exception as e:
                    print(f"  Embedding error: {e}")
            else:
                print(f"  ERROR: No embedding found!")
        
        # Compare with TXT vectors
        print(f"\nComparing with top TXT vectors:")
        txt_scores = []
        
        for i, (vector_id, vector_data) in enumerate(txt_vectors[:3]):
            text = vector_data.get('text', '')
            embedding = vector_data.get('embedding')
            metadata = vector_data.get('metadata', {})
            
            if embedding is not None:
                try:
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    doc_norm = embedding / np.linalg.norm(embedding)
                    similarity = np.dot(query_norm, doc_norm)
                    txt_scores.append((similarity, vector_id, text[:100]))
                    
                    print(f"\nTXT {i+1}: {vector_id}")
                    print(f"  File: {metadata.get('file_name', 'unknown')}")
                    print(f"  Manual similarity: {similarity:.4f}")
                    print(f"  Text preview: {text[:80]}...")
                    
                except Exception as e:
                    print(f"  TXT embedding error: {e}")
        
        # Sort and compare scores
        pdf_scores.sort(reverse=True)
        txt_scores.sort(reverse=True)
        
        print(f"\n" + "="*50)
        print("SCORE COMPARISON")
        print(f"Best PDF score: {pdf_scores[0][0]:.4f}" if pdf_scores else "No PDF scores")
        print(f"Best TXT score: {txt_scores[0][0]:.4f}" if txt_scores else "No TXT scores")
        
        if pdf_scores and txt_scores:
            if pdf_scores[0][0] > txt_scores[0][0]:
                print("✅ PDF has higher similarity than TXT")
            else:
                print("❌ TXT has higher similarity than PDF")
                print(f"   Difference: {txt_scores[0][0] - pdf_scores[0][0]:.4f}")
        
        # Test actual search function
        print(f"\nTesting actual search function:")
        search_results = agent.vector_store.search_similar(test_query, top_k=15)
        
        print(f"Total search results: {len(search_results)}")
        pdf_results = [r for r in search_results if r.get('metadata', {}).get('file_type') == 'pdf']
        txt_results = [r for r in search_results if r.get('metadata', {}).get('file_type') == 'txt']
        
        print(f"PDF results in search: {len(pdf_results)}")
        print(f"TXT results in search: {len(txt_results)}")
        
        if pdf_results:
            print(f"Best PDF result score: {pdf_results[0].get('score', 0):.4f}")
            print(f"PDF result rank: {search_results.index(pdf_results[0]) + 1}")
        
        # Show all results with scores
        print(f"\nAll search results (top 10):")
        for i, result in enumerate(search_results[:10]):
            metadata = result.get('metadata', {})
            file_type = metadata.get('file_type', 'unknown')
            score = result.get('score', 0)
            text_preview = result.get('text', '')[:40]
            print(f"  {i+1:2d}. {file_type.upper()} - {score:.4f} - {text_preview}...")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deep_pdf_debug()