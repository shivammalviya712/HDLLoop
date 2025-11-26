import os
import sys
from dotenv import load_dotenv

# Ensure the directory containing this script is in sys.path so we can import the sibling module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from matlab_ingestor import MatlabIngestor
except ImportError:
    # Fallback for when running from project root using module syntax
    from app.modules.ingestion.matlab_ingestor import MatlabIngestor

def main():
    # Load environment variables
    load_dotenv()
    
    print("Starting Ingestion Module Test...")
    
    try:
        # 1. Define Test File Path
        test_file_path = os.path.join(current_dir, "test_script.m")
        print(f"Using test file at: {test_file_path}")
        
        if not os.path.exists(test_file_path):
            print(f"Error: Test file not found at {test_file_path}")
            return

        # 2. Instantiation
        # Initialize with small chunk size to ensure we get multiple chunks for the test file
        ingestor = MatlabIngestor(chunk_size=50)
        print("MatlabIngestor instantiated.")
        
        # 3. Execution
        print("Ingesting file...")
        chunks = ingestor.ingest_file(test_file_path)
        
        # 4. Verification
        print(f"\nTotal chunks generated: {len(chunks)}")
        
        if chunks:
            first_chunk = chunks[0]
            print("\n--- First Chunk Content ---")
            print(first_chunk.text if hasattr(first_chunk, 'text') else first_chunk)
            print("---------------------------")
            
            if hasattr(first_chunk, 'token_count'):
                print(f"Token count of first chunk: {first_chunk.token_count}")
            else:
                print("Token count not available in chunk object.")
        else:
            print("Warning: No chunks were generated.")
            
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
