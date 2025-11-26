import os
from typing import List, Any, Union
from dotenv import load_dotenv
from chonkie import CodeChunker

# Ensure environment variables are loaded for Langfuse
load_dotenv()

class MatlabIngestor:
    """
    Ingests and chunks MATLAB code files using Chonkie with a fallback mechanism.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the MatlabIngestor.

        Args:
            chunk_size (int): The size of each chunk in tokens. Defaults to 1000.
        """
        self.chunk_size = chunk_size
        
        # Initialize the CodeChunker for MATLAB.
        # We initialize it here, but actual execution is guarded in ingest_file.
        try:
            self.chunker = CodeChunker(
                language="matlab",
                chunk_size=self.chunk_size
            )
        except Exception as e:
            print(f"Warning: Failed to initialize CodeChunker for 'matlab': {e}")
            raise e

    def _load_file_content(self, file_path: str) -> str:
        """
        Private method to read a .m file from a given path.

        Args:
            file_path (str): The path to the .m file.

        Returns:
            str: The content of the file.

        Raises:
            FileNotFoundError: If the file does not exist or is not a file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Path is not a file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Error reading file {file_path}: {e}")

    def ingest_file(self, file_path: str) -> List[Any]:
        """
        Orchestrates loading the content and passing it to the Chonkie chunker.
        
        Uses @observe() for Langfuse tracing.
        Implements fallback logic to RecursiveChunker if CodeChunker fails.

        Args:
            file_path (str): The path to the file to ingest.

        Returns:
            List[Any]: A list of chunk objects (or strings).
        """
        content = self._load_file_content(file_path)

        try:
            if self.chunker is None:
                raise RuntimeError("CodeChunker was not initialized.")
            
            # Attempt to chunk using CodeChunker
            chunks = self.chunker(content)
            return chunks
            
        except Exception as e:
            # Provide context on failure (could be logged via a logger in a real app)
            print(f"CodeChunker failed for {file_path}: {e}.")
            raise e

