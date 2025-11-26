import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root on path for module imports when running directly.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from matlab_agent import MatlabAgent
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


def ensure_test_file(file_path: Path) -> None:
    """Create a small MATLAB script for the agent to refactor."""
    if file_path.exists():
        return

    file_path.write_text(
        (
            "function result = nested_loops_example(A, B)\n"
            "    result = zeros(size(A));\n"
            "    for i = 1:length(A)\n"
            "        for j = 1:length(B)\n"
            "            result(i) = result(i) + A(i) * B(j);\n"
            "        end\n"
            "    end\n"
            "end\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()

    test_file_path = CURRENT_DIR / "agent_test_script.m"
    ensure_test_file(test_file_path)

    agent = MatlabAgent()
    query = "Change the nested for-loops to a matrix operation."
    generated_code = agent.run(
        query=query,
        file_path=str(test_file_path)
    )

    print("\n=== Generated MATLAB Code ===\n")
    print(generated_code)
    print("\n=============================\n")


if __name__ == "__main__":
    main()

