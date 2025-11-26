from __future__ import annotations

from dotenv import load_dotenv

from vector_memory import VectorMemory


def run_demo() -> None:
    """Simple smoke test for the in-memory vector store."""
    load_dotenv()

    plot_chunk = (
        "% Plot sine wave\n"
        "x = linspace(0, 2*pi, 1000);\n"
        "y = sin(x);\n"
        "plot(x, y);\n"
        "title('Sine Wave');\n"
        "xlabel('Time'); ylabel('Amplitude');"
    )
    matrix_chunk = (
        "% Matrix multiplication\n"
        "A = [1 2; 3 4];\n"
        "B = [5; 6];\n"
        "result = A * B;"
    )
    io_chunk = (
        "% File I/O\n"
        "data = load('input.mat');\n"
        "save('output.mat', 'data');"
    )
    chunks = [plot_chunk, matrix_chunk, io_chunk]

    memory = VectorMemory()
    memory.index_chunks(chunks)

    query = "How do I plot a graph?"
    results = memory.similarity_search(query, k=1)

    assert results, "VectorMemory returned no results."
    assert results[0] == plot_chunk, "Plotting chunk should be the top result."

    print("Retrieved chunk:\n", results[0])


if __name__ == "__main__":
    run_demo()