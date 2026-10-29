from src.chunking import chunk_text

def test_offsets_reconstruct_text():
    text = "A paragraph.\\nSecond paragraph.\\nThird paragraph."
    chunks = chunk_text(text, "doc.txt", chunk_chars=20, overlap_chars=5)
    assert chunks[0].start == 0
    assert all(text[c.start:c.end] == c.text for c in chunks)
    assert chunks[-1].end == len(text)
