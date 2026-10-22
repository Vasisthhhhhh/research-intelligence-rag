def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Combine PDF text blocks into retrieval-friendly chunks
    while preserving page metadata.
    """

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page_number"]
        blocks = page["blocks"]

        current_blocks = []
        current_length = 0

        for block in blocks:
            block_length = len(block)

            if current_blocks and current_length + block_length > chunk_size:
                chunk_text = "\n\n".join(current_blocks).strip()

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

                # Keep the final block as overlap.
                current_blocks = [current_blocks[-1]]
                current_length = len(current_blocks[-1])

            current_blocks.append(block)
            current_length += block_length

        # Store remaining blocks on the page.
        if current_blocks:
            chunk_text = "\n\n".join(current_blocks).strip()

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return chunks