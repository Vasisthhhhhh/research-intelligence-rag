import pymupdf


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract meaningful text blocks from each PDF page."""

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        blocks = page.get_text("blocks")

        page_blocks = []

        for block in blocks:
            text = block[4].strip()

            if text:
                page_blocks.append(text)

        if page_blocks:
            pages.append(
                {
                    "page_number": page_number,
                    "blocks": page_blocks,
                }
            )

    document.close()

    return pages