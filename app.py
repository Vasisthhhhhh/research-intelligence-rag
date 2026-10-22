
import streamlit as st

from src.retrieval.search import Search
from src.generation.generator import Generator


st.set_page_config(
    page_title="Research Intelligence RAG",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Research Intelligence RAG")
st.caption(
    "Ask questions about the research paper using "
    "BGE dense retrieval + Qdrant + Qwen."
)


@st.cache_resource
def load_models():
    searcher = Search()
    generator = Generator()
    return searcher, generator


searcher, generator = load_models()


query = st.text_input(
    "Ask a question about the paper",
    placeholder="What is the main objective of DREEP?",
)


if st.button("Ask Question", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching the paper..."):

            results = searcher.search(
                query,
                limit=5,
            )

            context_parts = []

            for result in results:

                chunk = result["chunk"]

                context_parts.append(
                    f"[Page {chunk['page_number']}, "
                    f"Chunk {chunk['chunk_id']}]\n"
                    f"{chunk['text']}"
                )

            context = "\n\n".join(
                context_parts
            )

        with st.spinner("Generating answer..."):

            answer = generator.generate(
                query=query,
                context=context,
            )

        st.subheader("Sources")

        for rank, result in enumerate(results, start=1):

            chunk = result["chunk"]

            with st.expander(
                f"Source {rank} · "
                f"Page {chunk['page_number']} · "
                f"Chunk {chunk['chunk_id']} · "
                f"Score {result['score']:.4f}"
            ):
                
                st.write(chunk["text"])
            