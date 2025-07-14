# %pip install pandas==2.2.3 jupyter==1.1.1 langchain==0.3.23 langchain-community==0.3.21 rich==14.0.0 openai==1.71.0 langchain-groq==0.3.2 langchain-ollama==0.3.1 faiss-gpu==1.7.2 numpy<2 rouge-score 

import logging
import json
import os
import re
import spacy
from rich.console import Console
from rich.logging import RichHandler
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rouge_score import rouge_scorer

# Logging setup
console = Console(stderr=True, record=True)
log_handler = RichHandler(rich_tracebacks=True, console=console, markup=True)
logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[log_handler])
log = logging.getLogger("rich")
log.setLevel(logging.DEBUG)

# Global Config
DEBUG: bool = False
PUBLIC_DATASET: str = "public_dataset.json"
PRIVATE_DATASET: str = "private_dataset.json"
OUTPUT_PATH: str = "Submissions/output.json"
ROUGE_OUTPUT_PATH: str = "Submissions/rouge_scores.json"

# Using Groq API
llm = ChatGroq(
  model="llama-3.1-8b-instant",
  api_key = "api_key", # Replace with your Groq API key
  temperature=0.05,
  max_tokens=128,
)

nlp_model = spacy.load("en_core_web_sm")
def dataset_processor(demo_full_text):
     # Split the full text into sections
    sections = demo_full_text.split("\n\n\n")[:-1]

    merged_sections = []
    for section in sections:
        combined_text = " ".join(section.split("\n")).strip()
        merged_sections.append(combined_text)

    documents = [Document(page_content = doc) for doc in merged_sections]

    sentence_docs = []
    for doc in documents:
        spacy_doc = nlp_model(doc.page_content)
        # Filter out short sentences
        sentences = [sent.text.strip() for sent in spacy_doc.sents if len(sent.text.strip()) >= 40]
        sentence_docs.append(Document(page_content = " ".join(sentences)))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512, # number of characters
        chunk_overlap = 256,
        separators=["\n\n", "\n", ".", "。", "!", "?", " ", ""],
        length_function = len,
        add_start_index = True
    )
    chunked_documents = text_splitter.split_documents(sentence_docs)

    return chunked_documents

def build_vector_store(document):
    # Preprocess and chunk the document
    docs_splits = (dataset_processor(document))
     # Filter out chunks
    valid_documents = [
        doc for doc in docs_splits
        if len(doc.page_content.strip()) >= 80 and len(doc.page_content.strip().split()) >= 12
    ]

    embeddings = OllamaEmbeddings(
      model="mxbai-embed-large", 
      keep_alive=3000,
    )
    vector_store = InMemoryVectorStore.from_documents(valid_documents, embeddings)
    return vector_store

def construct_retrieval_chain(llm, vector_store):

    SYSTEM_PROMPT: str = """You are a helpful assistant.
    Answer the question based on the context below.

    Make sure your answer:
    - is concise and based on facts from the context
    - does not make assumptions beyond the given information
    - does not include incomplete or fragmented sentences


    Answer in a complete sentence. Do not repeat the question.
    """
    CHAT_TEMPLATE_RAG = (
    f"""system: {SYSTEM_PROMPT}
    human: context: {{context}}\nquestion: {{input}}
    assistant: """
    )
    retrieval_qa_prompt = PromptTemplate.from_template(template=CHAT_TEMPLATE_RAG)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
    rag_qa_chain = create_retrieval_chain(
      retriever=vector_store.as_retriever(
          search_kwargs={"k":5, "fetch_k": 15, "lambda_mult": 0.9}, 
        search_type="mmr"
          
        ), 
      combine_docs_chain=combine_docs_chain
    )
    return rag_qa_chain

def evaluate_evidence_with_rouge(predicted_evidences, reference_evidences):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    fmeasure_scores = []

    for pred in predicted_evidences:
        scores = scorer.score_multi(
            targets=reference_evidences,
            prediction=pred
        )
        fmeasure_scores.append(scores["rougeL"].fmeasure)

    if fmeasure_scores:
        return sum(fmeasure_scores) / len(fmeasure_scores)
    return 0.0

def produce_answer_with_evidence(entry: dict, llm) -> tuple[dict, list[str]]:
    # Build vector store and RAG chain
    vector_store = build_vector_store(entry["full_text"])
    rag_chain = construct_retrieval_chain(llm, vector_store)

    # Question-answer processing
    question = entry["question"]
    query_input = {"input": question}
    response = rag_chain.invoke(query_input)

    # answer = str(response.get("answer", "I don't know"))
    # clean_answer = re.sub(r"\s+", " ", answer).strip()
    evidences = [doc.page_content for doc in response["context"]]

    result = {
        "title": entry["title"],
        # "answer": response.get("answer", "I don't know"),
        "answer": response["answer"],
        "evidence": evidences,
    }

    return result, evidences

def main():
    with open(PRIVATE_DATASET, "r") as f:
        data = json.load(f)

    output = []
    rouge_scores = []

    for index, entry in enumerate(data):
        # #test
        # if index >= 5:
        #     break
        log.info(f"question #{index + 1}")
        result, evidences = produce_answer_with_evidence(entry, llm)
        output.append(result)

        if "evidence" in entry and entry["evidence"]:
            rouge_score = evaluate_evidence_with_rouge(evidences, entry["evidence"])
            rouge_scores.append(rouge_score)
            log.info(f"score #{index + 1}: {rouge_score:.4f}")

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {OUTPUT_PATH}")

    # Save ROUGE scores
    if rouge_scores:
        with open(ROUGE_OUTPUT_PATH, "w") as f:
            json.dump(rouge_scores, f, indent=4)
        log.info(f"ROUGE scores saved to {ROUGE_OUTPUT_PATH}")

        # Display average ROUGE-L score
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        log.info(f"[bold green]Average ROUGE-L score: {avg_rouge:.4f}[/bold green]")

if __name__ == "__main__":
    main()