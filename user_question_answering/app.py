import streamlit as st
from transformers import pipeline, AutoTokenizer
import fitz  # PyMuPDF

# ✅ Streamlit page setup
st.set_page_config(page_title="Fast Question Answering System", layout="centered")

# ✅ Load models
@st.cache_resource
def load_pipelines():
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    qg = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
    return qa, qg, tokenizer

qa_pipeline, qg_pipeline, qg_tokenizer = load_pipelines()

# ✅ Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ UI layout
st.title("📄📥 Passage/PDF-Based QA System")
st.markdown("Upload a **PDF** or enter a **custom passage**, and either **ask a question** or let the system **generate questions**.")

# --- 📌 Input Mode Selection ---
input_mode = st.radio("Choose input type:", ["📘 Upload PDF", "📝 Enter Passage"])

# --- 📘 PDF Upload ---
context = ""
if input_mode == "📘 Upload PDF":
    uploaded_pdf = st.file_uploader("📂 Upload a PDF file:", type=["pdf"])
    if uploaded_pdf is not None:
        context = extract_text_from_pdf(uploaded_pdf)
        with st.expander("🧾 View Extracted Context"):
            st.write(context[:3000] + ("..." if len(context) > 3000 else ""))

# --- 📝 Manual Passage Entry ---
else:
    context = st.text_area("✍️ Paste or type your passage here:", height=200)
    if context:
        with st.expander("🧾 View Provided Passage"):
            st.write(context)

# --- ❓ Question Input ---
question = st.text_input("❓ Enter your question (optional if using Generate Questions):", placeholder="e.g. What is this document about?")

# --- 🧠 Action Buttons ---
col1, col2 = st.columns(2)

# --- Answer Question ---
with col1:
    if st.button("🧠 Get Answer"):
        if not context.strip() or not question.strip():
            st.warning("⚠️ Please provide a passage (or upload PDF) and enter a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa_pipeline(question=question, context=context)
                st.success(f"✅ **Answer:** {result['answer']}")

# --- Generate Questions (Safely Truncated) ---
with col2:
    if st.button("✨ Generate Questions"):
        if not context.strip():
            st.warning("⚠️ Please provide a passage or upload a PDF.")
        else:
            with st.spinner("Generating questions..."):
                # Truncate the passage to 512 tokens max
                max_input_tokens = 512
                encoded_input = qg_tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=max_input_tokens)
                decoded_context = qg_tokenizer.decode(encoded_input[0], skip_special_tokens=True)
                formatted_context = f"generate questions: {decoded_context}"

                # Generate questions
                output = qg_pipeline(
                    formatted_context,
                    max_length=64,
                    do_sample=False,
                    top_k=50,
                    num_beams=5,
                    num_return_sequences=5
                )

                st.markdown("### 📝 Possible Questions:")
                for i, res in enumerate(output):
                    st.markdown(f"**{i+1}.** {res['generated_text']}")
