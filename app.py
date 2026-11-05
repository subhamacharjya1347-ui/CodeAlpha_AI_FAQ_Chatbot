import json
import re
import difflib
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. PROFESSIONAL UI DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60a5fa;
}
.subtitle {
    color: #cbd5e1;
    font-size: 16px;
}
.stButton button {
    border-radius: 10px;
    border: 1px solid #60a5fa;
    background-color: #1e3a8a;
    color: white;
    width: 100%;
}
.stButton button:hover {
    background-color: #2563eb;
    border-color: #93c5fd;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD FAQ DATABASE
# ==================================================

@st.cache_data
def load_faqs():
    faq_path = Path(__file__).parent / "faqs.json"

    with open(faq_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("The FAQ database is empty or invalid.")

    for item in data:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise ValueError(
                "Each FAQ must contain both 'question' and 'answer' fields."
            )

    return data


try:
    faqs = load_faqs()
except Exception as error:
    st.error(f"Could not load faqs.json: {error}")
    st.stop()

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]


# Assign a simple display category to each FAQ from its question text.
# This does not change matching or require edits to faqs.json.
def get_faq_category(question):
    text = question.lower()

    if any(word in text for word in ["admission", "student id", "syllabus", "semester", "examination", "result", "fee"]):
        return "Student Services"
    if any(word in text for word in ["scholarship", "financial"]):
        return "Scholarships"
    if any(word in text for word in ["artificial intelligence", "machine learning", "deep learning", "natural language", "sentence transformer", "semantic similarity"]):
        return "AI & Machine Learning"
    if any(word in text for word in ["chatbot", "project", "technical support"]):
        return "Chatbot & Support"
    return "General"

faq_categories = [get_faq_category(question) for question in questions]


# ==================================================
# 4. LOAD SMART NLP MODEL
# ==================================================

@st.cache_resource
def load_nlp_model():
    # Downloads the model the first time the app runs.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def create_faq_embeddings(question_list):
    model = load_nlp_model()

    embeddings = model.encode(
        list(question_list),
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


try:
    with st.spinner("Loading Smart NLP model..."):
        nlp_model = load_nlp_model()
        faq_embeddings = create_faq_embeddings(tuple(questions))
except Exception as error:
    st.error(
        "The Smart NLP model could not be loaded. "
        "Check your internet connection and package installation, then restart the app."
    )
    st.exception(error)
    st.stop()


# ==================================================
# 5. SPELLING CORRECTION + SMART FAQ MATCHING
# ==================================================

# Common typing mistakes and short forms that are especially
# useful for the FAQ topics in this project.
COMMON_CORRECTIONS = {
    "wht": "what",
    "whats": "what",
    "machne": "machine",
    "machin": "machine",
    "lernng": "learning",
    "lerning": "learning",
    "learnng": "learning",
    "artifical": "artificial",
    "inteligence": "intelligence",
    "intellgence": "intelligence",
    "explane": "explain",
    "explain": "explain",
    "sentense": "sentence",
    "transfomer": "transformer",
    "tranformer": "transformer",
    "chatbot": "chatbot",
}

# Build a vocabulary from the FAQ questions and answers.
# This lets the app correct likely typos toward words it knows.
FAQ_VOCABULARY = set()
for faq_text in questions + answers:
    FAQ_VOCABULARY.update(
        word.lower()
        for word in re.findall(r"[a-zA-Z]{3,}", faq_text)
    )

FAQ_VOCABULARY.update({
    "what", "explain", "machine", "learning", "artificial",
    "intelligence", "deep", "natural", "language", "processing",
    "sentence", "transformer", "scholarship", "admission",
    "examination", "internship", "project", "python",
})


def correct_spelling(user_question):
    """Correct only likely misspellings using a small FAQ vocabulary."""
    corrected_words = []

    for word in re.findall(r"[A-Za-z]+|[^A-Za-z]+", user_question):
        if not word.isalpha():
            corrected_words.append(word)
            continue

        lower_word = word.lower()

        if lower_word in COMMON_CORRECTIONS:
            replacement = COMMON_CORRECTIONS[lower_word]
        elif len(lower_word) >= 4 and lower_word not in FAQ_VOCABULARY:
            matches = difflib.get_close_matches(
                lower_word,
                FAQ_VOCABULARY,
                n=1,
                cutoff=0.78
            )
            replacement = matches[0] if matches else lower_word
        else:
            replacement = lower_word

        # Preserve initial capitalization where practical.
        if word[:1].isupper():
            replacement = replacement.capitalize()

        corrected_words.append(replacement)

    return "".join(corrected_words).strip()


def get_answer(user_question):
    corrected_question = correct_spelling(user_question)

    user_embedding = nlp_model.encode(
        corrected_question,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Since embeddings are normalized, their dot product is cosine similarity.
    similarities = (faq_embeddings @ user_embedding).cpu().numpy()

    best_index = int(similarities.argmax())
    confidence = float(similarities[best_index])

    # Reject weak matches rather than returning an unrelated FAQ answer.
    if confidence < 0.40:
        answer = (
            "Sorry! I couldn't find a sufficiently relevant answer "
            "in my FAQ database. Please try rephrasing your question "
            "or ask about one of the suggested topics."
        )
    else:
        answer = answers[best_index]

    return answer, confidence, corrected_question


# ==================================================
# 6. SPLIT MULTIPLE QUESTIONS
# ==================================================

def split_questions(text):
    # Split at question marks, exclamation marks, new lines and semicolons.
    parts = re.split(r'[?!]+|[\n;]+', text)

    result = []

    for part in parts:
        cleaned = part.strip()

        # Remove common numbering and bullet symbols.
        cleaned = re.sub(
            r'^\s*(?:\d+[.)]\s*|[-*•]\s*)',
            '',
            cleaned
        ).strip()

        # Remove punctuation-only fragments such as a lone quote or symbol.
        cleaned = cleaned.strip()
        if re.search(r"[A-Za-z0-9]", cleaned):
            result.append(cleaned)

    return result


# ==================================================
# 7. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==================================================
# 8. PROCESS MULTIPLE QUESTIONS
# ==================================================

def process_question(user_text):
    question_list = split_questions(user_text)

    if not question_list:
        return

    for question in question_list:
        # Save user question.
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Find the closest FAQ using semantic meaning.
        answer, confidence, corrected_question = get_answer(question)

        # Save assistant answer.
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "original_question": question,
            "corrected_question": corrected_question
        })


# ==================================================
# 9. SIDEBAR
# ==================================================

with st.sidebar:
    st.title("🤖 AI Assistant")
    st.caption("Intelligent FAQ Support System")
    st.divider()

    st.subheader("💡 Suggested Questions")

    suggestions = [
        "What is this chatbot?",
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is a scholarship?",
        "How can I apply for admission?"
    ]

    for index, question in enumerate(suggestions):
        if st.button(question, key=f"suggestion_{index}"):
            st.session_state.pending_question = question
            st.rerun()

    st.divider()
    st.subheader("📚 Browse FAQ Categories")
    category_options = ["All categories"] + sorted(set(faq_categories))
    selected_category = st.selectbox(
        "Choose a category",
        category_options,
        key="faq_category_filter"
    )

    visible_faqs = [
        (question, category)
        for question, category in zip(questions, faq_categories)
        if selected_category == "All categories" or category == selected_category
    ]

    st.caption(f"{len(visible_faqs)} questions available")
    for index, (question, category) in enumerate(visible_faqs):
        if st.button(question, key=f"faq_browse_{selected_category}_{index}"):
            st.session_state.pending_question = question
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption("Powered by Python, Streamlit and Sentence Transformers")


# ==================================================
# 10. MAIN DASHBOARD
# ==================================================

st.markdown(
    '<div class="main-title">AI FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent assistant for instant answers.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# Dashboard metrics.
total_questions = sum(
    1 for message in st.session_state.messages
    if message["role"] == "user"
)
assistant_messages = [
    message for message in st.session_state.messages
    if message["role"] == "assistant"
]
rejected_questions = sum(
    1 for message in assistant_messages
    if "couldn't find a sufficiently relevant answer" in message["content"].lower()
)
matched_questions = len(assistant_messages) - rejected_questions
match_rate = (matched_questions / len(assistant_messages) * 100) if assistant_messages else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Available FAQs", len(faqs))

with col2:
    st.metric("Questions Asked", total_questions)

with col3:
    st.metric("Matched Answers", matched_questions)

with col4:
    st.metric("Not Matched", rejected_questions)

with st.expander("📊 FAQ Dashboard Overview", expanded=False):
    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:
        st.metric("Answer match rate", f"{match_rate:.1f}%")
        st.caption("Based on questions answered in this session.")

    with overview_col2:
        category_counts = {}
        for category in faq_categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        st.write("**FAQ categories**")
        for category, count in sorted(category_counts.items()):
            st.write(f"- {category}: {count}")

st.caption("AI Engine: Smart NLP · Sentence Transformers")
st.divider()


# ==================================================
# 11. WELCOME MESSAGE
# ==================================================

if not st.session_state.messages:
    st.markdown("### 👋 Welcome!")

    st.write(
        "Hello! I am your AI FAQ assistant. "
        "I can understand different ways of asking similar questions. "
        "You can ask one question or multiple questions in a single message."
    )

    st.info(
        "Tip: Separate multiple questions using question marks, "
        "new lines or semicolons."
    )


# ==================================================
# 12. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            confidence = message.get("confidence", 0)

            corrected_question = message.get("corrected_question", "")
            if corrected_question and corrected_question.lower() != message.get("original_question", "").lower():
                st.caption(f"Interpreted as: {corrected_question}")
            st.caption(f"Semantic similarity: {confidence:.1%}")


# ==================================================
# 13. PROCESS SUGGESTED QUESTION
# ==================================================

if st.session_state.pending_question:
    selected_question = st.session_state.pending_question
    st.session_state.pending_question = None

    process_question(selected_question)
    st.rerun()


# ==================================================
# 14. CHAT INPUT
# ==================================================

user_input = st.chat_input("Ask one or multiple questions...")

if user_input:
    process_question(user_input)
    st.rerun()


# ==================================================
# 15. DOWNLOAD CHAT HISTORY
# ==================================================

if st.session_state.messages:
    chat_text = ""

    for message in st.session_state.messages:
        role = message["role"].upper()
        chat_text += f"{role}: {message['content']}\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="FAQ_Chat_History.txt",
        mime="text/plain"
    )


# ==================================================
# 16. FOOTER
# ==================================================

st.divider()

st.caption(
    "CodeAlpha AI Internship Project | "
    "AI FAQ Chatbot | Python + Streamlit + Smart NLP"
)
