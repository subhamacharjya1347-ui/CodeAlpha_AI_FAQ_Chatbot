# 🤖 AI FAQ Chatbot
### Intelligent Question Answering System Using Natural Language Processing

<p align="center">
  <strong>A Smart, Interactive & NLP-Powered FAQ Chatbot</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit" />
  <img src="https://img.shields.io/badge/NLP-Sentence%20Transformers-green" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 🌟 Overview

The **AI FAQ Chatbot** is an intelligent question-answering application developed using Python and Natural Language Processing (NLP).

It understands the meaning behind user questions and finds relevant answers from a predefined FAQ database. Unlike basic keyword-based chatbots, it uses semantic similarity to identify questions with similar meanings, even when users phrase them differently.

The application also supports common spelling corrections, multiple-question handling, FAQ categories, and an interactive dashboard.

## ✨ Key Features

- 🧠 **Smart NLP:** Understands the meaning of user questions.
- 🔍 **Semantic Matching:** Uses Sentence Transformers to compare question embeddings.
- ✍️ **Spelling Correction:** Handles common spelling mistakes.
- 💬 **Interactive Chat:** Provides relevant answers through a simple chat interface.
- 📚 **FAQ Categories:** Browse questions by category.
- 📊 **Dashboard:** Displays FAQ and chatbot usage statistics.
- 📝 **Chat History:** Maintains conversation history.
- 📥 **Download Chat:** Allows users to download conversations.

## 🖥️ Project Screenshots

### 1. Chatbot Dashboard
![Chatbot Dashboard](screenshots/homepage.png)

### 2. Intelligent Answer Generation
![Successful Answer](screenshots/successful_answer.png)

### 3. Spelling Correction
![Spelling Correction](screenshots/spelling_correction.png)

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Sentence Transformers | Semantic embeddings |
| Scikit-learn | Similarity calculations |
| NLP | Natural language understanding |
| JSON | FAQ data storage |

## ⚙️ How It Works

1. The user enters a question.
2. The chatbot processes the input and corrects supported common spelling mistakes.
3. Sentence Transformers converts the question into a numerical embedding.
4. The system compares it with FAQ question embeddings.
5. Cosine similarity helps identify the closest matching question.
6. The chatbot returns the corresponding answer if the match meets the configured threshold.

## 📂 Project Structure

```text
CodeAlpha_FAQ_Chatbot/
│
├── app.py
├── faqs.json
├── requirements.txt
├── README.md
├── screenshots/
│   ├── homepage.png
│   ├── successful_answer.png
│   └── spelling_correction.png
│
└── app_spelling_backup.py
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or later
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd CodeAlpha_FAQ_Chatbot
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

Open the Local URL displayed in your terminal, usually:

`http://localhost:8501`

*Note: The Sentence Transformer model may need an internet connection for its initial download.*

## 🧪 Example Questions

Try asking the chatbot:

- What is Artificial Intelligence?
- Explain Machine Learning.
- What is Deep Learning?
- Explain Natural Language Processing.
- What is a Sentence Transformer?

The chatbot can also recognize supported common spelling mistakes.

## 🔮 Future Improvements

- Voice-based question input
- Multilingual chatbot support
- Admin panel for FAQ management
- Integration with larger knowledge bases
- Improved analytics and reporting
- Cloud deployment

## 🎯 Project Objective

The objective of this project is to explore practical applications of Artificial Intelligence and NLP by developing a chatbot that can understand user queries and retrieve relevant information efficiently.

## 👨‍💻 Developer

**Subham Acharjya**

B.Tech – Computer Science & Engineering (Artificial Intelligence)

**Project:** AI FAQ Chatbot  
**Internship:** CodeAlpha – Artificial Intelligence Internship

---

<p align="center">
  ⭐ If you find this project interesting, consider giving this repository a star!
</p>

<p align="center">
  <strong>Built with Python, NLP & a passion for Artificial Intelligence.</strong>
</p>