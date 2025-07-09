# 🚀 create-rag-app

A CLI tool that lets you generate production-ready RAG (Retrieval-Augmented Generation) applications in seconds.

⚙️ Choose your stack. 📦 Get a complete Dockerized project. 🧠 Power your app with your data + LLMs.

## 🌟 Why `create-rag-app`?

Building RAG apps is **complex**—you need to juggle LLMs, embeddings, chunking, retrieval, APIs, and more. With `create-rag-app`, all of this becomes **plug-and-play**:

✅ Standardized architecture  
✅ Developer-first CLI  
✅ Open-source & extensible

Whether you're building an AI chatbot over your docs or a powerful knowledge assistant, this CLI sets you up with a solid, scalable foundation.

## 🔧 What It Will Do

You run:

```bash
npx create-rag-app
```

And answer a few questions:

- **LLM Provider?** (OpenAI, Claude, Local)
- **Vector DB?** (Qdrant)
- **Frontend?** (None, Streamlit, Next.js)
- **Document Loader?** (Local, Web, Notion, YouTube)
- **Chunking Strategy?** (Fixed, Recursive, Metadata-aware)
- **Embedding Model?** (OpenAI, HuggingFace, Local)
- **RAG Framework?** (LangChain, LlamaIndex, None)
- **Prompt Type?** (Basic, Conversational, Agentic)
- **Auth?** (None, JWT, Basic)
- **Extras?** (Monitoring, Eval Setup, Dataset preload)

💥 **Boom.** Your RAG app scaffold is ready inside a Docker container.

## 📁 Example Output Structure

```
my-rag-app/
├── backend/
│   ├── api/
│   ├── loaders/
│   ├── retriever/
│   ├── llm/
│   └── main.py
├── frontend/ (optional)
│   └── pages/
├── data/
├── .env.template
├── docker-compose.yml
└── README.md
```

## 🚧 Features To Be Implemented

Here's what's on the roadmap for the first MVP:

### ✅ Core CLI
- Interactive CLI with `inquirer`
- Dynamic template scaffolding
- Backend stack choices (FastAPI, Express)

### 🧠 RAG Configuration
- LLM selection (OpenAI, Anthropic, local models)
- Embedder selection (OpenAI, BGE, Cohere, etc.)
- Vector DB support (Qdrant)
- Data loader types (PDF, YouTube, Notion, Web)

### 🧱 Frontend (optional)
- Streamlit or Next.js integration
- Basic chat UI with source highlighting

### 🧪 Extras
- Prompt customization options
- Eval flow scaffold (Precision@K, feedback loop)
- Monitoring/logging (basic + OpenTelemetry)
- Auth layer (JWT / basic auth)

### 🐳 DevOps
- Dockerized full-stack output
- Git auto-init and install
- `.env` templating and secrets handling

## 🤝 Contributing

Got an idea? Want to add a new integration (e.g., Qdrant or Supabase)? We'd love to have you onboard.

```bash
git clone https://github.com/yourname/create-rag-app
cd create-rag-app
npm install
npm link
create-rag-app
```

## 🛠 Tech Stack (Planned)

- **CLI:** Node.js + TypeScript
- **CLI UI:** Inquirer.js, Chalk
- **Templates:** EJS-based folders
- **Backend:** FastAPI (Python) or Express (Node)
- **Frontend:** Next.js or Streamlit
- **DevOps:** Docker, Docker Compose
