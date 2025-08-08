# chatbot_rag

# 🤖 Assistente RAG com Streamlit e LangChain

Este projeto é um assistente conversacional baseado em Retrieval-Augmented Generation (RAG), que utiliza arquivos PDF carregados pelo usuário para responder perguntas com base no conteúdo desses documentos.  

O backend usa LangChain para ingestão e indexação dos documentos, e OpenAI para geração das respostas.

---

## Funcionalidades

- Upload de múltiplos arquivos PDF
- Processamento e divisão dos PDFs em chunks para melhor busca
- Indexação dos textos com embeddings OpenAI via Chroma Vector Store
- Perguntas ao modelo de linguagem com contexto recuperado dos documentos
- Interface web interativa com Streamlit
- Histórico da conversa mantido na sessão do usuário
- Suporte a múltiplos modelos LLM da OpenAI (ex: gpt-3.5-turbo, gpt-4, etc)
- Respostas em português, formatadas em markdown com visualizações interativas

---

## Tecnologias usadas

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Chroma Vector Store](https://www.trychroma.com/)
- [OpenAI API](https://platform.openai.com/)
- [python-decouple](https://github.com/henriquebastos/python-decouple) para gerenciar variáveis de ambiente
- Temporários para manipulação dos PDFs

---

## Como usar

### Pré-requisitos

- Python 3.8 ou superior
- python -m venv .venv (para criar seu ambiente virtual)
- Conta e chave API da OpenAI -> crie um arquivo env e coloque sua chave lá (OPENAI_API_KEY="sua-chave-aqui")
- Instalar dependências:

```bash
pip install -r requirements.txt


