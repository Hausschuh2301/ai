import os
import json
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# Konfiguration
MODEL_NAME = "llama3.2"
DATA_FOLDER = "meine_daten"
DB_DIR = "chroma_db"
CONFIG_FILE = "config.json"

class PersonalAI:
    def __init__(self, model_name, data_folder, db_dir, config_file):
        self.model_name = model_name
        self.data_folder = data_folder
        self.db_dir = db_dir
        self.config_file = config_file

        print("[INFO] Lade Konfiguration...")
        self.load_config()

        print("[INFO] Ollama LLM initialisieren...")
        self.llm = OllamaLLM(model=self.model_name)

        print("[INFO] Ollama Embeddings initialisieren...")
        self.emb = OllamaEmbeddings(model=self.model_name)

        print("[INFO] Chroma DB aufbauen/öffnen...")
        self.db = Chroma(persist_directory=self.db_dir, embedding_function=self.emb)

        self._ensure_db()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            # Standard-Regeln
            self.config = {
                "rules": [
                    {"id": "no_kill", "description": "Töte keine Menschen",
                     "pattern": ["töte", "morde", "ermorde", "kill", "getöte"]},
                    {"id": "no_illicit_drugs", "description": "Keine Anleitung zur Herstellung illegaler Drogen",
                     "pattern": ["drogen herstellen", "herstellung von drogen", "meth herstellen"]}
                ],
                "k": 3
            }

    def _ensure_db(self):
        # Lade Dokumente aus dem Datenordner
        loader = DirectoryLoader(self.data_folder, glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        if docs:
            print(f"[INFO] Indexiere {len(docs)} Dokumente...")
            self.db.add_documents(docs)
        else:
            print("[WARN] Keine Dokumente zum Indexieren gefunden.")

        # Erstelle Retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": self.config.get("k", 3)})

        # Erstelle RetrievalQA Chain
        print("[INFO] Retrieval Chain initialisieren...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

    def ask(self, question):
        # Benutze invoke() anstelle von run()
        result = self.qa_chain.invoke({"query": question})
        antwort = result.get("result", "")
        quellen = result.get("source_documents", [])
        return antwort, quellen


def main():
    ai = PersonalAI(model_name=MODEL_NAME, data_folder=DATA_FOLDER, db_dir=DB_DIR, config_file=CONFIG_FILE)

    print("\nDeine persönliche KI läuft. Tippe 'exit' zum Beenden.\n")
    while True:
        question = input("Deine Frage: ")
        if question.lower() == "exit":
            break
        answer, sources = ai.ask(question)
        print("Antwort:", answer)
        if sources:
            print("Quellen:", [s.metadata.get("source", "Unbekannt") for s in sources])


if __name__ == "__main__":
    main()
