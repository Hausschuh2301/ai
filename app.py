import os
import json
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# -----------------------------
# Konfiguration
# -----------------------------
MODEL_NAME = "llama3.2"  # Lokales Ollama Model
DATA_FOLDER = "data"      # Ordner mit Dokumenten
DB_DIR = "db"             # Persistente DB
CONFIG_FILE = "config.json"

# -----------------------------
# Persönliche KI Klasse
# -----------------------------
class PersonalAI:
    def __init__(self, model_name, data_folder, db_dir, config_file):
        self.model_name = model_name
        self.data_folder = data_folder
        self.db_dir = db_dir
        self.config_file = config_file

        # Lade Konfiguration
        self.config = self._load_config()

        # LLM & Embeddings initialisieren
        print("[INFO] Ollama LLM initialisieren...")
        self.llm = OllamaLLM(model=self.model_name)
        print("[INFO] Ollama Embeddings initialisieren...")
        self.emb = OllamaEmbeddings(model=self.model_name)

        # Chroma DB initialisieren
        print("[INFO] Chroma DB aufbauen/öffnen...")
        self.db = Chroma(persist_directory=self.db_dir, embedding_function=self.emb)
        self._ensure_index()

        # Retrieval Chain
        print("[INFO] Retrieval Chain initialisieren...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": self.config.get("k", 3)}),
        )

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Default config
            return {"rules": [], "k": 3}

    def _ensure_index(self):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        loader = DirectoryLoader(self.data_folder)
        docs = loader.load()
        if len(docs) > 0:
            print(f"[INFO] Indexiere {len(docs)} Dokumente...")
            self.db.add_documents(docs)
        else:
            print("[WARN] Keine Dokumente zum Indexieren gefunden.")

    def ask(self, question: str):
        """Frage an die KI stellen."""
        # invoke statt run benutzen; Key muss 'query' heißen
        response = self.qa_chain.invoke({"query": question})
        return response.get("result", "I don't know")


# -----------------------------
# Main Funktion
# -----------------------------
def main():
    ai = PersonalAI(model_name=MODEL_NAME, data_folder=DATA_FOLDER, db_dir=DB_DIR, config_file=CONFIG_FILE)
    print("Deine persönliche KI läuft. Tippe 'exit' zum Beenden.")

    while True:
        question = input("Deine Frage: ").strip()
        if question.lower() == "exit":
            break
        answer = ai.ask(question)
        print("Antwort:", answer)

if __name__ == "__main__":
    main()
