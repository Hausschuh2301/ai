# KI mit Python

Dieses Projekt ist eine **KI**, die:
- Dokumente aus einem Ordner indexiert,
- Fragen beantwortet,
- und optional das Internet über Google Search (SerpAPI) einbezieht.

---

## Features
- Lokale Dokumenten-Indexierung mit **Chroma DB**
- Nutzung von **Ollama LLM** für Antworten
- Google Search Integration via **SerpAPI** (optional)
- Konfigurierbare Regeln, z.B. verbotene Inhalte

---

## Voraussetzungen
- Python 3.11+
- Lokales Ollama Model (`llama3.2` empfohlen)
- SerpAPI Key (wenn Internetabfragen genutzt werden sollen)

---

## Installation

### 1️⃣ Repository klonen
```bash
git clone https://github.com/dein-user/KI.git
cd KI
```
### 2️⃣ Virtuelle Umgebung erstellen
Code kopieren
```bash
python -m venv .venv
```
### 3️⃣ Virtuelle Umgebung aktivieren
#### Windows (PowerShell)
```
.venv\Scripts\activate.bat
```
#### Linux / macOS
```bash
source .venv/bin/activate
```
### 4️⃣ Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Konfiguration
###### config.json erstellen (falls nicht vorhanden):

```json
{
  "rules": [
    {
      "id": "no_kill",
      "description": "Töte keine Menschen",
      "pattern": ["töte", "morde", "ermorde", "kill", "getöte"]
    },
    {
      "id": "no_illicit_drugs",
      "description": "Keine Anleitung zur Herstellung illegaler Drogen",
      "pattern": ["drogen herstellen", "herstellung von drogen", "meth herstellen"]
    }
  ],
  "k": 3,
  "serpapi_key": "DEIN_SERPAPI_KEY_HIER"
}
```
+ Dokumente, die die KI nutzen soll, in den Ordner data/ legen.

## Nutzung
app.py starten:

```bash
python app.py
```
+ Fragen direkt im Terminal stellen.

+ Zum Beenden einfach ```exit``` eingeben.

## Hinweise
- Ollama Model: Stelle sicher, dass das Model lokal vorhanden ist (llama3.2).

- SerpAPI: Für Internetabfragen musst du einen kostenlosen API-Key von https://serpapi.com eintragen.

- Regeln: Inhalte wie „Menschen töten“ oder „Drogen herstellen“ werden automatisch blockiert.

- Daten: Nutze nur eigene oder lizenzfreie Dokumente.