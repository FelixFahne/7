# Hinweise für Codex-Agenten

Dieses Repository enthält ein Gradio-Interface und mehrere Python-Skripte für das SLEDA-Framework. Vor Commits sollten folgende Schritte ausgeführt werden:

1. **Syntaxprüfung** – Alle Python-Dateien müssen fehlerfrei kompilieren:
   ```bash
   python -m py_compile $(git ls-files '*.py')
   ```
2. **Formattierung** – Der Code soll mit `black` formatiert werden (Version 23 oder neuer):
   ```bash
   black $(git ls-files '*.py')
   ```
3. **Lokal testen** – Falls möglich `app.py` starten und die drei Tabs "Pre-processing", "Training" und "Application" manuell durchspielen.

Neue Funktionen sollen den bestehenden Workflow beibehalten:
- Daten werden mit `src/preprocessing.py` in das Verzeichnis `SLDEA_WORKDIR` (Standard: `/tmp/space`) geschrieben.
- Modelle erstellt `src/dialogue_pred.py` als `model.pkl`.
- Vorhersagen entstehen ebenfalls in `SLDEA_WORKDIR`.

Wenn neue Abhängigkeiten benötigt werden, müssen sie `requirements.txt` und gegebenenfalls `Dockerfile` hinzugefügt werden.
