FROM python:3.10-slim

# 1) System-Pakete
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2) Arbeitsordner
WORKDIR /app

# 3) Python-Abhängigkeiten
COPY space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Projekt hineinkopieren (src + space + alles andere)
COPY . /app

# 5) Notebooks beim Build in Skripte konvertieren (für späteren Aufruf)
RUN python - << 'PY'
import pathlib, subprocess, sys, os
src = pathlib.Path("src")
for nb in src.rglob("*.ipynb"):
    subprocess.check_call(["jupyter", "nbconvert", "--to", "script", nb])
PY

# 6) Standard-Port für Gradio
ENV PORT 7860
EXPOSE 7860

# 7) Start-Befehl
CMD ["python", "space/app.py"]
