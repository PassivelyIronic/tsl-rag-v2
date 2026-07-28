#!/usr/bin/env bash
# Uruchamia API w tle i UI na pierwszym planie.
#
# `set -e` celowo NIE jest tu użyte przy starcie API: gdyby uvicorn padł,
# chcemy, żeby użytkownik zobaczył UI z komunikatem o niedostępnym backendzie,
# a nie pusty ekran po ubitym kontenerze. UI ma obsłużone ConnectError.
set -u

echo "Start API (wewnętrznie :8000)…"
uvicorn tsl_rag.api.app:create_app --factory --host 127.0.0.1 --port 8000 &
API_PID=$!

# Czekamy na gotowość API, ale bez blokowania w nieskończoność: pierwszy start
# pobiera wagi modelu embeddingów (~1.1 GB), co na zimnym Space potrafi trwać
# minuty. Po tym czasie i tak podnosimy UI — samo pokaże, że backend nie odpowiada.
for _ in $(seq 1 60); do
    if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
        echo "API gotowe."
        break
    fi
    sleep 5
done

if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "UWAGA: proces API zakończył się. UI wystartuje, ale zapytania nie przejdą." >&2
fi

echo "Start UI (:7860)…"
exec streamlit run ui.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
