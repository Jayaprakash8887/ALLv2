/opt/conda/bin/uvicorn main:app --host 0.0.0.0 --port 8022 --root-path /all_bot --timeout-keep-alive 600 --workers 4
tail -f /dev/null