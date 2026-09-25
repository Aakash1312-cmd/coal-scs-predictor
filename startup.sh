#!/bin/bash
TARGET_PORT="${WEBSITES_PORT:-8000}"
python -m streamlit run app.py --server.port "${TARGET_PORT}" --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
