#!/bin/bash
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app
