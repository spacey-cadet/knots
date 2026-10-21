# Knots MCP server + Alexa endpoint. Runs anywhere Docker runs (EC2, App Runner, Lightsail, your laptop).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY knots ./knots
COPY server.py .

# Persistent state (memory database, event log) lives in /data: mount a volume there.
RUN useradd --system --uid 10001 knots && mkdir /data && chown knots /data
USER knots
ENV KNOTS_HOST=0.0.0.0 PORT=8000 \
    KNOTS_PROFILE_DB=/data/knots_profile.db KNOTS_EVENT_LOG=/data/events.jsonl
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','8000'),timeout=4)" || exit 1

CMD ["python", "server.py"]
