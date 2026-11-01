"""Stand-in for the Groq API: forwards /openai/v1/chat/completions to Ollama's OpenAI endpoint.

Usage: python groq_to_ollama.py <port> <ollama-url>
Groq model ids are mapped to the Ollama model that runs the same weights.
"""

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {"llama-3.1-8b-instant": "llama3.1:8b"}
OLLAMA = sys.argv[2].rstrip("/")
LOG = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body["model"] = MODELS.get(body["model"], body["model"])
        LOG.append(body)
        with open("proxy_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(body) + "\n")
        request = urllib.request.Request(
            f"{OLLAMA}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = response.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
