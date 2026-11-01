"""Drive the Streamlit app in headless Chrome: upload a PDF, ask, follow up, screenshot.

Usage: python drive.py <app-url> <pdf> <out-dir>
"""

import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

URL, PDF, OUT = sys.argv[1], str(Path(sys.argv[2]).resolve()), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333

proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", "--window-size=1100,1600",
                         f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"])
for _ in range(50):
    try:
        tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
        break
    except OSError:
        time.sleep(0.2)
ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"])
n = 0


def cdp(method, **params):
    global n
    n += 1
    ws.send(json.dumps({"id": n, "method": method, "params": params}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == n:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg.get("result", {})


def js(expr):
    return cdp("Runtime.evaluate", expression=expr, returnByValue=True)["result"].get("value")


def wait_for(expr, timeout=300, label=""):
    start = time.time()
    while time.time() - start < timeout:
        if js(expr):
            print(f"ok {label} after {time.time() - start:.0f}s", flush=True)
            return
        time.sleep(1)
    shot(f"timeout_{label}")
    raise TimeoutError(label)


def shot(name):
    height = js("document.querySelector('section.stMain, [data-testid=stMain]')?.scrollHeight || 1600")
    cdp("Emulation.setDeviceMetricsOverride", width=1100, height=min(int(height) + 100, 6000),
        deviceScaleFactor=1, mobile=False)
    time.sleep(0.5)
    data = cdp("Page.captureScreenshot", format="png")["data"]
    (OUT / f"{name}.png").write_bytes(base64.b64decode(data))
    print("saved", name, flush=True)


def ask(text):
    print("chat box found:", js("!!document.querySelector('textarea')"), flush=True)
    js("document.querySelector('textarea').focus()")
    cdp("Input.insertText", text=text)
    time.sleep(0.5)
    print("chat box holds:", js("document.querySelector('textarea').value"), flush=True)
    cdp("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", windowsVirtualKeyCode=13, text="\r")
    cdp("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)


def count(text):
    idle = "!document.querySelector('[data-testid=stStatusWidget]')"
    return f"({idle}) * [...document.querySelectorAll('summary')].filter(e => e.innerText.includes({json.dumps(text.split(' ', 1)[1])})).length"


try:
    cdp("Page.enable")
    cdp("DOM.enable")
    cdp("Page.navigate", url=URL)
    wait_for("!!document.querySelector('input[type=file]')", label="uploader")
    shot("01_start")
    root = cdp("DOM.getDocument", depth=-1)["root"]["nodeId"]
    node = cdp("DOM.querySelector", nodeId=root, selector="input[type=file]")["nodeId"]
    cdp("DOM.setFileInputFiles", nodeId=node, files=[PDF])
    wait_for("document.body.innerText.includes('Document ready')", label="embedded")
    shot("02_ready")

    ask("Which datasets did they experiment with?")
    wait_for(f"{count('📚 Sources')} >= 1", label="answer1")
    js("[...document.querySelectorAll('summary')].find(s => s.innerText.includes('Sources'))?.click()")
    time.sleep(1)
    shot("03_answer_with_sources")

    js("[...document.querySelectorAll('button')].find(b => b.innerText.includes('Can you elaborate on that?')).click()")
    wait_for(f"{count('📚 Sources')} >= 2", label="answer2")
    shot("04_follow_up")

    js("[...document.querySelectorAll('button')].find(b => b.innerText.includes('Clear Conversation History')).click()")
    wait_for(f"{count('📚 Sources')} == 0", timeout=60, label="cleared")
    time.sleep(3)
    shot("05_cleared")
    print("answers after clearing:", js(count("📚 Sources")))
    (OUT / "page_text.txt").write_text(js("document.body.innerText"), encoding="utf-8")
finally:
    proc.kill()
