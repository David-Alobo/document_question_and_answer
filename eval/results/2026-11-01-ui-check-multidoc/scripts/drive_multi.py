"""Drive the app with two PDFs: one file, then add a second, compare, ask about one paper.

Usage: python drive_multi.py <app-url> <pdf> <out-dir> <second-pdf>
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
ws = websocket.create_connection(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
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


def mouse_click(element_js):
    """Click the centre of the element an expression returns, with real mouse events."""
    rect = js(f"(() => {{ const r = ({element_js}).getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()")
    for kind in ("mousePressed", "mouseReleased"):
        cdp("Input.dispatchMouseEvent", type=kind, x=rect[0], y=rect[1], button="left", clickCount=1)


def ask(text):
    wait_for("!!document.querySelector('textarea') && !document.querySelector('[data-testid=stStatusWidget]')",
             label="chat_box")
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
    PDF2 = str(Path(sys.argv[4]).resolve())
    cdp("Page.enable")
    cdp("DOM.enable")
    cdp("Page.navigate", url=URL)
    wait_for("!!document.querySelector('input[type=file]')", label="uploader")

    def upload(path):
        root = cdp("DOM.getDocument", depth=-1)["root"]["nodeId"]
        node = cdp("DOM.querySelector", nodeId=root, selector="input[type=file]")["nodeId"]
        cdp("DOM.setFileInputFiles", nodeId=node, files=[path])

    upload(PDF)
    wait_for("document.body.innerText.includes('Document ready')", label="first_file")
    ask("What NER models were evaluated?")
    wait_for(f"{count('📚 Sources')} >= 1", label="answer1")
    shot("01_one_file")

    t0 = time.time()
    upload(PDF2)
    wait_for("document.body.innerText.includes('2 documents ready')", label="second_file")
    print(f"second file ready in {time.time() - t0:.1f}s", flush=True)
    shot("02_two_files")

    js("[...document.querySelectorAll('button')].find(b => b.innerText.includes('Compare these documents')).click()")
    wait_for(f"{count('📚 Sources')} >= 2", label="compare")
    js("[...document.querySelectorAll('summary')].filter(s => s.innerText.includes('Sources')).pop().click()")
    time.sleep(1)
    shot("03_compare")

    # Ask about one paper only: open the "Ask about" select and pick the file
    js("[...document.querySelectorAll('[data-testid=stSelectbox]')].find(b => b.innerText.includes('Ask about')).scrollIntoView({block: 'center'})")
    time.sleep(0.5)
    mouse_click("[...document.querySelectorAll('[data-testid=stSelectbox]')].find(b => b.innerText.includes('Ask about')).querySelector('input') || [...document.querySelectorAll('[data-testid=stSelectbox]')].find(b => b.innerText.includes('Ask about')).querySelector('[role=combobox]')")
    wait_for("document.querySelectorAll('[role=option]').length > 0", timeout=20, label="options_open")
    time.sleep(0.5)
    print("options:", js("[...document.querySelectorAll('[role=option]')].map(o => o.textContent)"), flush=True)
    mouse_click("[...document.querySelectorAll('[role=option]')].find(o => o.textContent.includes('1912.01214'))")
    wait_for("[...document.querySelectorAll('[data-testid=stSelectbox]')].find(b => b.innerText.includes('Ask about')).innerText.includes('1912.01214.pdf')",
             timeout=30, label="scope_chosen")
    ask("Which datasets were used for zero-shot translation?")
    wait_for(f"{count('📚 Sources')} >= 3", label="single_paper_question")
    js("[...document.querySelectorAll('summary')].filter(s => s.innerText.includes('Sources')).pop().click()")
    time.sleep(1)
    shot("04_question_about_one_paper")
    (OUT / "page_text.txt").write_text(js("document.body.innerText"), encoding="utf-8")
finally:
    proc.kill()
