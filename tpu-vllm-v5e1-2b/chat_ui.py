#!/usr/bin/env python3
"""Browser chat UI for this rig's vLLM endpoint. Standard library only.

Serves one page on localhost and forwards /v1/* to vLLM, so the browser never makes a
cross-origin request (vLLM rejects the CORS preflight). Two more routes feed the page:
/ui/status (is vLLM's /health answering) and /ui/info (what is running on the TPU, from
vLLM's /version, /v1/models and /metrics plus this rig's tpu.env).

    python3 chat_ui.py --upstream http://<tpu-ip>:8000 [--port 8765]
"""
import argparse
import http.server
import json
import os
import re
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma on vLLM TPU</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #667085; --line: #e3e6eb;
    --accent: #2f6fed; --accent-text: #ffffff; --user: #e8effd; --bot: #ffffff; --code: #f1f3f6;
    --ok: #1f9d55; --bad: #d64545; --warn: #c58a00;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12151b; --panel: #1a1e26; --text: #e6e8ec; --muted: #98a2b3; --line: #2a303b;
      --accent: #5b8cff; --accent-text: #0b0d12; --user: #1f2a44; --bot: #1a1e26; --code: #232833;
      --ok: #3ccf7a; --bad: #ff6b6b; --warn: #f0b429;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .app { display: grid; grid-template-columns: 1fr 300px; height: 100vh; }
  @media (max-width: 860px) { .app { grid-template-columns: 1fr; } aside { display: none; } }
  main { display: flex; flex-direction: column; min-width: 0; }
  header { display: flex; align-items: center; gap: 12px; padding: 12px 20px;
           border-bottom: 1px solid var(--line); background: var(--panel); }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 13px; }
  .badge { margin-left: auto; display: inline-flex; align-items: center; gap: 6px; font-size: 13px;
           padding: 4px 10px; border: 1px solid var(--line); border-radius: 999px; white-space: nowrap; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
  .dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); }
  #log { flex: 1; overflow-y: auto; padding: 24px 20px; display: flex; flex-direction: column; gap: 16px; }
  .empty { margin: auto; text-align: center; color: var(--muted); max-width: 520px; }
  .empty h2 { color: var(--text); font-size: 20px; margin: 0 0 8px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
  .chip { border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 999px;
          padding: 6px 12px; cursor: pointer; font: inherit; font-size: 13px; }
  .chip:hover { border-color: var(--accent); }
  .msg { max-width: 820px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 4px; }
  .bubble { padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line); overflow-wrap: anywhere; }
  .msg.user .bubble { background: var(--user); border-color: transparent; align-self: flex-end; max-width: 85%; }
  .msg.bot .bubble { background: var(--bot); }
  .who { font-size: 12px; color: var(--muted); }
  .msg.user .who { align-self: flex-end; }
  .stats { font-size: 12px; color: var(--muted); }
  details.think { color: var(--muted); font-size: 13px; border-left: 3px solid var(--line);
                  padding-left: 10px; margin-bottom: 8px; white-space: pre-wrap; }
  details.think summary { cursor: pointer; }
  .bubble p { margin: 0 0 8px; } .bubble p:last-child { margin-bottom: 0; }
  .bubble pre { background: var(--code); padding: 10px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
  .bubble code { background: var(--code); padding: 1px 4px; border-radius: 4px; font-size: 13px; }
  .bubble pre code { padding: 0; background: none; }
  .bubble ul, .bubble ol { margin: 4px 0 8px 20px; padding: 0; }
  .cursor .answer::after { content: "▍"; animation: blink 1s steps(1) infinite; color: var(--accent); }
  @keyframes blink { 50% { opacity: 0; } }
  form { border-top: 1px solid var(--line); background: var(--panel); padding: 12px 20px; }
  .row { max-width: 820px; margin: 0 auto; display: flex; gap: 8px; align-items: flex-end; }
  textarea { flex: 1; resize: none; min-height: 44px; max-height: 200px; padding: 11px 12px; border-radius: 10px;
             border: 1px solid var(--line); background: var(--bg); color: var(--text); font: inherit; }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button.primary { background: var(--accent); color: var(--accent-text); border: 0; border-radius: 10px;
                   padding: 0 18px; height: 44px; font: inherit; font-weight: 600; cursor: pointer; }
  button.ghost { background: none; border: 1px solid var(--line); color: var(--text); border-radius: 10px;
                 height: 44px; padding: 0 12px; font: inherit; cursor: pointer; }
  .hint { max-width: 820px; margin: 6px auto 0; font-size: 12px; color: var(--muted); }
  aside { border-left: 1px solid var(--line); background: var(--panel); padding: 16px; overflow-y: auto; font-size: 13px; }
  aside h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 18px 0 8px; }
  aside h3:first-child { margin-top: 0; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 0; }
  dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
  label.ctl { display: block; margin: 8px 0 4px; color: var(--muted); }
  input[type=range] { width: 100%; }
  aside textarea { width: 100%; min-height: 70px; font-size: 13px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<div class="app">
  <main>
    <header>
      <div>
        <h1 id="title">Gemma 4 · vLLM on Cloud TPU</h1>
        <div class="sub" id="sub">connecting…</div>
      </div>
      <span class="badge"><span class="dot" id="dot"></span><span id="badgeText">checking…</span></span>
    </header>
    <div id="log">
      <div class="empty" id="empty">
        <h2>Ask Gemma on the TPU</h2>
        <div>Replies stream from vLLM on a Cloud TPU, through this machine's local proxy.</div>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <form id="form">
      <div class="row">
        <textarea id="input" rows="1" placeholder="Message Gemma…  (Enter to send, Shift+Enter for a new line)"></textarea>
        <button type="button" class="ghost" id="clear" title="Start a new conversation">New</button>
        <button type="submit" class="primary" id="send">Send</button>
      </div>
      <div class="hint">The conversation is sent with each message · Stop cancels the request on the server</div>
    </form>
  </main>
  <aside>
    <h3>Settings</h3>
    <label class="ctl" for="maxTokens">Max tokens: <span id="maxTokensVal">512</span></label>
    <input type="range" id="maxTokens" min="16" max="2048" step="16" value="512">
    <label class="ctl" for="temp">Temperature: <span id="tempVal">server default</span></label>
    <input type="range" id="temp" min="-0.1" max="1.5" step="0.1" value="-0.1">
    <label class="ctl" for="topk">Top-k: <span id="topkVal">off</span></label>
    <input type="range" id="topk" min="0" max="128" step="8" value="0">
    <div class="note">Top-k costs about 14 ms per token on this TPU build of vLLM (45 against 121 tok/s at top-k 64).</div>
    <label class="ctl" for="system">System prompt (optional)</label>
    <textarea id="system" placeholder="e.g. Answer in one short paragraph."></textarea>
    <h3>Last reply</h3>
    <dl id="last"><dt>—</dt><dd>nothing yet</dd></dl>
    <h3>On the TPU</h3>
    <dl id="engine"><dt>—</dt><dd>loading…</dd></dl>
    <h3>Server load</h3>
    <dl id="load"><dt>—</dt><dd>loading…</dd></dl>
    <div class="note">Times are measured in this browser, so they include the network to the TPU.</div>
  </aside>
</div>
<script>
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send");
let history = [], model = null, controller = null;

const examples = [
  "Why is the sky blue? One paragraph.",
  "Write a haiku about TPUs.",
  "Explain a key-value cache in a transformer, simply.",
  "Give me three fun facts about octopuses.",
];
for (const ex of examples) {
  const b = document.createElement("button");
  b.className = "chip"; b.type = "button"; b.textContent = ex;
  b.onclick = () => { input.value = ex; submit(); };
  $("chips").appendChild(b);
}

function esc(s) { return s.replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>")
               .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
               .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}
function render(md) {
  const out = []; const lines = md.split("\n"); let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("```")) {
      const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++; out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>"); continue;
    }
    if (/^\s*([-*•]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l); const items = [];
      while (i < lines.length && /^\s*([-*•]|\d+\.)\s+/.test(lines[i]))
        items.push("<li>" + inline(lines[i++].replace(/^\s*([-*•]|\d+\.)\s+/, "")) + "</li>");
      out.push((ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>")); continue;
    }
    if (/^#{1,6}\s/.test(l)) { out.push("<p><strong>" + inline(l.replace(/^#+\s/, "")) + "</strong></p>"); i++; continue; }
    if (l.trim() === "") { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```") && !/^\s*([-*•]|\d+\.)\s+/.test(lines[i]))
      para.push(inline(lines[i++]));
    out.push("<p>" + para.join("<br>") + "</p>");
  }
  return out.join("");
}

function addMsg(role, text) {
  $("empty")?.remove();
  const m = document.createElement("div"); m.className = "msg " + role;
  m.innerHTML = `<div class="who">${role === "user" ? "You" : "Gemma · TPU"}</div><div class="bubble"></div><div class="stats"></div>`;
  const bubble = m.querySelector(".bubble");
  if (role === "user") bubble.innerHTML = render(text);
  else bubble.innerHTML = `<details class="think" hidden><summary>Reasoning</summary><div></div></details><div class="answer"></div>`;
  log.appendChild(m); log.scrollTop = log.scrollHeight;
  return m;
}

function setDL(id, rows) {
  rows = rows.filter(([, v]) => v !== undefined && v !== null && v !== "");
  $(id).innerHTML = rows.length ? rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("")
                                : "<dt>—</dt><dd>unavailable</dd>";
}

async function status() {
  try {
    const s = await (await fetch("/ui/status")).json();
    const ok = s.health === 200;
    $("dot").className = "dot " + (ok ? "ok" : "bad");
    $("badgeText").textContent = ok ? "TPU online" : "TPU not answering";
    return ok;
  } catch { $("dot").className = "dot bad"; $("badgeText").textContent = "proxy not answering"; return false; }
}

const fmtPct = (x) => x == null ? null : (100 * x).toFixed(1) + " %";
async function info() {
  try {
    const e = await (await fetch("/ui/info")).json();
    const m = e.model || {}, r = e.rig || {}, c = e.cache || {}, l = e.load || {};
    model = m.id || model;
    $("sub").textContent = [r.ACCELERATOR_TYPE, model, e.version && "vLLM " + e.version].filter(Boolean).join(" · ");
    setDL("engine", [
      ["model", (m.id || "").split("/").pop()],
      ["weights", m.root && m.root !== m.id ? m.root : null],
      ["context", m.max_model_len && m.max_model_len + " tokens"],
      ["engine", e.version && "vLLM " + e.version],
      ["chip", r.ACCELERATOR_TYPE],
      ["tensor parallel", r.TENSOR_PARALLEL_SIZE],
      ["zone", r.GOOGLE_CLOUD_ZONE],
      ["KV blocks", c.num_gpu_blocks && c.block_size ? `${c.num_gpu_blocks} × ${c.block_size} tokens` : null],
      ["KV dtype", c.cache_dtype],
      ["HBM share", c.gpu_memory_utilization],
      ["prefix cache", c.enable_prefix_caching],
      ["upstream", e.upstream],
    ]);
    setDL("load", [
      ["running", l.running], ["waiting", l.waiting], ["KV cache used", fmtPct(l.kv_cache_usage)],
      ["prefix hit rate", l.prefix_queries ? fmtPct(l.prefix_hits / l.prefix_queries) : null],
      ["prompt tokens", l.prompt_tokens], ["generated tokens", l.generation_tokens],
    ]);
  } catch { setDL("engine", []); setDL("load", []); }
}

async function submit() {
  const text = input.value.trim();
  if (!text || controller) return;
  if (!model) await info();
  input.value = ""; autosize();
  history.push({ role: "user", content: text });
  addMsg("user", text);
  const bot = addMsg("bot", ""); const bubble = bot.querySelector(".bubble");
  const think = bubble.querySelector(".think"), thinkBody = think.querySelector("div"), answer = bubble.querySelector(".answer");
  bubble.classList.add("cursor");
  send.textContent = "Stop"; controller = new AbortController();
  const sys = $("system").value.trim();
  const messages = (sys ? [{ role: "system", content: sys }] : []).concat(history);
  const body = { model, messages, stream: true, max_tokens: +$("maxTokens").value, stream_options: { include_usage: true } };
  if (+$("temp").value >= 0) body.temperature = +$("temp").value;
  body.top_k = +$("topk").value || -1;
  const t0 = performance.now(); let ttft = null, reply = "", reasoning = "", usage = null, finish = null;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" }, signal: controller.signal,
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim(); if (data === "[DONE]") continue;
        const j = JSON.parse(data);
        if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
        if (j.usage) usage = j.usage;
        const c = (j.choices || [])[0] || {};
        const d = c.delta || {};
        const rc = d.reasoning_content || d.reasoning;
        if ((rc || d.content) && ttft === null) ttft = (performance.now() - t0) / 1000;
        if (rc) { reasoning += rc; think.hidden = false; thinkBody.textContent = reasoning; }
        if (d.content) { reply += d.content; answer.innerHTML = render(reply); }
        if (c.finish_reason) finish = c.finish_reason;
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) {
    if (err.name === "AbortError") finish = "stopped";
    else answer.innerHTML = render(reply + `\n\n**Error:** ${err.message}`);
  }
  bubble.classList.remove("cursor");
  const total = (performance.now() - t0) / 1000;
  const n = usage ? usage.completion_tokens : null;
  const rate = n && n > 1 && ttft !== null && total > ttft ? (n - 1) / (total - ttft) : null;
  bot.querySelector(".stats").textContent =
    [ttft !== null ? `first token ${ttft.toFixed(2)} s` : null, rate ? `${rate.toFixed(1)} tok/s` : null,
     n !== null ? `${n} tokens` : null, `${total.toFixed(1)} s`, finish].filter(Boolean).join(" · ");
  setDL("last", [["first token", ttft !== null ? ttft.toFixed(3) + " s" : "—"], ["decode", rate ? rate.toFixed(1) + " tok/s" : "—"],
                 ["tokens", n ?? "—"], ["prompt tokens", usage ? usage.prompt_tokens : "—"],
                 ["reasoning", reasoning ? reasoning.length + " chars" : null],
                 ["total", total.toFixed(2) + " s"], ["finish", finish ?? "—"]]);
  if (reply) history.push({ role: "assistant", content: reply }); else history.pop();
  controller = null; send.textContent = "Send"; input.focus();
  info();
}

function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });
$("form").addEventListener("submit", (e) => { e.preventDefault(); if (controller) controller.abort(); else submit(); });
$("clear").onclick = () => { if (controller) controller.abort(); location.reload(); };
$("maxTokens").oninput = (e) => { $("maxTokensVal").textContent = e.target.value; };
$("topk").oninput = (e) => { $("topkVal").textContent = +e.target.value ? e.target.value : "off"; };
$("temp").oninput =(e) => { $("tempVal").textContent = +e.target.value < 0 ? "server default" : (+e.target.value).toFixed(1); };

status().then((ok) => { if (ok) info(); });
setInterval(async () => { if (await status() && !controller) info(); }, 15000);
input.focus();
</script>
</body>
</html>"""


def read_env(path):
    """KEY=VALUE lines from the rig's tpu.env; comments and blanks skipped."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return out


METRIC = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_metrics(text):
    """Prometheus text format -> {name: [(labels, value)]}."""
    out = {}
    for line in text.splitlines():
        m = METRIC.match(line)
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        out.setdefault(m.group(1), []).append((dict(LABEL.findall(m.group(2) or "")), val))
    return out


def metric_sum(metrics, *names):
    """Sum of the first metric name present (vLLM has renamed several between releases)."""
    for n in names:
        if n in metrics:
            return sum(v for _, v in metrics[n])
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    upstream = ""
    rig = {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/ui/status":
            self._json({"upstream": self.upstream, "health": self._get(self.upstream + "/health", 5)[0]})
        elif self.path == "/ui/info":
            self._json(self._info())
        elif self.path.startswith("/v1/"):
            self._forward(None)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        self._forward(self.rfile.read(length))

    def _get(self, url, timeout=10):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, ""
        except OSError:
            return None, ""

    def _info(self):
        info = {"upstream": self.upstream, "rig": self.rig}
        code, body = self._get(self.upstream + "/v1/models")
        if code == 200:
            try:
                info["model"] = json.loads(body)["data"][0]
            except (ValueError, KeyError, IndexError):
                pass
        code, body = self._get(self.upstream + "/version")
        if code == 200:
            try:
                info["version"] = json.loads(body).get("version")
            except ValueError:
                pass
        code, body = self._get(self.upstream + "/metrics")
        if code == 200:
            m = parse_metrics(body)
            cache = m.get("vllm:cache_config_info")
            if cache:
                info["cache"] = cache[0][0]
            info["load"] = {
                "running": metric_sum(m, "vllm:num_requests_running"),
                "waiting": metric_sum(m, "vllm:num_requests_waiting"),
                "kv_cache_usage": metric_sum(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
                "prefix_hits": metric_sum(m, "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
                "prefix_queries": metric_sum(m, "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
                "prompt_tokens": metric_sum(m, "vllm:prompt_tokens_total"),
                "generation_tokens": metric_sum(m, "vllm:generation_tokens_total"),
            }
            for k, v in info["load"].items():
                if isinstance(v, float) and v.is_integer() and k != "kv_cache_usage":
                    info["load"][k] = int(v)
        return info

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _forward(self, body):
        req = urllib.request.Request(
            self.upstream + self.path,
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            resp = e
        except OSError as e:
            self._json({"error": {"message": f"upstream {self.upstream} unreachable: {e}"}}, 502)
            return
        # HTTP/1.0 response with no Content-Length: the body ends when the connection
        # closes, which lets the event stream pass through chunk by chunk. When the
        # browser stops (Stop button), the write fails and closing `resp` drops the
        # upstream connection, which makes vLLM abort the request.
        try:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--upstream", required=True, help="vLLM base URL, e.g. http://34.16.207.135:8000")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    Handler.upstream = a.upstream.rstrip("/")
    env = read_env(os.path.join(HERE, "tpu.env"))
    Handler.rig = {k: env[k] for k in ("ACCELERATOR_TYPE", "TENSOR_PARALLEL_SIZE", "GOOGLE_CLOUD_ZONE", "MODEL_NAME") if k in env}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Chat UI on http://localhost:{a.port} -> {Handler.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
