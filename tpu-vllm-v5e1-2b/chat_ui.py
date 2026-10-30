#!/usr/bin/env python3
"""Minimal browser chat UI for this rig's vLLM endpoint. Standard library only.

Serves one page on localhost and forwards /v1/* to vLLM, so the browser never makes a
cross-origin request (vLLM rejects the CORS preflight).

    python3 chat_ui.py --upstream http://<tpu-ip>:8000 [--port 8765]
"""
import argparse
import http.server
import json
import urllib.error
import urllib.request

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma on TPU</title>
<style>
:root{--bg:#fff;--fg:#1f1f1f;--muted:#5f6368;--line:#dadce0;--me:#e8f0fe;--accent:#1a73e8}
@media (prefers-color-scheme:dark){:root{--bg:#1f1f1f;--fg:#e8eaed;--muted:#9aa0a6;--line:#3c4043;--me:#283142;--accent:#8ab4f8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
header b{font-size:18px}header span{color:var(--muted);font-size:14px}
#log{flex:1;overflow-y:auto;padding:16px;max-width:900px;width:100%;margin:0 auto}
.msg{margin:0 0 14px;padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.user{background:var(--me);margin-left:15%}.assistant{border:1px solid var(--line);margin-right:15%}
.think{color:var(--muted);font-size:14px;border-left:3px solid var(--line);padding-left:10px;margin-bottom:8px}
.stats{color:var(--muted);font-size:13px;margin-top:6px}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);max-width:900px;width:100%;margin:0 auto}
textarea{flex:1;resize:none;font:inherit;padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:0 18px;border:0;border-radius:10px;background:var(--accent);color:var(--bg);cursor:pointer}
button:disabled{opacity:.5}
</style></head><body>
<header><b>Gemma on TPU</b><span id="model">connecting…</span></header>
<div id="log"></div>
<form id="f"><textarea id="q" rows="2" placeholder="Ask something… (Enter to send, Shift+Enter for newline)"></textarea><button id="b">Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b');
let model=null;const history=[];
fetch('/v1/models').then(r=>r.json()).then(j=>{model=j.data[0].id;document.getElementById('model').textContent=model+' · vLLM on Cloud TPU'})
 .catch(e=>document.getElementById('model').textContent='endpoint unreachable: '+e);
function add(cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
document.getElementById('f').addEventListener('submit',async e=>{
 e.preventDefault();const text=q.value.trim();if(!text||!model)return;
 q.value='';b.disabled=true;add('user',text);history.push({role:'user',content:text});
 const box=add('assistant','');const think=document.createElement('div');think.className='think';think.hidden=true;
 const body=document.createElement('div');const stats=document.createElement('div');stats.className='stats';
 box.append(think,body,stats);
 const t0=performance.now();let first=null,answer='',chunks=0;
 try{
  const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
   body:JSON.stringify({model,messages:history,stream:true,stream_options:{include_usage:true}})});
  if(!r.ok)throw new Error(r.status+' '+await r.text());
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='',usage=null;
  for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
   let i;while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+1);
    if(!line.startsWith('data:'))continue;const data=line.slice(5).trim();if(data==='[DONE]')continue;
    const j=JSON.parse(data);if(j.usage)usage=j.usage;const d=j.choices&&j.choices[0]&&j.choices[0].delta;if(!d)continue;
    const rc=d.reasoning_content||d.reasoning;if(rc){think.hidden=false;think.textContent+=rc}
    if(d.content){if(first===null)first=performance.now();answer+=d.content;body.textContent=answer;chunks++}
    log.scrollTop=log.scrollHeight}}
  const secs=(performance.now()-t0)/1000;const toks=usage?usage.completion_tokens:chunks;
  stats.textContent=`first token ${first?((first-t0)/1000).toFixed(2):'–'} s · ${toks} tokens in ${secs.toFixed(2)} s`+(usage?` · ${(toks/secs).toFixed(1)} tok/s`:'');
  history.push({role:'assistant',content:answer});
 }catch(err){body.textContent='Error: '+err.message;history.pop()}
 b.disabled=false;q.focus();
});
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    upstream = ""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
            msg = json.dumps({"error": f"upstream {self.upstream} unreachable: {e}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(msg)
            return
        # HTTP/1.0 response with no Content-Length: the body ends when the connection
        # closes, which lets the event stream pass through chunk by chunk.
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        self.end_headers()
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--upstream", required=True, help="vLLM base URL, e.g. http://34.16.207.135:8000")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    Handler.upstream = a.upstream.rstrip("/")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Chat UI on http://localhost:{a.port} -> {Handler.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
