# tpu-vllm-v5e1-2b

Gemma 4 E2B (`google/gemma-4-E2B-it`) served by vLLM on one Cloud TPU v5e chip (v5litepod-1), flex-start, us-west4-a, project `aisprint-491218`. Parameters live in `tpu.env`. `server.py` is the MCP server (`.mcp.json` registers it as `tpu-vllm-v5e1-2b`); `chat_ui.py` is the browser chat page.

#### Start it up

**1. Create the TPU.** The MCP tool `create_tpu_queued_resource` creates a 4-hour flex-start TPU. For a different limit, render the same startup script and call gcloud directly (2 hours here):

```bash
cd ~/tpu-inference/tpu-vllm-v5e1-2b
python3 -c "import asyncio, server as s; open('/tmp/vllm_startup.sh','w').write(asyncio.run(s._get_formatted_startup_script(s.MODEL_NAME, zone=s.ZONE)))"
gcloud alpha compute tpus queued-resources create tpu-vllm-v5e1-2b \
  --zone=us-west4-a --project=aisprint-491218 \
  --runtime-version=v2-alpha-tpuv5-lite --node-id=tpu-vllm-v5e1-2b-node \
  --accelerator-type=v5litepod-1 \
  --metadata-from-file=startup-script=/tmp/vllm_startup.sh \
  --provisioning-model=flex-start --max-run-duration=2h --valid-until-duration=2h \
  --labels=purpose=flex-start
```

The startup script reads the Hugging Face token from the Secret Manager secret `hf-token` at boot; `server.get_secret()` checks that it is readable.

**2. Wait for vLLM.** Get the address, then poll until `/v1/models` answers:

```bash
gcloud compute tpus tpu-vm describe tpu-vllm-v5e1-2b-node --zone us-west4-a --project aisprint-491218 \
  --format='value(networkEndpoints[0].accessConfig.externalIp)'
curl -s http://<ip>:8000/v1/models
```

**3. Open the chat page** at `http://localhost:8765`:

```bash
python3 chat_ui.py --upstream http://<ip>:8000
```

**4. Clean up** when done. Flex-start deletes the node at the run limit, but the queued resource stays listed:

```bash
gcloud alpha compute tpus queued-resources delete tpu-vllm-v5e1-2b --zone us-west4-a --project aisprint-491218 --force --quiet
```

#### Timings measured on 2026-11-03

| Step | Time |
| --- | --- |
| TPU granted → vLLM answering (includes pulling `vllm/vllm-tpu:nightly`) | 17 min (00:40:42 → 00:57:49 UTC) |
| Container restart → vLLM answering (model load and compile) | 16 min (01:43:56 → 02:00:06 UTC) |

Waiting for flex-start capacity comes on top and varies; it was not timed. Allow 20 minutes after the TPU is granted.

#### Decode speed and top-k

vLLM `0.33.0rc1.dev430`, one request, 400 tokens, v5e-1:

| Sampling | Time per token | Decode |
| --- | --- | --- |
| Temperature 0 | 8.0 ms | 125 tok/s |
| Temperature 1, top-p 0.95, no top-k | 8.3 ms | 121 tok/s |
| Temperature 1, top-k 64 | 22.1 ms | 45 tok/s |

Top-k costs about 14 ms per token on this build; streaming, `--stream-interval` and the chat parsers cost nothing measurable. Gemma 4's `generation_config.json` sets top-k 64, so vLLM starts with `--generation-config vllm` (in `startup_script_template.sh` and `_vllm_serve_flags`) and a request that sets no sampling runs at full speed. The chat page's top-k setting is off by default.

Several requests at once raise the total: 416 tok/s at 4 concurrent and 1095 tok/s at 16 (69 tok/s each), at temperature 0.

#### The chat page

`chat_ui.py` is standard-library Python. It serves the page on localhost and forwards `/v1/*` to vLLM, because vLLM rejects the browser's cross-origin preflight. The side panel has max tokens, temperature, top-k and a system prompt; the last reply's first-token time, decode rate and token counts; what is running on the TPU (model, context, vLLM version, chip, KV cache layout), from `/version`, `/v1/models`, `/metrics` and `tpu.env`; and live server load from `/metrics`. Reasoning shows in a collapsible section. Stop closes the connection, which makes vLLM abort the request.

#### Flex-start capacity

Google Cloud can take a flex-start TPU back before its run limit. On 2026-11-02 the sister Rust rig's queued resource (`gemma4-rust-qr`) went to SUSPENDED with `stateInitiator: SERVICE` twice, each time well inside its limit. A SUSPENDED queued resource does not come back; delete it and create a new one.
