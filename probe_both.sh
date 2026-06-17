#!/bin/bash
set +e

pkill -9 -f llama-server 2>/dev/null
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "MiB"

cd /home/awee/code/worktrees/tc15-pr
nohup ./build-current/bin/llama-server --host 127.0.0.1 --port 18124 \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --ctx-checkpoints 0 --cache-ram 0 -np 1 \
  -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --reasoning off > /tmp/both-probes.log 2>&1 &
SVR=$!
echo "PID=$SVR"
for i in $(seq 1 120); do
  grep -q 'server is listening' /tmp/both-probes.log 2>/dev/null && { echo "ready ${i}s"; break; }
  kill -0 $SVR 2>/dev/null || { echo DIED; tail -10 /tmp/both-probes.log; exit 1; }
  sleep 1
done

echo "=== TC-11 PROBE ==="
curl -s -o /tmp/tc11-responses.json http://127.0.0.1:18124/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf","messages":[{"role":"system","content":"You are a helpful assistant with access to tools. Use the provided tools when appropriate."},{"role":"user","content":"What is 15% of 200?"}],"tools":[{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},{"type":"function","function":{"name":"calculator","description":"Calculate expressions","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],"temperature":0,"stream":false}'

python3 -c "
import json
d = json.load(open('/tmp/tc11-responses.json'))
t = d.get('timings', {})
msg = d['choices'][0]['message']
tc = msg.get('tool_calls')
print(f'prompt_n: {t.get(\"prompt_n\")}  predicted_n: {t.get(\"predicted_n\")}  eval_tok/s: {t.get(\"predicted_per_second\",0):.2f}')
if tc:
    print(f'TC-11: FAIL (called {[f[\"function\"][\"name\"] for f in tc]})')
else:
    print(f'TC-11: PASS (no tool calls, content: {msg.get(\"content\",\"\")[:100]})')
"

echo "=== TC-15 PROBE ==="
curl -s -o /tmp/tc15-responses.json http://127.0.0.1:18124/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf","messages":[{"role":"system","content":"You are a helpful assistant with access to tools. Use the provided tools when appropriate."},{"role":"user","content":"Search for the population of Iceland and calculate what 2% of it would be."}],"tools":[{"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},{"type":"function","function":{"name":"calculator","description":"Calculate expressions","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],"temperature":0,"stream":false}'

python3 -c "
import json
d = json.load(open('/tmp/tc15-responses.json'))
t = d.get('timings', {})
msg = d['choices'][0]['message']
tc = msg.get('tool_calls')
print(f'prompt_n: {t.get(\"prompt_n\")}  predicted_n: {t.get(\"predicted_n\")}  eval_tok/s: {t.get(\"predicted_per_second\",0):.2f}')
if tc:
    names = [f['function']['name'] for f in tc]
    print(f'TC-15: called {names}')
    if 'web_search' in names:
        print('TC-15: PARTIAL PASS (called web_search at least)')
else:
    print(f'TC-15: FAIL (no tool calls, content: {msg.get(\"content\",\"\")[:100]})')
"

echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.current.sm --format=csv,noheader
echo "=== DRAFT ==="
grep -iE 'draft |acceptance|speculative' /tmp/both-probes.log | tail -5
