#!/bin/bash
set +e
LOG=/tmp/tc11-probe.log
RESP=/tmp/tc11-response.json
STAT=/tmp/tc11-status.txt
PAYLOAD=/tmp/tc11-payload.json

pkill -9 -f llama-server 2>/dev/null
sleep 5
echo "--- startup ---" > $STAT
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $STAT

# Create tc-11 payload
cat > $PAYLOAD << 'PAYLOAD'
{
  "model": "Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to tools. Use the provided tools when appropriate."
    },
    {
      "role": "user",
      "content": "What is 15% of 200?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string"
            }
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Calculate expressions",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {
              "type": "string"
            }
          },
          "required": ["expression"]
        }
      }
    }
  ],
  "temperature": 0,
  "stream": false
}
PAYLOAD

echo "--- payload size ---" >> $STAT
wc -c $PAYLOAD >> $STAT

cd /home/awee/code/worktrees/tc15-pr
nohup ./build-current/bin/llama-server --host 127.0.0.1 --port 18124 \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --ctx-checkpoints 0 --cache-ram 0 -np 1 \
  -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --reasoning off > $LOG 2>&1 &
SVR=$!
echo "PID=$SVR" >> $STAT

for i in $(seq 1 120); do
  if grep -q 'server is listening' $LOG 2>/dev/null; then
    echo "ready ${i}s" >> $STAT
    break
  fi
  if ! kill -0 $SVR 2>/dev/null; then
    echo "DIED at ${i}s" >> $STAT
    tail -10 $LOG >> $STAT
    exit 1
  fi
  sleep 1
done

echo "--- tc-11 probe ---" >> $STAT
curl -s -o $RESP http://127.0.0.1:18124/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @$PAYLOAD

# Parse result
python3 -c "
import sys, json
d = json.load(open('$RESP'))
t = d.get('timings', {})
print('prompt_tokens:', t.get('prompt_n', '?'))
print('predicted_tokens:', t.get('predicted_n', '?'))
print('eval tok/s: {:.2f}'.format(t.get('predicted_per_second', 0)))
print('prompt eval tok/s: {:.2f}'.format(t.get('prompt_per_second', 0)))
choices = d.get('choices', [])
if choices:
    msg = choices[0].get('message', {})
    content = msg.get('content', '')[:200]
    print('--- content ---')
    print(content if content else '(empty - likely tool_call)')
    print('--- tool_calls ---')
    tc = msg.get('tool_calls')
    if tc:
        for t in tc:
            fn = t.get('function', {})
            print(f'  {fn.get(\"name\")}: {fn.get(\"arguments\")}')
        print('RESULT: FAIL (called calculator when should answer directly)')
    else:
        print('  (none - direct answer)')
        print('RESULT: PASS (answered directly, no tool call)')
    print('--- stop_reason ---')
    print(choices[0].get('stop_reason'))
" >> $STAT

echo "--- GPU ---" >> $STAT
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.current.sm --format=csv,noheader >> $STAT

pkill -9 -f llama-server
sleep 2
echo "DONE" >> $STAT
cat $STAT
