#!/bin/bash
set +e
LOG=/tmp/probe-B-worktree.log
RESP=/tmp/probe-B-response.json
SUMM=/tmp/probe-B-summary.txt
STAT=/tmp/probe-B-status.txt

pkill -9 -f llama-server 2>/dev/null
sleep 4
echo "--- startup ---" > $STAT

cd /home/awee/code/worktrees/tc15-pr
nohup ./build-current/bin/llama-server --host 127.0.0.1 --port 18125 \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --ctx-checkpoints 0 --cache-ram 0 -np 1 \
  -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --reasoning off > $LOG 2>&1 &
SVR=$!
echo "PID=$SVR" >> $STAT

for i in $(seq 1 150); do
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

echo "--- probe ---" >> $STAT
curl -s -o $RESP http://127.0.0.1:18125/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @/tmp/tc15-payload.json

python3 /tmp/parse_tc15.py < $RESP >> $STAT

echo "--- GPU ---" >> $STAT
nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.current.sm --format=csv,noheader >> $STAT

echo "--- DRAFT ---" >> $STAT
grep -iE 'draft |nextn|acceptance|speculative' $LOG | tail -8 >> $STAT

pkill -9 -f llama-server
sleep 2
echo "DONE" >> $STAT
cat $STAT
