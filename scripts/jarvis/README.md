# Jarvis lane — setup (JarvisLabs A30 to start, PyTorch template, ≥ 200 GB storage)

1. JupyterLab: upload `data/6ab10eb3b23ba_student_resource.zip` from the laptop into `/home`.
2. JupyterLab terminal (GitHub asks for username + a fine-grained token for this repo, once):
       git config --global credential.helper "store --file /home/.git-credentials"
       cd /home && git clone https://github.com/Blacknix809/amazon-ml-2026.git
       bash /home/amazon-ml-2026/scripts/jarvis/setup_jarvis.sh
3. `claude` → log in → `/exit` (if `claude` is not found yet: `/root/.local/bin/claude`).
4. Start the lane:
       tmux new -d -s jarvis 'bash /home/amazon-ml-2026/scripts/jarvis/jarvis_loop.sh'
       sleep 90; tail -3 /home/amazon-ml-2026/runs/jarvis/driver.log
Stop: `touch /home/amazon-ml-2026/runs/jarvis/STOP`. To switch GPU: STOP, wait for 'driver end', pause, then
`jl resume <id> --gpu <type>` (same region; /home is kept). After any resume: re-run the setup line (tools outside
/home are gone), log in to claude again, then step 4.
