# Deploying the SFSA Agentic RAG Assistant on AWS EC2

This guide takes a blank AWS account to a running, password-protected web UI at
`https://your-host/` that SFSA members can use for batch question runs.

Everything the assistant needs — the LLM (via Ollama), the embedding model, and the
FAISS vector database — runs **on the instance**. No question text or document content
leaves the box except when Agent 2 calls the Tavily web-search API.

**Contents**

1. [Choosing an instance](#1-choosing-an-instance)
2. [Launching the instance](#2-launching-the-instance)
3. [Base system setup](#3-base-system-setup)
4. [Installing Ollama and pulling the model](#4-installing-ollama-and-pulling-the-model)
5. [Installing the application](#5-installing-the-application)
6. [Getting the vector database onto the instance](#6-getting-the-vector-database-onto-the-instance)
7. [Configuration (`.env`)](#7-configuration-env)
8. [Smoke test](#8-smoke-test)
9. [Creating member accounts](#9-creating-member-accounts)
10. [Running the GUI as a service](#10-running-the-gui-as-a-service)
11. [Putting HTTPS in front of it](#11-putting-https-in-front-of-it-recommended)
12. [Day-2 operations](#12-day-2-operations)
13. [Troubleshooting](#13-troubleshooting)
14. [Cost notes](#14-cost-notes)

---

## 1. Choosing an instance

Three things drive the sizing:

| Component | Resource demand |
|---|---|
| `llama3.1:8b` via Ollama | ~5 GB of GPU VRAM (or ~8 GB RAM if CPU-only) |
| `Alibaba-NLP/gte-large-en-v1.5` embeddings | ~2 GB, loaded per process |
| FAISS index (`index.faiss` + `index.pkl`) | Loaded fully into RAM — currently ~660 MB |

### Recommended: GPU instance

| Instance | vCPU | RAM | GPU | Notes |
|---|---|---|---|---|
| **`g5.xlarge`** | 4 | 16 GB | A10G 24 GB | **Recommended.** Comfortable for `llama3.1:8b`, headroom for a larger model later. |
| `g4dn.xlarge` | 4 | 16 GB | T4 16 GB | Cheaper, noticeably slower generation. Fine for low-volume use. |

A batch of questions runs the LLM 2–5 times *per question* (contextualize → Agent 1 →
generate → Agent 3, plus refinement loops). On CPU that is minutes per question; on a
GPU it is seconds. If you plan to run CSV batches of any size, use a GPU instance.

### CPU-only (works, but slow)

| Instance | vCPU | RAM | Notes |
|---|---|---|---|
| `c7i.4xlarge` | 16 | 32 GB | Roughly 10–20× slower per question than `g5.xlarge`. |
| `t3.2xlarge` | 8 | 32 GB | Burstable — CPU credits will run out during a batch. Testing only. |

**Do not go below 16 GB RAM.** The FAISS index, the embedding model, and Ollama are all
resident at once.

### Storage

Use a **100 GB gp3** root volume. Budget:

- Ollama models: ~5 GB per model
- Conda environment (incl. PyTorch): ~8–12 GB
- HuggingFace model cache: ~3 GB
- Vector database: ~1 GB
- Output CSVs and logs: grows over time

### AMI

**Ubuntu Server 22.04 LTS (x86_64)** for CPU instances, or the **AWS Deep Learning AMI
(Ubuntu 22.04)** for GPU instances — the DLAMI ships with NVIDIA drivers and CUDA already
installed, which saves a fussy step. This guide assumes Ubuntu 22.04 with user `ubuntu`.

---

## 2. Launching the instance

1. **EC2 → Launch instance.**
2. **Name**: `sfsa-rag-assistant`
3. **AMI**: as above.
4. **Instance type**: as above.
5. **Key pair**: create or select one. Keep the `.pem` file safe — it is your only way in.
6. **Network settings → Create security group**, with these inbound rules:

   | Type | Port | Source | Why |
   |---|---|---|---|
   | SSH | 22 | **My IP** | Administration. Never `0.0.0.0/0`. |
   | HTTP | 80 | `0.0.0.0/0` | Only if you set up nginx + TLS (section 11). |
   | HTTPS | 443 | `0.0.0.0/0` | Only if you set up nginx + TLS (section 11). |
   | Custom TCP | 7860 | **My IP** | Direct access to the app during setup. |

   > **Do not open 7860 to `0.0.0.0/0`.** The app's login page is served over plain
   > HTTP, so credentials would cross the internet in the clear. Either keep 7860
   > restricted to your own IP for testing, or put nginx + TLS in front of it
   > (section 11) and expose only 443.

7. **Configure storage**: 100 GB gp3.
8. **Launch**, then note the **public IPv4 address** (or attach an Elastic IP so the
   address survives a stop/start).

Connect:

```bash
chmod 400 ~/Downloads/sfsa-key.pem
ssh -i ~/Downloads/sfsa-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## 3. Base system setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl build-essential
```

On a GPU instance, confirm the driver is live (skip if you used the DLAMI and it already
works):

```bash
nvidia-smi
```

Install Miniforge (conda + mamba), which the project's `environment.yml` targets:

```bash
curl -L -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
exec bash
```

---

## 4. Installing Ollama and pulling the model

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The installer registers and starts a `systemd` service listening on
`127.0.0.1:11434`, which is exactly what you want — the application talks to Ollama over
localhost, and Ollama should **never** be exposed to the internet (it has no
authentication).

Pull the model:

```bash
ollama pull llama3.1:8b
ollama list
```

Verify the service:

```bash
systemctl status ollama --no-pager
curl -s http://localhost:11434/api/tags | head -c 400
```

### Keeping the model warm

By default Ollama unloads a model from memory after 5 minutes idle, so the first question
after a quiet period pays a reload penalty. To keep it resident:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

Then `sudo systemctl restart ollama`.

---

## 5. Installing the application

Clone the private repository. Two options:

**Option A — deploy key (recommended for a server).** On the instance:

```bash
ssh-keygen -t ed25519 -C "sfsa-ec2-deploy" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Copy that public key, then in GitHub go to the repository → **Settings → Deploy keys →
Add deploy key**, paste it, leave "Allow write access" **unchecked**, and save. Then:

```bash
cd ~
git clone git@github.com:<your-username>/sfsa-rag-assistant.git
cd sfsa-rag-assistant
```

**Option B — personal access token.** Create a fine-grained PAT with read-only Contents
access to this repository and use `git clone https://github.com/<you>/sfsa-rag-assistant.git`,
supplying the token when prompted for a password.

Build the environment (this takes 10–20 minutes — PyTorch and friends are large):

```bash
conda env create -f environment.yml
conda activate sfsa_rag_assistant
```

The environment installs the package in editable mode (`pip install -e .`), so the
console scripts `sfsa-rag`, `sfsa-rag-gui`, and `sfsa-rag-admin` are on your `PATH` once
the environment is active.

---

## 6. Getting the vector database onto the instance

**The FAISS index is not stored in this repository** — at ~660 MB it exceeds GitHub's
100 MB per-file limit. You must supply it separately. Pick one:

### Option A — copy it from your workstation (simplest)

From your local machine, where the database already exists:

```bash
scp -i ~/Downloads/sfsa-key.pem -r \
  "src/sfsa_rag_assistant/data/vectordb" \
  ubuntu@<EC2_PUBLIC_IP>:~/sfsa-rag-assistant/src/sfsa_rag_assistant/data/
```

(Create the parent directory first if needed:
`ssh ... 'mkdir -p ~/sfsa-rag-assistant/src/sfsa_rag_assistant/data'`.)

### Option B — stage it in S3 (best for rebuilding instances)

Once, from your workstation:

```bash
aws s3 cp --recursive \
  "src/sfsa_rag_assistant/data/vectordb" \
  s3://your-sfsa-bucket/vectordb/
```

Then on each instance (attach an IAM instance role granting `s3:GetObject` on that
prefix — do not put access keys on the box):

```bash
sudo snap install aws-cli --classic
aws s3 cp --recursive s3://your-sfsa-bucket/vectordb/ \
  ~/sfsa-rag-assistant/src/sfsa_rag_assistant/data/vectordb/
```

### Option C — rebuild it from the source PDFs

Put the SFSA PDFs on the instance (for example in `~/sfsa-rag-assistant/data/raw/`) and:

```bash
cd ~/sfsa-rag-assistant
conda activate sfsa_rag_assistant
python - <<'PY'
from sfsa_rag_assistant.data_processing import DataProcessor

DataProcessor(
    data_path="data/raw",
    vectordb_path="src/sfsa_rag_assistant/data/vectordb",
).process_and_create_db()
PY
```

This walks the directory recursively, splits each PDF into 1500-character chunks with
100-character overlap, embeds them, and writes the FAISS index. Expect this to take a
while and to need the GPU (or a lot of patience) for a large document set.

Verify whichever route you took:

```bash
ls -lh ~/sfsa-rag-assistant/src/sfsa_rag_assistant/data/vectordb/
# index.faiss and index.pkl should both be present
```

---

## 7. Configuration (`.env`)

```bash
cd ~/sfsa-rag-assistant
cp .env.example .env
nano .env
```

The values that matter on a server:

```ini
# Web search (Agent 2). Get a key at https://tavily.com — free tier is fine to start.
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx

# Ollama runs locally on this instance
SFSA_OLLAMA_BASE_URL=http://localhost:11434
SFSA_OLLAMA_MODEL=llama3.1:8b
SFSA_OLLAMA_TEMPERATURE=0.3
SFSA_OLLAMA_MAX_TOKENS=1000

# Vector database (relative paths resolve from the project root)
SFSA_VECTORDB_PATH=src/sfsa_rag_assistant/data/vectordb
SFSA_EMBEDDING_MODEL=Alibaba-NLP/gte-large-en-v1.5
SFSA_RETRIEVAL_K=5

# Agent 3 refinement loops
SFSA_MAX_VALIDATION_ATTEMPTS=2

# Where batch results land on the server
SFSA_OUTPUT_DIR=data/outputs

# Member accounts and the usage audit log
SFSA_AUTH_DB_PATH=data/auth/sfsa_auth.db
```

**LangSmith tracing is on by default** and will send prompts and responses to
LangSmith's servers. If you do not want that — and on a members-facing deployment you
probably do not — set:

```ini
LANGCHAIN_TRACING_V2=false
```

and leave `LANGCHAIN_API_KEY` unset.

Lock the file down, since it holds your Tavily key:

```bash
chmod 600 .env
```

> **Paths are resolved relative to the working directory** the process starts in. Always
> launch the app from `~/sfsa-rag-assistant` (the systemd unit in section 10 does this
> for you).

---

## 8. Smoke test

The first run downloads the embedding model from HuggingFace (~2 GB) and will pause for a
few minutes. It also needs outbound internet access.

```bash
cd ~/sfsa-rag-assistant
conda activate sfsa_rag_assistant
python -m sfsa_rag_assistant "What is steel casting?" --show-metadata
```

You should see the answer, a `SOURCES` block citing SFSA Wiki documents, and metadata
showing which agents fired. If that works, test the batch path:

```bash
printf 'question\nWhat is steel casting?\nWhat causes hot tearing?\n' > /tmp/test.csv
python -m sfsa_rag_assistant --batch /tmp/test.csv --output /tmp/test_results.csv
head -c 600 /tmp/test_results.csv
```

And the GUI, in the foreground for now:

```bash
python -m sfsa_rag_assistant --gui --host 0.0.0.0 --port 7860
```

Browse to `http://<EC2_PUBLIC_IP>:7860`. With no member accounts created yet, the app
redirects straight to `/app` with **no authentication** — that is why port 7860 is
restricted to your IP. Stop it with `Ctrl-C` and create accounts next.

---

## 9. Creating member accounts

Authentication switches on automatically as soon as **one active account exists**. Until
then the UI is wide open, so create the first account before exposing the service.

```bash
cd ~/sfsa-rag-assistant
conda activate sfsa_rag_assistant

sfsa-rag-admin create-user sfsa_member \
  --full-name "SFSA Member" \
  --email "member@example.com"
```

You are prompted for the password twice; it is never passed on the command line. It is
stored as a PBKDF2-HMAC-SHA256 hash (600,000 iterations) with a per-user salt in the
SQLite database at `data/auth/sfsa_auth.db`.

Other admin commands:

```bash
sfsa-rag-admin list-users                    # accounts, status, last login, last use
sfsa-rag-admin deactivate-user sfsa_member   # revoke access and kill live sessions
sfsa-rag-admin activate-user sfsa_member     # restore access
sfsa-rag-admin usage --limit 50              # recent activity
```

The usage log records username, action, status, model, source IP, output file, and run
summary for every batch job — useful for seeing how each credential is being used.

Re-running `create-user` for an existing username **resets that user's password** and
invalidates their sessions. Sessions expire after 12 hours of inactivity.

Back up the auth database — it holds your member accounts and the audit trail:

```bash
cp data/auth/sfsa_auth.db ~/backups/sfsa_auth_$(date +%F).db
```

---

## 10. Running the GUI as a service

So the app survives logout and reboots.

```bash
sudo nano /etc/systemd/system/sfsa-rag.service
```

```ini
[Unit]
Description=SFSA Agentic RAG Assistant
After=network-online.target ollama.service
Wants=network-online.target
Requires=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sfsa-rag-assistant
Environment=PATH=/home/ubuntu/miniforge3/envs/sfsa_rag_assistant/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/miniforge3/envs/sfsa_rag_assistant/bin/python -m sfsa_rag_assistant --gui --host 127.0.0.1 --port 7860
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

`--host 127.0.0.1` binds to localhost only, so the app is reachable exclusively through
nginx (section 11). If you are not using nginx yet, use `--host 0.0.0.0` and keep port
7860 locked to your own IP in the security group.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sfsa-rag
systemctl status sfsa-rag --no-pager
journalctl -u sfsa-rag -f
```

---

## 11. Putting HTTPS in front of it (recommended)

Members log in with a username and password. Without TLS those credentials travel in
plaintext. Point a DNS A record (for example `sfsa-rag.example.org`) at the instance's
Elastic IP, then:

```bash
sudo apt install -y nginx
sudo nano /etc/nginx/sites-available/sfsa-rag
```

```nginx
server {
    listen 80;
    server_name sfsa-rag.example.org;

    # Batch runs are long; don't let the proxy time them out.
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    # CSV uploads
    client_max_body_size 64M;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Gradio uses websockets for live progress
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/sfsa-rag /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo snap install --classic certbot
sudo ln -sf /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d sfsa-rag.example.org
```

Certbot rewrites the config for TLS and installs an auto-renewal timer. Afterwards,
**remove the port 7860 rule from the security group entirely** — nothing outside the
instance needs it.

> Note: the session cookie is set without the `Secure` flag. Behind TLS it still travels
> encrypted, but if you ever serve the same hostname over plain HTTP the cookie will go
> with it. Redirect HTTP → HTTPS (certbot offers to do this) so that cannot happen.

---

## 12. Day-2 operations

**Deploy a code update:**

```bash
cd ~/sfsa-rag-assistant
git pull
conda env update -f environment.yml --prune   # only if dependencies changed
sudo systemctl restart sfsa-rag
```

**Change the model:** pull it in Ollama, update `SFSA_OLLAMA_MODEL` in `.env`, restart
the service.

```bash
ollama pull llama3.1:70b   # needs a much larger GPU
```

**Rotate the Tavily key:** edit `.env`, `sudo systemctl restart sfsa-rag`.

**Where things live:**

| Path | Contents |
|---|---|
| `~/sfsa-rag-assistant/.env` | Secrets and configuration |
| `~/sfsa-rag-assistant/data/outputs/` | Batch result CSVs |
| `~/sfsa-rag-assistant/data/auth/sfsa_auth.db` | Accounts, sessions, usage log |
| `~/sfsa-rag-assistant/src/sfsa_rag_assistant/data/vectordb/` | FAISS index |
| `~/.ollama/models/` | Downloaded LLMs |
| `journalctl -u sfsa-rag` | Application logs |

**Back up** `.env`, the auth database, and `data/outputs/`. The vector database and the
Ollama models are reproducible; the accounts and results are not.

**Save money when idle:** `aws ec2 stop-instances --instance-ids i-xxxx`. Stopped
instances bill only for EBS storage. Use an Elastic IP so the address is stable across
stop/start. On restart, `ollama` and `sfsa-rag` both come back automatically via systemd.

---

## 13. Troubleshooting

**`Cannot connect to Ollama`**

```bash
systemctl status ollama --no-pager
curl -s http://localhost:11434/api/tags
ollama list          # is the model in SFSA_OLLAMA_MODEL actually pulled?
```

The model name must match what `ollama list` shows, including the tag (`llama3.1:8b`,
not `llama3.1`).

**Startup hangs for several minutes on first run** — expected. It is downloading
`Alibaba-NLP/gte-large-en-v1.5` from HuggingFace. Watch with
`journalctl -u sfsa-rag -f`. It is cached in `~/.cache/huggingface` afterwards.

**`RuntimeError` or `FileNotFoundError` about the vector store** — the FAISS index is
missing or `SFSA_VECTORDB_PATH` is wrong. Confirm `index.faiss` and `index.pkl` exist at
the configured path, and that the process's working directory is the project root
(relative paths resolve from there).

**Process killed / OOM during a batch** — check `dmesg | tail`. The FAISS index, the
embedding model, and Ollama together need more than 16 GB on smaller instances. Move to
a larger instance, or lower `SFSA_RETRIEVAL_K`.

**Web search never runs** — `TAVILY_API_KEY` is unset or invalid. Agent 2 only fires when
Agent 1 judges the retrieved SFSA context insufficient, so also confirm you are asking
something outside the wiki's coverage.

**Login page loops back to itself** — the session cookie is being dropped. Check that
nginx forwards `Host` and that you are not mixing `http://IP` and `https://hostname`.

**Batch times out in the browser** — raise `proxy_read_timeout` in nginx. The run
continues server-side regardless; the CSV still lands in `data/outputs/`.

**GPU not being used** — `nvidia-smi` during a query should show `ollama` resident. If
not, the driver is missing or Ollama fell back to CPU; check `journalctl -u ollama`.

---

## 14. Cost notes

Approximate on-demand US-East-1 pricing (check current rates — these move):

| Item | Rough cost |
|---|---|
| `g5.xlarge`, 24×7 | ~$730/month |
| `g5.xlarge`, 8h × 22 weekdays | ~$180/month |
| `c7i.4xlarge`, 24×7 | ~$520/month |
| 100 GB gp3 | ~$8/month |
| Elastic IP (attached to a running instance) | free |
| Tavily | free tier, then usage-based |

The biggest lever is **stopping the instance when nobody is using it**. A Savings Plan or
Reserved Instance cuts 24×7 GPU costs substantially if the service needs to stay up.
