# Deploying Knots on one EC2 instance

One small EC2 instance runs two containers: **Caddy** (gets and renews the HTTPS certificate, forwards traffic) and
**Knots** (the MCP server and the Alexa endpoint). State (memory database, event log) lives on a Docker volume on
the instance's disk, so nothing needs a database service.

## What was and was not tested

| Tested here | Not tested here (so expect to debug it) |
| --- | --- |
| The server as a real process with auth on, over HTTP: no token or a forged token gets 401, a valid token works, a foreign Host header gets 421, unsigned Alexa POSTs get 400, ratings are recorded and cross-person ratings are refused | Building the Docker image and the compose file (there was no Docker daemon) |
| Alexa signature verification against a throwaway certificate authority, all the failure cases | Caddy getting a certificate, and every AWS step below (instance, role, security group, IP) |
| 217 automated tests | A live Alexa device or the Alexa simulator; Bedrock, TMDB and YouTube with real keys; memory use on a small instance |

## Cost, honestly

- A public IPv4 address costs **$0.005 per hour** (about $3.60 a month) whether or not it is in use
  ([AWS VPC pricing](https://aws.amazon.com/vpc/pricing/)). The instance, disk, Bedrock calls and data transfer are
  extra. With the Free Plan these are paid from your credits ($100, up to $200 with activities).
- A Free Plan account ends after six months or when credits run out, and then must move to a Paid Plan within 90 days
  or it is closed ([terms](https://aws.amazon.com/free/terms)). The plan is also "for evaluation" and not for sensitive
  data. Knots stores embeddings of what people say, which this repo treats as personal data, so **move to a Paid Plan
  before real listeners use it.**
- I could not confirm which instance types the Free Plan allows. Pick one your console offers. I suggest 2 GB of
  memory (for example `t3.small`) because numpy and scikit-learn are loaded; I have not measured the real figure, so
  watch `docker stats` and resize if it swaps.

## 1. Launch the instance

1. EC2 -> Launch instance. Amazon Linux 2023, 20 GB disk.
2. **Security group:** allow TCP 80 and 443 from anywhere, and TCP 22 **only from your own IP**. Never open 8000.
3. Allocate an **Elastic IP** and attach it, so the address survives a restart.
4. **Instance role** (IAM -> Roles -> EC2) with a policy like the one below, attached to the instance, so there are no
   AWS keys on the machine. Narrow `Resource` to your own model or inference profile once it works.

```json
{"Version": "2012-10-17", "Statement": [{
  "Effect": "Allow", "Action": ["bedrock:InvokeModel"],
  "Resource": ["arn:aws:bedrock:*::foundation-model/*", "arn:aws:bedrock:*:<ACCOUNT_ID>:inference-profile/*"]}]}
```

5. Containers cannot reach the instance role unless the metadata hop limit is 2 (the default of 1 stops them). Run
   this from a machine with the AWS CLI:

```bash
aws ec2 modify-instance-metadata-options --instance-id i-XXXX --http-tokens required --http-put-response-hop-limit 2
```

## 2. A hostname

HTTPS needs a name that points at the Elastic IP.

- **Free:** a [DuckDNS](https://www.duckdns.org) subdomain (sign in, pick a name, set its IP to your Elastic IP). I could
  not verify how Let's Encrypt's per-domain limits treat shared domains like this; if Caddy cannot get a certificate,
  `docker compose logs caddy` says why.
- **About $10 a year:** your own domain with an `A` record to the Elastic IP. Most reliable.

Alexa will only call a name whose certificate is signed by a trusted authority and whose name matches.

## 3. Install Docker and Knots

I have not run these commands; read them before pasting.

```bash
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user          # then log out and back in
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
     -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone https://github.com/spacey-cadet/knots && cd knots
cp .env.example .env && chmod 600 .env
```

## 4. Configure `.env`

Fill in at least:

| Variable | Value |
| --- | --- |
| `DOMAIN` | your hostname, e.g. `knots-yourname.duckdns.org` |
| `KNOTS_USER_SALT` | `openssl rand -hex 32`. Keep it: changing it orphans every stored profile |
| `AWS_REGION`, `BEDROCK_MODEL_ID`, `EMBED_MODEL_ID` | as in `.env.example` (credentials come from the instance role) |
| `TMDB_API_KEY`, `YOUTUBE_API_KEY` | your keys |
| `KNOTS_JWT_SECRET`, `KNOTS_JWT_AUDIENCE`, `KNOTS_JWT_ISSUER` | for MCP callers, see below. Audience: `https://<DOMAIN>/mcp` |
| `KNOTS_ALEXA_SKILL_ID` | from the Alexa console, see `alexa/README.md` |

## 5. Start and check

```bash
docker compose up -d --build
docker compose logs -f                      # Caddy should report an obtained certificate
curl https://<DOMAIN>/health                # {"ok":true}
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://<DOMAIN>/mcp    # 401: the token is required
```

## 6. Who may call the MCP endpoint

Every MCP call needs `Authorization: Bearer <token>`; the person is the token's `sub`.

- **For yourself or a demo:** set `KNOTS_JWT_SECRET` (32+ characters) and mint a token with
  `python scripts/mint_token.py --sub you --days 7`. The token is a password.
- **For real listeners:** use an identity provider and set `KNOTS_JWKS_URL` to its keys. Tokens must carry your
  `KNOTS_JWT_AUDIENCE` and be unexpired.
- **Alexa+:** how Alexa+ hands a signed-in person to an MCP server is not documented, and I have not verified how to
  register an MCP server with it. The endpoint is standard Streamable HTTP with bearer auth. Without a token the
  tools still answer but stay stateless (nothing remembered, no feedback).

## 7. Operate it

- **Logs:** `docker compose logs knots`. **Update:** `git pull && docker compose up -d --build`.
- **Online metrics:** the image holds only the server, so copy the log out and read it with the repo's tools:
  `docker compose cp knots:/data/events.jsonl ./events.jsonl`, then (once)
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, then
  `KNOTS_EVENT_LOG=./events.jsonl .venv/bin/python eval/monitor.py`. It reports the metrics and whether a retrain is
  due; add `--registry <dir> --freeze-baseline` once to record what normal looks like (`docs/METRICS.md`).
- **Backups:** snapshot the instance's EBS volume; `knots-data` (memory database, event log) is on it.
- **The event log has no rotation yet.** It only grows; check its size now and then.
- **Stopping charges:** stopping the instance does not stop the IPv4 charge. Terminate the instance and release the
  Elastic IP to stop all of it.
