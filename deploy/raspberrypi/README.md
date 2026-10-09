# AI Trader scheduler coordination fix

## Behavior
- Both scripts use `/var/lock/ai-trader-operation.lock`.
- Update skips if a scan/update is active; scan skips if update/scan is active.
- Updater uses `git fetch` + `git merge --ff-only`; no `reset`, `clean`, `docker compose down`, or `docker compose up`.
- Incoming Git paths are checked against local changes; overlapping paths abort the update.
- Docker image is built only when `HEAD` differs from the last successfully built commit marker.
- Failed builds do not update the marker, so a later hourly run retries.
- Scan uses `docker compose run --rm --no-build ai-trader`.
- No Telegram credentials are embedded. Optional update notifications read `/home/luigi/tools/ai-trader-telegram.env` after the bot token has been rotated.

## PC validation (PowerShell)
From `C:\Scripts\Python\ai-trader` after copying this directory into the repo:
```powershell
git diff --check
bash -n deploy/raspberrypi/update_ai_trader.sh
bash -n deploy/raspberrypi/run_ai_trader.sh
```
`bash` must be available (for example via Git Bash). These checks parse the scripts without running them.

Review and commit these files from the PC, then push to `origin/main`.

## One-time deployment
Use your normal PC-to-Pi deployment method to copy these two scripts to:
- `/home/luigi/tools/update_ai_trader.sh`
- `/home/luigi/tools/run_ai_trader.sh`

Then on the Pi set ownership/permissions:
```bash
sudo chown root:root /home/luigi/tools/update_ai_trader.sh /home/luigi/tools/run_ai_trader.sh
sudo chmod 750 /home/luigi/tools/update_ai_trader.sh /home/luigi/tools/run_ai_trader.sh
```

After rotating the exposed Telegram bot token, optional notifications can be enabled with a root-owned file `/home/luigi/tools/ai-trader-telegram.env`:
```bash
AI_TRADER_TELEGRAM_BOT_TOKEN="NEW_TOKEN"
AI_TRADER_TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```
Protect it with `sudo chown root:root /home/luigi/tools/ai-trader-telegram.env && sudo chmod 600 /home/luigi/tools/ai-trader-telegram.env`. Do not put the token in Git.

Run one controlled updater/build:
```bash
sudo /home/luigi/tools/update_ai_trader.sh
cat /home/luigi/tools/ai-trader-built-commit
cd /home/luigi/ai-trader && git rev-parse HEAD && docker compose images
```
The marker must match `git rev-parse HEAD`. Then run one controlled scan:
```bash
sudo /home/luigi/tools/run_ai_trader.sh
```
Review `/home/luigi/tools/run_ai_trader.log` and confirm it completes normally.

Only after both pass, enable these entries in root's crontab (`sudo crontab -e`):
```cron
30 16 * * 1-5 /home/luigi/tools/update_ai_trader.sh
*/15 * * * * /home/luigi/tools/run_ai_trader.sh
```
Do not leave duplicate active entries in other schedulers.

## Scope and safeguards
This change does not modify `app/main.py`, trading logic, risk gates, market cache, prediction files, or Dockerfile. If an incoming commit changes a locally modified path, updater aborts rather than overwriting it. No automatic `reset` or `clean`.
Because a Telegram bot token was exposed in chat, rotate it separately; these scripts do not use Telegram.
