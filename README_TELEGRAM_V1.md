# AI Trader — Telegram Analysis V1

## Flow

```text
Telegram -> app.telegram.bot -> handlers -> app.services.analysis_service
         -> Watchlist Scanner V2 -> Orchestrator -> Decision Gate -> Signal Engine
         -> formatter -> Telegram
```

Telegram is only the interface; no second trading algorithm is created.

## Command

```text
/analyze GDXU
```

Also: `/start`, `/help`.

## Environment

Required:

```text
TELEGRAM_BOT_TOKEN=...
```

Recommended:

```text
TELEGRAM_ALLOWED_USER_IDS=123456789
```

If the allow-list is absent, `TELEGRAM_CHAT_ID` is used as the single allowed ID. If neither is set, access is denied.

## Run

From project root:

```powershell
python -m app.telegram.bot
```

`python-telegram-bot` and `python-dotenv` are already present in the audited requirements.

## V1 scope

Included: `/analyze`, authentication, polling, worker-thread execution, formatted result, dynamic levels when available, no fixed buy target.

Not included: compare, watchlist, buttons, charts, webhook, institutional drill-down.
