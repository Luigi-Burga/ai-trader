import asyncio

from app.alerts.telegram_alert import send_alert

async def main():

    await send_alert("TEST ALERT FROM AI TRADING AGENT")

asyncio.run(main())