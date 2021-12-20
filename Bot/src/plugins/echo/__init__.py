from nonebot import on_command
from nonebot import logger
from nonebot.adapters.cqhttp import Bot, Message, MessageEvent


echo = on_command('echo')


@echo.handle()
async def handle_echo(bot: Bot, event: MessageEvent):
    arg = str(event.message).strip()

    logger.debug(f"echo: {arg}")

    await echo.send(arg)
