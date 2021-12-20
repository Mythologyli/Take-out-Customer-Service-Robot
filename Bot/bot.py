import nonebot
from nonebot import logger
from nonebot.adapters.cqhttp import Bot as CQHTTPBot


logger.add("./logs/{time:YYYY-MM-DD}.log", rotation="0:00")

nonebot.init()
app = nonebot.get_asgi()
driver = nonebot.get_driver()
driver.register_adapter("cqhttp", CQHTTPBot)

nonebot.load_plugins("src/plugins")
