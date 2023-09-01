import json
import os
from pathlib import Path
import subprocess
import threading
from types import resolve_bases
import uvicorn
import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles


async def run_krr():
    subprocess.check_output(["python", "custom.py", "custom", "-f", "custom", "--mem-min", "16", "-w", "2"])


def run_krr_once():
    asyncio.run(run_krr())


async def startup_event():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_krr, CronTrigger.from_crontab("30 */2 * * *"))
    scheduler.start()
    threading.Thread(target=run_krr_once).start()


async def home(request: Request) -> HTMLResponse:
    env = os.getenv("ENV")
    result_file = "static/result.svg"
    html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>{env} 集群资源优化 dashboard</title>
        </head>
        <body>
        <h2>{env} 集群资源优化 dashboard</h2>
        {result}
        </img>
        </body>
        </html>
        """.format(env=env,
                   result="<img src=\"/%s\">" % result_file if os.path.exists(result_file) else "<p>结果暂未生成，请稍等...</p>")
    return HTMLResponse(html)


async def json_result(request: Request) -> JSONResponse:
    with open("static/result.json", "r") as f:
        d = json.load(f)
    return JSONResponse(d)

Path("static").mkdir(parents=True, exist_ok=True)

Routes = [
    Route("/", endpoint=home),
    Route("/api/json", endpoint=json_result),
    Mount("/static", app=StaticFiles(directory="static"))
]

app = Starlette(on_startup=[startup_event], routes=Routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
