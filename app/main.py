from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.resources.user import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="/app/templates")

API_PREFIX = "/api"

app.include_router(router, prefix=API_PREFIX)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/pages/{page_name}", response_class=HTMLResponse)
async def get_page(page_name: str):
    page_path = Path(f"/app/pages/{page_name}.html")
    if page_path.exists():
        return HTMLResponse(
            content=page_path.read_text(encoding="utf-8"), status_code=200
        )
    return HTMLResponse(content="Page not found", status_code=404)
