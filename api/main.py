from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BEULXSM - Advanced AI Agent")
templates = Jinja2Templates(directory="api/frontend/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="api/frontend/static"), name="static")

# Import Agent
from agent.core.agent import BeulxsmAgent

agent = BeulxsmAgent()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    if not user_message:
        return {"response": "Pesan tidak boleh kosong."}

    try:
        console = __import__('rich.console').console.Console()
        console.print(f"[yellow]User:[/] {user_message}")

        # Jalankan agent
        result = await agent.run(user_message)

        # Ambil response terakhir
        response_text = "Goal telah dieksekusi. Silakan lihat log terminal untuk detail lengkap."

        # Kalau mau ambil output lebih baik, bisa di-improve nanti
        if isinstance(result, dict) and "results" in result:
            response_text = f"Agent telah memproses goal Anda.\nJumlah step: {len(result.get('results', {}))}"

        return {"response": response_text}

    except Exception as e:
        return {"response": f"Error saat menjalankan agent: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)