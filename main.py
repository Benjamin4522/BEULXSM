import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console

console = Console()

async def main():
    load_dotenv()
    
    console.print("[bold magenta]🚀 BEULXSM Agent Starting...[/]")
    
    console.print(f"Groq Key        : {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
    console.print(f"Mistral Key     : {'✅' if os.getenv('MISTRAL_API_KEY') else '❌'}")
    console.print(f"Together Key    : {'✅' if os.getenv('TOGETHER_API_KEY') else '❌'}\n")

    try:
        from agent.core.agent import BeulxsmAgent
        console.print("[green]✅ Import berhasil[/]")
        
        agent = BeulxsmAgent()
        
        goal = "Buat fungsi Python sederhana untuk menghitung faktorial."
        
        console.rule("[bold blue]Menjalankan BEULXSM Agent[/]")
        result = await agent.run(goal)
        
        console.print("[bold green]✅ Test selesai![/]")
        
    except Exception as e:
        console.print(f"[bold red]ERROR: {type(e).__name__}: {e}[/]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
