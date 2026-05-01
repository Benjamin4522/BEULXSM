from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from typing import Dict, Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

class BeulxsmAgent:
    """
    Main Class BEULXSM Agent
    Menggabungkan Planner + Executor + Verifier + Memory (akan ditambah)
    """

    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()
        self.verifier = Verifier()
        self.name = "BEULXSM"
        console.print("[bold green]BEULXSM Agent initialized successfully![/]")

    async def run(self, goal: str, context: Optional[Dict] = None):
        """Main entry point untuk menjalankan agent"""
        console.print(Panel(f"[bold yellow]New Goal:[/] {goal}", title="🚀 BEULXSM Started", style="yellow"))

        # Step 1: Planning (Goal Decomposition)
        console.print("[bold blue]Phase 1: Creating Detailed Plan...[/]")
        plan = await self.planner.create_plan(goal, context)

        # Step 2: Execution
        console.print("[bold blue]Phase 2: Executing Plan...[/]")
        execution_result = await self.executor.execute_plan(plan, context)

        # Step 3: Final Verification & Summary
        console.print("[bold blue]Phase 3: Final Review & Lessons Learned...[/]")
        
        console.print(Panel(
            f"Goal: {goal}\n"
            f"Complexity: {plan.estimated_complexity}\n"
            f"Steps Completed: {len(plan.steps)}\n"
            f"Lessons Learned: {len(execution_result['lessons_learned'])}",
            title="✅ Mission Summary",
            style="green"
        ))

        return execution_result

    async def chat(self, message: str):
        """Mode chat sederhana (bisa dikembangkan jadi ReAct loop)"""
        return await self.run(message)