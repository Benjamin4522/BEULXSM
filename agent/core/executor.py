from llm.providers.router import LLMRouter
from .verifier import Verifier
from typing import Dict, Any, List
import asyncio
from rich.console import Console
from rich.panel import Panel

console = Console()

class Executor:
    """
    Executor BEULXSM - Menjalankan setiap step dari plan dengan tool calling,
    code execution, browser automation, dll (akan dikembangkan bertahap).
    """

    def __init__(self):
        self.router = LLMRouter()
        self.verifier = Verifier()
        self.execution_history = []

    async def execute_step(self, step: Any, context: Dict = None) -> Dict:
        """Eksekusi satu step dari plan"""
        console.print(Panel(f"[bold cyan]Executing Step {step.step_number}:[/] {step.description}", 
                           title="BEULXSM Executor", style="blue"))

        # Untuk tahap awal, kita gunakan LLM untuk menghasilkan action
        # Nanti akan diganti dengan real tool calling (coding, recon, browser, os_exec, dll)

        prompt = f"""
Kamu adalah Executor dari BEULXSM Agent.
Jalankan step berikut dengan sebaik mungkin:

Step {step.step_number}: {step.description}
Goal: {step.goal}
Expected Output: {step.expected_output}

Context saat ini: {context or 'No additional context'}

Berikan:
1. Action yang akan dilakukan (jelas dan detail)
2. Code / Command / Query yang perlu dijalankan (jika ada)
3. Expected result

Pikirkan step-by-step dan berikan output yang actionable.
"""

        response = await self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            task_type="coding" if "code" in step.description.lower() else "normal"
        )

        execution_result = {
            "step_number": step.step_number,
            "action": response.content,
            "raw_output": response.content,
            "model_used": response.model,
            "timestamp": asyncio.get_event_loop().time()
        }

        self.execution_history.append(execution_result)

        # Lakukan self-verification otomatis
        verification = await self.verifier.verify_step(
            step.description,
            step.expected_output,
            response.content
        )

        console.print(f"[green]✓ Step {step.step_number} executed | Confidence: {verification.confidence_score:.2f}[/]")

        return {
            "execution": execution_result,
            "verification": verification.dict()
        }

    async def execute_plan(self, plan: Any, initial_context: Dict = None) -> Dict:
        """Jalankan seluruh plan secara berurutan dengan verification"""
        results = {}
        context = initial_context or {}

        for step in plan.steps:
            result = await self.execute_step(step, context)
            results[step.step_number] = result

            # Update context untuk step berikutnya
            context[f"step_{step.step_number}_result"] = result["execution"]["raw_output"]

            # Jika confidence rendah, bisa trigger replanning nanti
            if result["verification"]["confidence_score"] < 0.6:
                console.print(f"[yellow]⚠ Low confidence on step {step.step_number}, may need replan[/]")

        # Simpan lessons learned ke memory nanti
        lessons = self.verifier.get_lessons_learned()
        if lessons:
            console.print(f"[magenta]📚 Learned {len(lessons)} lessons from this execution[/]")

        return {
            "plan": plan.dict(),
            "results": results,
            "lessons_learned": lessons
        }