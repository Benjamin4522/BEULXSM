from llm.providers.router import LLMRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from rich.console import Console

console = Console()

class PlanStep(BaseModel):
    step_number: int
    description: str
    goal: str
    expected_output: str
    tools_needed: List[str]
    reasoning: str

class AgentPlan(BaseModel):
    overall_goal: str
    steps: List[PlanStep]
    estimated_complexity: str
    self_verification_strategy: str
    potential_risks: List[str]

class Planner:
    def __init__(self):
        self.router = LLMRouter()

    async def create_plan(self, user_goal: str, context: Optional[Dict] = None) -> AgentPlan:
        system_prompt = "Kamu adalah BEULXSM Planner. Buat rencana yang jelas, ringkas, dan actionable."

        prompt = f"""
{system_prompt}

Goal: {user_goal}

Buat rencana eksekusi dalam format JSON yang valid.
Pecah menjadi 3 sampai 6 step yang realistis.
Setiap step harus punya description, expected_output, dan tools_needed.

Output HANYA JSON, tanpa penjelasan tambahan.
"""

        try:
            response = await self.router.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                task_type="planner"   # ini akan memaksa pakai Groq (cepat)
            )

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            plan_dict = json.loads(content)
            plan = AgentPlan(**plan_dict)
            console.print(f"[green]Plan dibuat dengan {len(plan.steps)} steps[/]")
            return plan

        except Exception as e:
            console.print(f"[yellow]Planner error: {e}, menggunakan fallback[/]")
            return self._create_fallback_plan(user_goal)

    def _create_fallback_plan(self, goal: str) -> AgentPlan:
        return AgentPlan(
            overall_goal=goal,
            steps=[
                PlanStep(step_number=1, description="Analisis goal", goal=goal, 
                        expected_output="Pemahaman yang jelas", tools_needed=[], reasoning="Initial analysis"),
                PlanStep(step_number=2, description="Kumpulkan informasi yang dibutuhkan", goal=goal,
                        expected_output="Data dan referensi", tools_needed=["web_search"], reasoning="Research"),
                PlanStep(step_number=3, description="Buat solusi atau kode", goal=goal,
                        expected_output="Implementasi", tools_needed=["code_execution"], reasoning="Execution")
            ],
            estimated_complexity="medium",
            self_verification_strategy="Manual check",
            potential_risks=["Kurangnya detail"]
        )
