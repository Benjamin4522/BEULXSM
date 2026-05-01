from llm.providers.router import LLMRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from datetime import datetime

class VerificationResult(BaseModel):
    step_number: int
    is_success: bool
    confidence_score: float  # 0.0 - 1.0
    issues_found: List[str]
    suggestions: List[str]
    lessons_learned: str
    timestamp: str

class Verifier:
    """
    Self-Verification Engine untuk BEULXSM
    Bertanggung jawab memeriksa hasil setiap step, mendeteksi kesalahan,
    dan menyimpan lessons learned untuk long-term memory.
    """

    def __init__(self):
        self.router = LLMRouter()
        self.lessons_db = []  # nanti akan disimpan ke long-term memory / vector store

    async def verify_step(self, 
                         step_description: str, 
                         expected_output: str, 
                         actual_output: str,
                         context: Optional[Dict] = None) -> VerificationResult:
        """Verify satu step eksekusi"""

        prompt = f"""
Kamu adalah Verifier BEULXSM yang sangat kritis dan teliti.

Step: {step_description}
Expected Output: {expected_output}
Actual Output: {actual_output}

Tugas Verifikasi:
1. Bandingkan actual vs expected dengan sangat detail
2. Beri penilaian confidence score (0.0 - 1.0)
3. Temukan semua issues / kesalahan / ketidaksesuaian
4. Berikan saran perbaikan yang konkret
5. Ekstrak "lessons learned" yang berharga untuk masa depan

Berikan output dalam format JSON yang valid sesuai struktur VerificationResult.
Jangan tambahkan penjelasan di luar JSON.
"""

        response = await self.router.generate(
            messages=[{"role": "user", "content": prompt}],
            task_type="deep_reasoning"   # Pakai Gemini untuk verifikasi kritis
        )

        try:
            result_dict = json.loads(response.content)
            result = VerificationResult(**result_dict)
            
            # Simpan lessons learned
            if result.lessons_learned and result.lessons_learned.strip():
                self.lessons_db.append({
                    "timestamp": result.timestamp,
                    "lesson": result.lessons_learned,
                    "step": step_description
                })
            
            return result

        except Exception as e:
            print(f"Verifier parsing error: {e}")
            # Fallback
            return VerificationResult(
                step_number=0,
                is_success=False,
                confidence_score=0.4,
                issues_found=["Failed to parse verification result"],
                suggestions=["Manual review needed"],
                lessons_learned="Parser error occurred - improve JSON output stability",
                timestamp=datetime.now().isoformat()
            )

    async def verify_plan(self, plan, execution_results: Dict) -> List[VerificationResult]:
        """Verify keseluruhan plan setelah dieksekusi"""
        results = []
        for step in plan.steps:
            actual = execution_results.get(step.step_number, "No output recorded")
            verification = await self.verify_step(
                step.description, 
                step.expected_output, 
                actual
            )
            results.append(verification)
        return results

    def get_lessons_learned(self) -> List[Dict]:
        """Return semua lessons untuk dimasukkan ke long-term memory"""
        return self.lessons_db