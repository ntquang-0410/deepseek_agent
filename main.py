from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import AsyncOpenAI

app = FastAPI()

# Khởi tạo client kết nối với Groq API
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", "Chưa_có_API_Key")
)

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    # Router phân loại câu hỏi
    keywords_r1 = ["thuật toán", "giải bài", "toán", "giải thích", "code"]
    is_complex = any(kw in req.question.lower() for kw in keywords_r1)
    
    # Sử dụng chung một model ổn định nhất cho mọi loại câu hỏi
    model_name = "llama-3.1-8b-instant"
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI của ứng dụng SmartStudy AI. Hãy trả lời ngắn gọn, dễ hiểu bằng tiếng Việt."},
                {"role": "user", "content": req.question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}