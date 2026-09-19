from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import AsyncOpenAI

app = FastAPI()

# Khởi tạo client kết nối với OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "Chưa_có_API_Key")
)

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    # Router đơn giản: Dùng DeepSeek-R1 cho bài tập/thuật toán phức tạp, V3 cho giao tiếp cơ bản
    keywords_r1 = ["thuật toán", "giải bài", "toán", "giải thích", "code"]
    is_complex = any(kw in req.question.lower() for kw in keywords_r1)
    
    # OpenRouter cung cấp các endpoint :free cho DeepSeek
    model_name = "deepseek/deepseek-r1:free" if is_complex else "deepseek/deepseek-chat:free"
    
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