import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = FastAPI(title="Smart Agent Backend - GitHub Models")

# Chỉ cần khởi tạo 1 client duy nhất cho mọi model
client = ChatCompletionsClient(
    endpoint=os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com"),
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN"))
)

class QueryRequest(BaseModel):
    question: str

def route_question(question: str) -> str:
    """Sử dụng DeepSeek-V3 để phân loại câu hỏi nhanh"""
    router_prompt = """
    Bạn là bộ định tuyến. Hãy phân loại yêu cầu của học sinh vào 1 trong 2 nhóm:
    1. "COMPLEX": Các bài toán, câu hỏi cần giải thích từng bước, suy luận logic phức tạp.
    2. "GENERAL": Chào hỏi, trò chuyện thông thường, hoặc tra cứu kiến thức đơn giản.
    Chỉ trả về 1 từ duy nhất là "COMPLEX" hoặc "GENERAL".
    """
    response = client.complete(
        messages=[
            SystemMessage(content=router_prompt),
            UserMessage(content=question)
        ],
        model="DeepSeek-V3", # Gọi thẳng V3 thông qua GitHub Models
        max_tokens=10
    )
    return response.choices[0].message.content.strip().upper()

def extract_thought_process(text: str):
    """Bóc tách luồng suy luận của R1 ra khỏi câu trả lời"""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        answer = text.replace(think_match.group(0), '').strip()
        return thought, answer
    return "", text

@app.post("/api/ask")
async def ask_agent(req: QueryRequest):
    try:
        # Bước 1: Phân loại câu hỏi
        intent = route_question(req.question)
        
        # Bước 2: Gọi mô hình xử lý
        if "COMPLEX" in intent:
            # Bài tập khó -> Gọi R1 để suy luận từng bước
            response = client.complete(
                messages=[
                    SystemMessage(content="Bạn là chuyên gia giải bài tập. Hãy suy luận cẩn thận từng bước."),
                    UserMessage(content=req.question)
                ],
                model="DeepSeek-R1", # Đổi tham số model thành R1
                max_tokens=2048
            )
            raw_result = response.choices[0].message.content
            thought, final_answer = extract_thought_process(raw_result)
            
            return {
                "agent_type": "DeepSeek-R1",
                "thought_process": thought,
                "answer": final_answer
            }
            
        else:
            # Câu hỏi bình thường -> V3 xử lý trực tiếp
            response = client.complete(
                messages=[
                    SystemMessage(content="Bạn là trợ lý học tập thân thiện. Trả lời ngắn gọn, súc tích."),
                    UserMessage(content=req.question)
                ],
                model="DeepSeek-V3", 
                max_tokens=800
            )
            
            return {
                "agent_type": "DeepSeek-V3",
                "thought_process": "",
                "answer": response.choices[0].message.content
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))