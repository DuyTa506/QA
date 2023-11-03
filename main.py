from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel 
from core import return_answer, load_data, embeddingvector, load_embedding,return_answer_aligned,return_answer_aligned_lite
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from datasets import load_dataset, DatasetDict, load_from_disk
import json
from corrector import check_spell
from pydantic import BaseModel
from datetime import datetime
import pytz
from pyvi import ViUtils
app = FastAPI(debug=True)
embeddings_dataset = None


data_file_path = "./qa_data.json"

class QuestionRequest(BaseModel):
    id_khach: int
    id_conversation: int
    question: str


# Hàm để tính toán embeddings và tạo embeddings_dataset
def load_embedding_dataset():
    global embeddings_dataset
    try:
        embeddings_dataset = load_from_disk("./embeddingbkai")
        embeddings_dataset.add_faiss_index(column="embeddings")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình tính toán embeddings_dataset")

def create_embeddings_dataset():
    global embeddings_dataset
    try:
        processed_dataset = load_from_disk("./qa")
        embeddings_dataset = processed_dataset.map(
            lambda x: {"embeddings": embeddingvector(x["Câu hỏi"]).numpy()[0]},
            remove_columns=['__index_level_0__'],
            keep_in_memory=True
        )
        embeddings_dataset.add_faiss_index(column="embeddings")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình tính toán embeddings_dataset")
# Route để gửi câu hỏi và nhận câu trả lời
@app.post("/api/answers")
def get_answer(question_request: QuestionRequest):
    global embeddings_dataset
    isSuccess = "True"
    answers = []
    scores = []
    try:
        answers, scores = return_answer_aligned_lite(check_spell(((question_request.question)).lower()), 0.6, 0.4, embeddings_dataset)
        if len(answers) == 0:
            isSuccess = "False"
        qa_pairs = []
        for answer, score in zip(answers, scores):
            qa_pair = {"question_exact": question_request.question, "answer": answer, "score": score}
            qa_pairs.append(qa_pair)
        # vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh') 
        # current_time = datetime.now(vietnam_tz).strftime('%H:%M:%S %d-%m-%Y')
        response_data = {
            "isSuccess": isSuccess,
            # "time" : current_time,
            "id_khach": question_request.id_khach,
            "id_conversation": question_request.id_conversation,
            "data": qa_pairs,
        }
        json_qa_pairs = json.dumps(response_data, indent=4, ensure_ascii=False)
        return Response(content=json_qa_pairs, media_type='application/json')
    except Exception as e:
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình xử lý câu hỏi")

# Xử lý ngoại lệ HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(content={"error": str(exc.detail)}, status_code=exc.status_code)

if __name__ == "__main__":
    print("Tính toán embeddings và tạo embeddings_dataset...")
    load_embedding_dataset() 
    import uvicorn 
    uvicorn.run(app,host='0.0.0.0', port = 3000)