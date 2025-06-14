from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# 환경 변수 또는 직접 입력
os.environ["OPENAI_API_KEY"] = "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"

# FastAPI 앱 생성
app = FastAPI()

# 데이터 클래스: 질문 받기용
class QueryRequest(BaseModel):
    query: str

# 1. 임베딩 및 벡터DB 로딩
print("FAISS 인덱스 로딩 중...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

db = FAISS.load_local("embeddings/vehicle_manual_index", embedding_model, allow_dangerous_deserialization=True)

# 2. LLM (OpenAI GPT-4o)
print("OpenAI LLM 설정 중...")
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# 3. LangChain Retrieval QA 체인
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = db.as_retriever(search_kwargs={"k": 10}),
    chain_type_kwargs={
        "prompt": PromptTemplate(
            input_variables=["context", "question"],
            template="""
                    당신은 차량 매뉴얼 안내 전문가입니다.
                    
                    사용자가 입력한 질문이 비공식적이고 모호하고, 공식 차량 메뉴얼에서 찾을 수 없을 거 같은 단어여도, 단어의 의미를 유추해서
                    공식 차량 매뉴얼에서 검색하기 좋은 형태의 질문으로 바꿔주세요.
                    예:
                    엉뜨 -> 좌석 히터

                    그 질문과 관련된 정보를 문서 내용을 기준으로 자연스러운 문장으로 응답하세요.
                    답변은 최대 3문장으로 간결하고 핵심만 담아 운전자가 이해하기 쉬운 형태로 답변해주세요. (답변을 제공할 땐 오로지 답변만 해주세요.)
                    정보가 '전혀' 없을 경우에만 당신이 알고 있는 선에서 정확한 답변을 제공하세요.

                    문서:
                    {context}

                    질문:
                    {question}

                    답변:"""
                            )
                        }
                    )

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query
    relevant_docs = db.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(query)
    print(f"🔍 검색된 문서 개수: {len(relevant_docs)}")
    
    for i, doc in enumerate(relevant_docs):
        print(f"\n📄 문서 {i+1} 내용:\n{doc.page_content[:500]}")
        
    print(f"질문 수신: {query}")

    try:
        answer_dict = qa_chain.invoke(query)
        answer = answer_dict["result"]
        answer = answer.replace("**", "").replace("\n", " ").strip()
        print(f"응답 생성 완료: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": str(e)}