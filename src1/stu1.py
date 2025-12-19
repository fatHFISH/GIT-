import pickle
from openai import OpenAI
import os
import faiss
from sentence_transformers import SentenceTransformer
get_key=os.getenv('DASHSCOPE_API_KEY')
RAG = SentenceTransformer('shibing624/text2vec-base-chinese')
index = faiss.read_index("university")
llm=OpenAI(
    api_key=get_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
with open("university.pkl","rb") as f:
    unviersity = pickle.load(f)
def read(query,k=3):
    query_vec = RAG.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k)
    context = "\n".join([unviersity[idx] for idx in indices[0]])
    return context
def use_(user_query):
    text = read(user_query)
    prompt = f"""
        你是一个学校的智能助手，名字叫“叶同学”。
        切记学校全称为广州城市理工学院或者广城理
        【已知信息】：
        {text}
        【用户问题】：
        {user_query}
        你叫叶同学，是一个学习小助手。你可以根据提供的【参考资料】来回答同学关于广州城市理工学院的问题。如果参考资料中没有相关信息，请诚实告知。
        """
    return prompt
messages=[{
                "role": "system", "content": "你叫叶同学，你是一个学习小助手，你的任务是帮住同学完成学习任务"
            }]



first_text="你好啊，我是叶同学，欢迎你使用我，我能帮你解决问题呢！"

print(first_text)
while True:
    x=input()
    is_answer = False
    answer_content = ""
    if x=="exit":
        break
    else:
        messages.append({"role": "user", "content": f"{use_(x)}"})
        completion = llm.chat.completions.create(
            model="qwen-plus",
            temperature=1.0,
            stream=True,
            stream_options={"include_usage": True},
            messages=messages,
            extra_body={"enable_thinking":True}
        )
        if not is_answer:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
        for chunk in completion:
            if not chunk.choices:
                continue
            #处理思考过程
            if hasattr(chunk.choices[0].delta,"reasoning_content") and chunk.choices[0].delta.reasoning_content is not None:
                    print(chunk.choices[0].delta.reasoning_content,end="",flush=True)
            #处理正式回答
            if hasattr(chunk.choices[0].delta,"content") and chunk.choices[0].delta.content is not None:
                if not is_answer:
                    print("\n" + "=" * 20 + "回复过程" + "=" * 20 + "\n")
                    is_answer=True
                if is_answer:
                    answer_content+=chunk.choices[0].delta.content
    messages.append({"role": "assistant", "content": f"{answer_content}"})
    print(answer_content)