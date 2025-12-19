import os
import re
from sentence_transformers import SentenceTransformer
import faiss
import pickle as pkl
model = SentenceTransformer('shibing624/text2vec-base-chinese')
#处理文件
def document():
    with open("../data/question.txt", encoding="utf-8") as f:
        question = f.read()
        chunk=question.replace("\r","")
        chunk=re.sub(r'\n+','\n',chunk)
        chunk=re.split(r'\n(?=\d+\.)',chunk)
        answer =[]
        ques =[]
        for item in chunk:
            item=item.strip()
            line = item.split("\n",1)
            if len(line)>1:
                ques_ = line[0].strip()
                answer_ = line[1].strip()

                ques.append(ques_)
                answer.append(item)
            else:
                ques.append(item)
                answer.append(item)
        return ques,answer


if __name__ == "__main__":
    ques, answer = document()
    embedding = model.encode(ques,show_progress_bar=True).astype('float32')
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding)
    faiss.write_index(index,"university")
    with open("university.pkl", "wb") as f:
        pkl.dump(answer, f)
