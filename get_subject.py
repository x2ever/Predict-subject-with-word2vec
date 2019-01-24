from konlpy.tag import Okt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


def soft_max(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


pos_vectors = KeyedVectors.load_word2vec_format('pos.vec', binary=False)
okt = Okt()

subjects = ["여행", "스포츠", "애완동물", "음식"]

test_sentence = ["배 고픈데 뭐 먹지?", "라면에 김밥이나 먹어야겠다.", "아니면 그냥 차 타고 춘천가서 닭갈비나 먹을까?",
                 "그 다음엔 설악산 가서 등산해도 좋고", "아 근데 오늘 프로 야구 개막전이구나", "그러면 그냥 짜장면 먹으면서 야구 중계나 봐야겠다.",
                 "우리 강아지 목욕 시켜주면서 슬슬 준비해야겠다."]
last_subject = 1 / len(subjects)
for s_idx, sentence in enumerate(test_sentence):
    pos_list = okt.pos(sentence, norm=True)
    subjects_distance = np.zeros(len(subjects))
    for pos in pos_list:
        for idx, subject in enumerate(subjects):
            subject_pos = okt.pos(subject, norm=True)
            try:
                distance = pos_vectors.distance(str(pos).replace(" ", ""), str(subject_pos[0]).replace(" ", ""))
            except KeyError:
                # print(str(pos).replace(" ", ""))
                distance = 0
            subjects_distance[idx] += distance

    percentage = soft_max(- subjects_distance)
    # print(percentage - 0.5 * last_subject)

    print("\n%s. %s\n[Subject: %s,\t Percentage %.2f]" % (s_idx + 1, sentence,
                                                          subjects[np.argmax(percentage - 0.5 * last_subject)],
                                                          percentage[np.argmax(percentage - 0.5 * last_subject)]))

    last_subject = percentage

