# word2vec를 이용하여 문장의 주제 예측하기

## word2vec (kor2vec)

[kor2vec](https://github.com/dongjun-Lee/kor2vec)를 참고하였습니다.

## 문장의 주제 예측하기

### Concept

언어를 벡터로 표현할 수 있다는 점을 이용하여, 주어진 주제들 중 어떤 주제와 관련된 문장인지 분류합니다.

1. 문장을 구성하는 형태소로 쪼갠다.
    ```python
    for s_idx, sentence in enumerate(test_sentence):
        pos_list = okt.pos(sentence, norm=True)
    ```
2. 각 형태소가 얼마나 주제와 가까운지 거리를 구한다.
    ```python
        for pos in pos_list:
            for idx, subject in enumerate(subjects):
                subject_pos = okt.pos(subject, norm=True)
                try:
                    distance = pos_vectors.distance(str(pos).replace(" ", ""), str(subject_pos[0]).replace(" ", ""))
                except KeyError:
                    distance = 0
                 subjects_distance[idx] += distance
    ```
3. 거리의 평균을 계산한 뒤 SoftMax 연산을 한다.
    ```python
    def soft_max(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
 
    percentage = soft_max(- subjects_distance)
    ```
4. 현재문장이 직전 문장에서 얼마나 변화하였지를 측정한다.
    ```python
    subjects[np.argmax(percentage - 0.5 * last_subject)]
    ```
    > 아 근데 오늘 프로 야구 개막전이구나. 그러면 그냥 짜장면 먹으면서 야구 중계나 봐야겠다.
    
    단순하게 이 문장을 하나씩만 분석하게 되면, 두 번째 문장에서 '야구 중계' 라는 단어 때문에 주제가
     '음식'이 아니라 '스포츠'가 될 수도 있습니다. 반면 위 수식에 의해 계산하게 되면 다음과 같은 결과가 나옵니다.
    
    ```
    1. 아 근데 오늘 프로 야구 개막전이구나
    [Subject: 스포츠, Percentage 0.38]
    
    2. 그러면 그냥 짜장면 먹으면서 야구 중계나 봐야겠다.
    [Subject: 음식,	  Percentage 0.31]
    ```

## Expectation

시나리오 기반의 챗봇이 사용자가 말하고자 하는 주제를 인지하여, 보다 정확한 답변을 낼 수 있도록 사용할 수 있습니다.
현재는 [http://mmlab.snu.ac.kr/~djlee/pos.vec](http://mmlab.snu.ac.kr/~djlee/pos.vec)의 pre-trained weight 를 사용하고 있는데
사용 장소에 어울리는 새로운 데이터로 학습을 하고 사용하게 된다면 더 좋은 성능이 기대됩니다.

## Pros and Cons

### Pros

1. 간편하게 예측할 주제를 추가할 수 있다.
2. 각 주제에 대한 데이터셋을 필요로 하지 않는다.

### Cons

딥러닝으로 문장과 주제를 직접 학습하는 것이 효율과 성능이 훨씬 좋을 것이다. 
