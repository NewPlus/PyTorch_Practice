kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print(kor_text.split())

from konlpy.tag import Okt
tokenizer = Okt()
# 한국어 형태소로 나눈 경우 1
print(tokenizer.morphs(kor_text))
# 한국어 형태소로 나눈 경우 2
print(tokenizer.morphs(u'단독입찰보다 복수입찰의 경우'))
# 한국어 명사로 나눈 경우
print(tokenizer.nouns(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
# 한국어 구문으로 나눈 경우
print(tokenizer.phrases(u'날카로운 분석과 신뢰감 있는 진행으로'))
# 한국어 품사 태깅의 경우
print(tokenizer.pos(u'이것도 되나욬ㅋㅋ'))
# normalize tokens=True 정규화(되나욬 -> 되나요)
print(tokenizer.pos(u'이것도 되나욬ㅋㅋ', norm=True))
# stem tokens=True 어간 추출(되나요 -> 되다)
print(tokenizer.pos(u'이것도 되나욬ㅋㅋ', norm=True, stem=True))