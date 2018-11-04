# Simple retrieval bot

Prepare tap separated pairwise corpus such like `(question, answer)`

```python
from retrieval_bot import PairCorpus

corpus_path = 'YOUR_CORPUS_PATH'
corpus = PairCorpus(corpus_path)
print('num pairs = {}'.format(len(corpus)))
# num pairs = 5903206
```

Train unsupervised tokenizer for Korean text.

```python
from retrieval_bot.tokenizer import CohesionScore

cohesion_path = 'YOUR_COHESION_MODEL_PATH'

corpus.iter_pair = False
cohesion_trainer = CohesionScore(debug=False)
scores = cohesion_trainer.train_and_scores(corpus)
cohesion_trainer.save(cohesion_path)
# cohesion_trainer.load(cohesion_path)
scores = cohesion_trainer.scores()
```

Test example of tokenization 

```python
from retrieval_bot.tokenizer import MaxScoreTokenizer

tokenizer = MaxScoreTokenizer(scores=scores)
tokenizer.tokenize('아니지금어디냐고?왜아직도안와?')
# ['아니', '지금', '어디', '냐', '고?', '왜', '아직도', '안와?']
```

Vectorzing send messages and reply messages.

```python
from retrieval_bot.vectorizer import Vectorizer

vectorizer_path = 'YOUR_VECTORIZER_MODEL_PATH'

corpus.limit_pairs = -1 # use all pairs
corpus.iter_pair = False
vectorizer = Vectorizer(tokenizer=tokenizer, min_tf=2)
vectorizer = vectorizer.fit(corpus)
vectorizer.save(vectorizer_path)
# vectorizer.load(vectorizer_path)
len(vectorizer.vocabulary_)
```

Build `send to reply` graph.

```python
from retrieval_bot.db import Send2Reply

graph_path = 'YOUR_GRAPH_MODEL_PATH'

corpus.iter_pair = True
send2reply = Send2Reply()
send2reply.train(corpus)
send2reply.save(graph_path)
# send2reply.load(graph_path)
len(send2reply)
```

Load matrix of send & reply messages

```python
from scipy.io import mmwrite
from scipy.io import mmread

send_x = vectorizer.transform(send2reply.sends)
reply_x = vectorizer.transform(send2reply.replies)
mmwrite(send_x_path, send_x)
mmwrite(reply_x_path, reply_x)

# send_x = mmread(send_x_path).tocsr()
# reply_x = mmread(reply_x_path).tocsr()
```

Build send message index

```python
from retrieval_bot.db import FullSearchIndexer

send_indexer = FullSearchIndexer(send_x)
```

Set unknown message responser

```python
from retrieval_bot.engine import DefaultMessage

default_message = DefaultMessage()
for _ in range(5):
    print(default_message.get_random_message())
```

    음...
    음...
    어...
    무슨 말이에요?
    응??

Evaluate pmi scorer

```python
from retrieval_bot.evaluator import TermPairEvaluator

evaluator = TermPairEvaluator(evaluator_model_path)
for send_term in '뭐먹 어디'.split():
    send_term_idx = vectorizer.encode_a_doc_to_bow(send_term)
    for reply_term in '피자 치킨 지하철 사당역'.split():
        reply_term_idx = vectorizer.encode_a_doc_to_bow(reply_term)
        score = evaluator.evaluate(send_term_idx, reply_term_idx)
        print('{} - {} : {}'.format(send_term, reply_term, score))
```

    뭐먹 - 피자 : 3.08618742336944
    뭐먹 - 치킨 : 3.852132624607777
    뭐먹 - 지하철 : 0
    뭐먹 - 사당역 : 0
    어디 - 피자 : 0
    어디 - 치킨 : 0.4396905736037917
    어디 - 지하철 : 2.259871019456051
    어디 - 사당역 : 0.2413882193124051

Ready for retrieval based response engine

```python
from retrieval_bot.engine import InstanceBasedRetrievalEngine

vectorizer.verbose = False
send2reply.verbose = False

engine = InstanceBasedRetrievalEngine(
    vectorizer = vectorizer,
    send2reply = send2reply,
    send_indexer = send_indexer,
    reply_x = reply_x,
    evaluator = evaluator
)
```

Test default responser

```python
engine.default_message.get_random_message()
# '음...'
```

Test responser

```python
engine.process('지금 어디냥? ')
# ['냐옹', '가구있엉', '안알랴줌', '집이댜', '카페']
```