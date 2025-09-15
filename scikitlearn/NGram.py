from collections import defaultdict
import random

corpus="you're very nice you're very kind you're very awesome".split()

bigrams=defaultdict(list)
for i in range(len(corpus)-1):
    bigrams[corpus[i]].append(corpus[i+1])

word="you're"    
sentence = [word]
for _ in range(10):  # max length
    if not bigrams[word]:  # if no next word
        break
    word = random.choice(bigrams[word])
    if word == "<END>":
        break
    sentence.append(word)
print("generated (bigram):", " ".join(sentence))    