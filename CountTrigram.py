import numpy as np
names = [name.strip() for name in open('names.txt','r').readlines()]
names = ['!'+name+'!' for name in names]
trigrams={}

for name in names:
    for i in range(len(name)-2):
        trigrams[name[i]+name[i+1]]=[]

for name in names:
    for i in range(len(name)-2):
        trigrams[name[i:i+2]].append(name[i+2])


unique_chars = list(set((letter for name in names for letter in name)))
prob = np.zeros((len(trigrams),len(unique_chars)),dtype=np.float16)
conversion_2 = {char: i for i, char in enumerate(trigrams.keys())}   
convBack_2 = {value: key for key, value in conversion_2.items()}
conversion_1 = {char: i for i, char in enumerate(unique_chars)}
convBack_1 = {value: key for key, value in conversion_1.items()} 

for key, values in trigrams.items():
    for value in values:
        prob[conversion_2[key],conversion_1[value]]+=1

prob /= np.sum(prob,axis=1)[:,None]

def generate_name():
    name = '!'+ np.random.choice(unique_chars)
    while name[-1] != '!':
        name += convBack_1[np.random.choice(len(unique_chars),p=prob[conversion_2[name[-2:]]])]
    return name[1:-1]

print('Trigram statistical model generated names:')
for i in range(10):
    print(generate_name())