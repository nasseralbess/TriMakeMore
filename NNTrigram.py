import numpy as np
import torch
names = [name.strip() for name in open('names.txt','r').readlines()]
names = ['!'+name+'!' for name in names]
trigrams={}
for name in names:
    for i in range(len(name)-2):
        trigrams[name[i]+name[i+1]]=[]
unique_chars = sorted(list(set((letter for name in names for letter in name))))

for name in names:
    for i in range(len(name)-2):
        trigrams[name[i:i+2]].append(name[i+2])


conversion_1 = {char: i for i, char in enumerate(unique_chars)}
convBack_1 = {i: char for i, char in enumerate(unique_chars)}
combs = [i+j for i in unique_chars for j in unique_chars[1:]]
conversion_2 = {comb: i for i, comb in enumerate(combs)}
convBack_2 = {i: comb for i, comb in enumerate(combs)}

X,y = [],[] 
for key in trigrams.keys():
    for value in trigrams[key]:
        feature = np.zeros((len(combs)))
        feature[conversion_2[key]]=1
        X.append(feature)
        y.append(conversion_1[value])

X,y = np.array(X),np.array(y)
idx = np.random.permutation(len(X))
X,y = X[idx],y[idx]
X,y = torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.int32)

#for reproducibility
gen = torch.Generator().manual_seed(69)
W = torch.randn((len(combs),len(unique_chars)), generator=gen, requires_grad=True)

rate = 30 #arbitrary rate that seems to reduce the loss steadily given the simplicity of the model
for i in range(1500): #This seems like a reasonable number of iterations, as the model was still improving after 1000 iterations
    logits = X @ W
    probs = torch.nn.functional.softmax(logits,dim=1)
    loss = -torch.mean(probs[torch.arange(len(y)),y].log())
    loss.backward()
    with torch.no_grad():
        W -= rate * W.grad
        W.grad.zero_()

    if i % 100 == 0:
        print(f'Iteration {i}, loss: {loss.item()}')
        rate *= 0.9

def generate_name_nn():
    name = '!'+ np.random.choice(unique_chars)
    while name[-1] != '!':
        feature = np.zeros((len(combs)))
        feature[conversion_2[name[-2:]]]=1
        feature = torch.tensor(feature, dtype=torch.float32)
        logits = feature @ W
        probs = torch.nn.functional.softmax(logits, dim=0)
        ToAdd = torch.multinomial(probs,1).item()
        name += convBack_1[ToAdd]
    return name[1:-1]

print('NN generated names:')
for i in range(10):
    print(generate_name_nn())