data=[(["Technical","Senior","Excellent","Good","Urban"],"Yes"),(["Technical","Junior","Excellent","Good","Urban"],"Yes"),(["Non-Technical","Junior","Average","Poor","Rural"],"No"),(["Technical","Senior","Average","Good","Rural"],"No"),(["Technical","Senior","Excellent","Good","Rural"],"Yes")]
num_attributes=5
S=["Ø"]*num_attributes
G=[["?"]*num_attributes]
def is_consistent(hypothesis,example):
    for h,e in zip(hypothesis,example):
        if h!="?" and h!=e:
            return False
    return True
def more_general(h1,h2):
    for x,y in zip(h1,h2):
        if x!="?" and (y=="?" or x!=y):
            return False
    return True
def prune_G(G):
    pruned=[]
    for g in G:
        if not any(more_general(other,g) and other!=g for other in G):
            pruned.append(g)
    return pruned
for index,(example,label) in enumerate(data,start=1):
    print(f"\nExample {index}: {example} -> {label}")
    if label=="Yes":
        for i in range(num_attributes):
            if S[i]=="Ø":
                S[i]=example[i]
            elif S[i]!=example[i]:
                S[i]="?"
        G=[g for g in G if is_consistent(g,example)]
    else:
        new_G=[]
        for g in G:
            if is_consistent(g,example):
                for i in range(num_attributes):
                    if g[i]=="?":
                        if S[i]!="?" and S[i]!=example[i]:
                            new_hypothesis=g.copy()
                            new_hypothesis[i]=S[i]
                        new_G.append(new_hypothesis)
            else:
                new_G.append(g)
        G=new_G
    G=prune_G(G)
    print("Specific Boundary S:",S)
    print("General Boundary G:",G)
print("\nFINAL RESULT")
print("Final Specific Hypothesis (S):",S)
print("Final General Hypotheses (G):",G)

