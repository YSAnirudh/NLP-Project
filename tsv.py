import pandas as pd
import random

a = pd.read_csv('knowledge_graph.csv').to_numpy().tolist()

train = []
test = []
valid = []
for i in range(int(len(a) / 10)):
    ran = random.randint(0, len(a)-1)
    valid.append(a[ran][0:3])
    #print(len(a), ran)
    del a[ran]
    ran = random.randint(0, len(a)-1)
    #print(len(a), ran)
    test.append(a[ran][0:3])
    del a[ran]
#print("Train")
for i in range(len(a)):
    train.append(a[i][0:3])

print("Train:", len(train))
print("Test:", len(test))
print("Valid:", len(valid))

# print("train: ")
# for i in range(len(train)):
#     print(train[i])
# print("test: ")
# for i in range(len(test)):
#     print(test[i])
# print("valid: ")
# for i in range(len(valid)):
#     print(valid[i])

pd.DataFrame(train).to_csv('hello/hello-train.txt', sep='\t', header=False, index=False, index_label=None)
pd.DataFrame(test).to_csv('hello/hello-test.txt', sep='\t', header=False, index=False, index_label=None)
pd.DataFrame(valid).to_csv('hello/hello-valid.txt', sep='\t', header=False, index=False, index_label=None)

