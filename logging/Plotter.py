import os
import pandas as pd
import matplotlib.pyplot as plt
os.chdir('Dec-03-2019_1047')

Data = pd.read_csv('4000.csv')

Epoch = 1
supl = 0
usupl = 0
trainacc = 0
count = 1
sup_loss = []
usup_loss = []
train_acc = []
for i in range(len(Data)):
    if Data.Epoch[i] != Epoch:
        supl /= count
        usupl /= count
        trainacc /= count
        sup_loss.append(supl)
        usup_loss.append(usupl)
        train_acc.append(trainacc)
        Epoch = Data.Epoch[i]
        count = 1
        supl = 0
        usupl = 0
        trainacc = 0
    supl += Data.Supervised_Loss[i]
    usupl += Data.Unsupervised_Loss[i]
    trainacc += Data.Training_Accuracy[i]
    count +=1

val_acc = Data.Validation_Accuracy.dropna().values


N = 10
cumsum, moving_aves = [0], []
for i, x in enumerate(val_acc, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

val_acc = moving_aves

N = 10
cumsum, moving_aves = [0], []

for i, x in enumerate(train_acc, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

train_acc = moving_aves

plt.plot(train_acc)
plt.plot(val_acc)

plt.show()

plt.plot(sup_loss)
plt.plot(usup_loss)
plt.show()