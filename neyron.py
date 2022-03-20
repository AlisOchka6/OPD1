import numpy as np

x1 = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
x2 = np.array([-1.0, -2.0, -3.0, -1.0, -2.0, -3.0, -1.0, -2.0, -3.0])
x = np.append(x1, x2)
x = np.reshape(x, [len(x1), 2])
y = np.transpose(sum(np.transpose(x)))
for i in range(len(x1)):
    print(x[i,0],'+',x[i,1],'=',y[i])

from pybrain3.datasets import SupervisedDataSet
train_data = SupervisedDataSet(2, 1)

for i in range(len(x1)):
    train_data.addSample(x[i, :], y[i])

print(train_data)

from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.structure import TanhLayer
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

nn = buildNetwork(2, 2, 1, bias=True, hiddenclass=TanhLayer)

trainer = BackpropTrainer(nn, train_data)

for epoch in range(500):
   trainer.train()

print('training')
print(trainer.testOnData(dataset=train_data))
trainer.testOnData(dataset=train_data, verbose=True)


print(nn.activate([-2, 3]))
print(nn.activate([-0.5, 2.5]))
for i in range(len(x1)):
    print(x1[i], '+', x2[i], '=', nn.activate(x[i,:]))

import pybrain3
#если возникает ошибка см. - https://github.com/AlexProgramm/pybrain3/issues/1
networkwriter.NetworkWriter.writeToFile(nn, 'network.xml')
from pybrain3.tools.xml import networkreader
from pybrain3.tools.xml import networkwriter

net = networkreader.NetworkReader.readFrom('network.xml')


print('loaded: 2 + 3 ->', net.activate([-2, 3]))
#https://www.tutorialspoint.com/pybrain/pybrain_quick_guide.htm
#https://www.tutorialspoint.com/pybrain/pybrain_tutorial.pdf
