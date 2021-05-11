import torch
import torchaudio
import audiofolder as ds
import torch.nn as nn
import os
import time

def mean(list):
    return sum(list)/len(list)

mnist_set = ds.AudioFolder("D:/audio_classification_data/ESC-50-master/ESC-50-master/sorted/mnist")
mnist_loader = torch.utils.data.DataLoader(mnist_set,shuffle=True)

noise_set = ds.AudioFolder("D:/audio_classification_data/ESC-50-master/ESC-50-master/sorted/noises")
noise_loader = torch.utils.data.DataLoader(noise_set,shuffle=True)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv1d(in_channels,out_channels,3,padding=1,stride=2)
        self.activation = nn.LeakyReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return(self.activation(self.norm(self.conv(x))))

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.input = ConvModule(1,32)
        self.layer1 = ConvModule(32,32)
        self.layer2 = ConvModule(32,32)
        self.layer3 = ConvModule(32,64)
        self.layer4 = ConvModule(64,64)
        self.layer5 = ConvModule(64,64)
        self.layer6 = ConvModule(64,128)
        self.layer7 = ConvModule(128,128)
        self.layer8 = ConvModule(128,128)
        self.ir1 = None
        self.ir2 = None
        self.ir3 = None
        self.ir4 = None
        self.ir5 = None
        self.ir6 = None
        self.ir7 = None
        self.ir8 = None

    def forward(self,x):
        x = self.input(x)
        self.ir1 = x
        x = self.layer1(x)
        self.ir2 = x
        x = self.layer2(x)
        self.ir3 = x
        x = self.layer3(x)
        self.ir4 = x
        x = self.layer4(x)
        self.ir5 = x
        x = self.layer5(x)
        self.ir6 = x
        x = self.layer6(x)
        self.ir7 = x
        x = self.layer7(x)
        self.ir8 = x
        x = self.layer8(x)

        return x

class MnistTail(nn.Module):
    def __init__(self):
        super(MnistTail,self).__init__()
        self.classifier = nn.Linear(128,10)
        self.activation = nn.Softmax()
    def forward(self,x):
        x = torch.mean(x,dim=2).squeeze()
        x = self.classifier(x)
        return x

class NoiseTail(nn.Module):
    def __init__(self):
        super(NoiseTail,self).__init__()
        self.classifier = nn.Linear(128,50)
        self.activation = nn.Softmax()
    def forward(self,x):
        x = torch.mean(x,dim=2).squeeze()
        x = self.classifier(x)
        return x

#want to end up with 128d vector
lossnet = VGGNet()
lossnet.train()
lossnet.cuda()

mnist_tail = MnistTail()
mnist_tail.train()
mnist_tail.cuda()

noise_tail = NoiseTail()
noise_tail.train()
noise_tail.cuda()

loss_fn = nn.CrossEntropyLoss()
criterion = torch.optim.Adam(lossnet.parameters(), lr = .001, betas = (.5,.999))

state = torch.load("lossnet_param.pth")
lossnet.load_state_dict(state)

#freeze VGG, only train classifiers
#for param in lossnet.parameters():
#    param.requires_grad = False

mnist_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

keep_training = True
counter = 1
training_losses = []

#there is something real wierd going on here
while keep_training:
    epoch_losses = []
    print("epoch started!")
    start_time = time.time()
    for mnist_label, mnist_example in enumerate(mnist_loader):

        criterion.zero_grad()
        #print(mnist_label)
        #print(mnist_example)
        #do an mnist
        mnist_wav, fs = mnist_example
        mnist_gt_label = int(fs)
        #print(mnist_gt_label)
        tensor_mnist_label = (torch.ones(1) * mnist_gt_label).long()

        params = lossnet(mnist_wav[0].cuda())
        out_vec = mnist_tail(params)
        loss = loss_fn(out_vec.unsqueeze(0),fs.long().cuda())
        loss.backward()
        criterion.step()

        epoch_losses.append(loss.item())

        #then do a noise
        criterion.zero_grad()

        noise_example = iter(noise_loader).next()
        #print(noise_example)
        noise_wav,fs = noise_example
        #print(noise_example)
        tensor_noise_label = (torch.ones(1) * int(fs)).long()

        params = lossnet(noise_wav[0].cuda())
        out_vec = noise_tail(params)
        loss = loss_fn(out_vec.unsqueeze(0),tensor_noise_label.cuda())
        loss.backward()
        criterion.step()

        epoch_losses.append(loss.item())

    print("Epoch {} finished! Average Loss: {}, Total Time: {}".format(counter,mean(epoch_losses),time.time()-start_time))
    counter = counter + 1

    if counter > 3:
        if mean(training_losses[-2:]) < mean(epoch_losses):
            keep_training = False
            print("training finished!")

    training_losses.append(mean(epoch_losses))
#print("finished training")

#torch.save(lossnet.state_dict(), 'lossnet_param.pth')

#print("checking!")

#torch.save(mnist_tail.state_dict(),"mnist_tail.pth")
#torch.save(noise_tail.state_dict(),"noise_tail.pth")

#for mnist_label, mnist_example in enumerate(noise_loader):
#    mnist_wav, fs = mnist_example
#    tensor_noise_label = (torch.ones(1) * int(fs)).long()
#    print(tensor_noise_label)

#    params = lossnet(mnist_wav[0].cuda())
#    params = torch.mean(params,dim=2).squeeze()
#    print(params)
    #out_vec = mnist_tail(params)
    #out_vec = torch.nn.Softmax()(out_vec)
    #print(out_vec)
