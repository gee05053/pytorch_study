import torch
import pickle
import matplotlib.pyplot as plot

broken_image = torch.FloatTensor(
    pickle.load(open('/home/kimbyeongjo/pytorch_study/3_pytorch_base/broken_image_t.p','rb'),
                encoding='latin1')) #손상된 이미지 불러옴

plot.imshow(broken_image.view(100,100))
plot.show()

## 랜덤으로 만든 이미지를 이미지를 손상시키는 코드에 넣어 손상된 이미지 와 비슷하게 만들면
## 랜덤으로 만든 이미지가 원래 이미지라고 볼 수 있음


def weird_function(x, n_iter=5) : #이미지를 손상시키는 코드
    h=x
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter) :
        zero_tensor= torch.tensor([1.0*0])
        h_l = torch.cat((zero_tensor, h[:-1]),0)
        h_r = torch.cat((h[1:], zero_tensor),0)
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i%2 == 0 :
            h = torch.cat( (h[h.shape[0]//2:], h[:h.shape[0]//2]),0)
    return h

def distance_loss(hypothesis, broken_image) : #가설 텐서와 오염된 이미지 사이의 오차
    return torch.dist(hypothesis, broken_image)

random_tensor = torch.randn(10000, dtype = torch.float) #무작위 가설 텐서 생성

lr = 0.8 #learning rate

for i in range(0, 20000) :
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad() : #autogade를 false로 바꿈
        random_tensor = random_tensor - lr*random_tensor.grad
    if i%1000 == 0 :
        print('Loss at {} = {}'.format(i, loss.item()))

plot.imshow(random_tensor.view(100,100).data)
plot.show()
