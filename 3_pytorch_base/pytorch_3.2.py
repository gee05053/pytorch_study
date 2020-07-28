import torch
import pickle
import matplotlib.pyplot as plt

broken_image = torch.FloatTensor(
    pickle.load(open('/home/kimbyeongjo/바탕화면/pytorch/broken_image_t.p','rb'),
                encoding='latin1'))

plt.imshow(broken_image.view(100,100))
plt.show()
