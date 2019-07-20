from dataloader import TrainPhotos,TestPhotos
from torchvision.utils import save_image

if __name__ == '__main__':
    TrainData=TrainPhotos('./train')
    TestData=TestPhotos('./test')
    print(TrainData[2],'\n')
    print(TestData.__getitem__(2).size())  
    save_image(TrainData[2], './dc_img/aaa.png')
   