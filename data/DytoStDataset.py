import os
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
# from tqdm import tqdm
npyf = 15

osize = [256,256]
Tran = transforms.Resize(osize, interpolation=transforms.InterpolationMode.BICUBIC)

########################
# A is Dynamic/ Label/ Content
# B is Static/ Ground Truth/Real "Image"/ Style 
########################

class DytoStKITTIDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DytoStKITTIDataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/kitti/range_image/scan"

        # dataroot = "../prash/data"
        self.dir_dynamic = os.path.join(dataroot, 'dynamic')
        self.dir_static = os.path.join(dataroot, 'static')
        
        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])
        
        
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,5:45] for i in range(2)])))
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,:,::4] for i in range(2)])))
        # import pdb; pdb.set_trace()
        # self.B = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,:,:,::4] for i in range(2)])))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,0] for i in tqdm(range(16))]))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,:1,:,::4] for i in range(2)]))
        

        self.A = torch.from_numpy(np.random.random((5, 3, 64, 128))).float()
        self.B = torch.from_numpy(np.random.random((5, 3, 64, 128))).float()
        # self.C = torch.from_numpy(np.random.random((4541, 1, 64, 128))).float()

        
        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code

        self.A = Tran(self.A)
        self.B = Tran(self.B)

        print("A SHAPE: ", self.A.shape)
        
        """
        st1 = []
        dy1 = []

        npy = [i for i in range(11)]
        npy.remove(8)
        # skip = [6,2,6,2,1,4,2,2,3,2]
        skip = [3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1]

        for i in npy:
            # (64 128)
            # dy = np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8]
            # st = np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8]
            # ma = np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8]
            print("Loading data: ", i)
            # (16 128)
            dy = np.load(self.dir_dynamic+f"/{i}.npy")[:,:,::4,::8]
            st = np.load(self.dir_static+f"/{i}.npy")[:,:,::4,::8]

            st1.append(st[::skip[i]])
            dy1.append(dy[::skip[i]])



        st1 = np.concatenate(st1, axis=0)
        dy1 = np.concatenate(dy1, axis=0)
        
        
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy1)
        self.B = torch.from_numpy(st1)
        """
        


        
        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]

        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B}

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class TestDytoStKITTIDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDytoStKITTIDataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/kitti/range_image/scan"
        NUM_DATA = 9 # number of npy files in train data

        # dataroot = "../prash/data"
        self.dir_dynamic = os.path.join(dataroot, 'dynamic')
        self.dir_static = os.path.join(dataroot, 'static')
        
        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])

        
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,5:45] for i in range(2)])))
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,:,::4] for i in range(2)])))
        # import pdb; pdb.set_trace()
        # self.B = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,:,:,::4] for i in range(2)])))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,0] for i in tqdm(range(16))]))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,:1,:,::4] for i in range(2)]))
        

        # self.A = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.B = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.C = torch.from_numpy(np.random.random((4541, 1, 64, 128))).float()

        
        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code


        
        # (64 128)
        # dy = np.load(self.dir_dynamic+f"/8.npy")[:,:,:,::8]
        # st = np.load(self.dir_static+f"/8.npy")[:,:,:,::8]
        # ma = np.load(self.dir_mask+f"/08dy-mask.npy")[:,:,::8]

        # (16 128)
        dy = np.load(self.dir_dynamic+f"/8.npy")[:,:,::4,::8]
        st = np.load(self.dir_static+f"/8.npy")[:,:,::4,::8]

        print(f"Shapes: dy {dy.shape} st {st.shape}")   
               
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy)
        self.B = torch.from_numpy(st)

        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]


        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B}

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')


class DytoStCARLADataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DytoStCARLADataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/DSLR/lidar/"

        
        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])

        
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,5:45] for i in range(2)])))
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,:,::4] for i in range(2)])))
        # import pdb; pdb.set_trace()
        # self.B = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,:,:,::4] for i in range(2)])))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,0] for i in tqdm(range(16))]))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,:1,:,::4] for i in range(2)]))
        

        # self.A = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.B = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.C = torch.from_numpy(np.random.random((4541, 1, 64, 128))).float()

        
        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code

        st1 = []
        dy1 = []


        npy = [i for i in range(3)]
        

        for i in npy:
            # (64 128)
            dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,:,::4])
            st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,:,::4])
            print("Loading data: ", i)
            # (16 128)
            # dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,::4,::4])
            # st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,::4,::4])
            # ma = np.load(self.dir_mask+f"/mask{i}.npy")[:,::4,::4]


            st1.append(st)
            dy1.append(dy)




        st1 = np.concatenate(st1, axis=0)
        dy1 = np.concatenate(dy1, axis=0)


        print("ST1 shape:", st1.shape)
        print("DY1 shape:", dy1.shape)

        
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy1)
        self.B = torch.from_numpy(st1)
        
        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]


        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B,}

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')


class TestDytoStCARLADataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDytoStCARLADataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/DSLR/lidar/"

        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])

        
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,5:45] for i in range(2)])))
        # self.A = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,:,:,::4] for i in range(2)])))
        # import pdb; pdb.set_trace()
        # self.B = torch.from_numpy(self.from_polar_np(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,:,:,::4] for i in range(2)])))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,0] for i in tqdm(range(16))]))
        # self.C = torch.from_numpy(np.concatenate([np.load(self.dir_mask+f"/s{i}.npy")[:,:1,:,::4] for i in range(2)]))
        

        # self.A = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.B = torch.from_numpy(np.random.random((4541, 3, 64, 128))).float()
        # self.C = torch.from_numpy(np.random.random((4541, 1, 64, 128))).float()

        
        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code

        st1 = []
        dy1 = []


        npy = [npyf]
        

        for i in npy:
            # (64 128)
            # dy = np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8]
            # st = np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8]
            # ma = np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8]
            print("Loading data: ", i)
            # (16 128)
            dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,:,::4])
            st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,:,::4])



            st1.append(st)
            dy1.append(dy)




        st1 = np.concatenate(st1, axis=0)
        dy1 = np.concatenate(dy1, axis=0)


        print("ST1 shape:", st1.shape)
        print("DY1 shape:", dy1.shape)

        
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy1)
        self.B = torch.from_numpy(st1)
        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]


        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B
                }

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')


class DytoStAtiDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(DytoStAtiDataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/DSLR/ARD/lidar"

        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])


        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code

        st1 = []
        dy1 = []


        npy = [i for i in range(3)]
        

        for i in npy:
            # (64 128)
            dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,:,::8])
            st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,:,::8])
            print("Loading data: ", i)
            # (16 128)
            # dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,::4,::4])
            # st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,::4,::4])
            # ma = np.load(self.dir_mask+f"/mask{i}.npy")[:,::4,::4]


            st1.append(st)
            dy1.append(dy)




        st1 = np.concatenate(st1, axis=0)
        dy1 = np.concatenate(dy1, axis=0)


        print("ST1 shape:", st1.shape)
        print("DY1 shape:", dy1.shape)

        
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy1)
        self.B = torch.from_numpy(st1)
        
        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]


        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B}

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')


class TestDytoStAtiDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDytoStAtiDataset, self).__init__()
        dataroot = "/home/prashant/scratch/data/DSLR/ARD/lidar"

        # self.ABC_paths = sorted([os.path.join(self.dir_ABC, name) for name in os.listdir(self.dir_ABC)])

        
        #original dataset: ABC is RGB | GT | mask => dynamic | static | mask

        ###################################################### SKIP data code

        st1 = []
        dy1 = []

        npy = [3]
        

        for i in npy:
            # (64 128)
            # dy = np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8]
            # st = np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8]
            # ma = np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8]
            print("Loading data: ", i)
            # (16 128)
            dy = self.from_polar_np(np.load(dataroot+f"/d{i}.npy")[:,:,:,::8])
            st = self.from_polar_np(np.load(dataroot+f"/s{i}.npy")[:,:,:,::8])



            st1.append(st)
            dy1.append(dy)




        st1 = np.concatenate(st1, axis=0)
        dy1 = np.concatenate(dy1, axis=0)


        print("ST1 shape:", st1.shape)
        print("DY1 shape:", dy1.shape)

        
        # For 64,128
        #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_dynamics = np.concatenate([np.load(self.dir_dynamic+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 3, 64, 1024) -> (4541, 3, 64, 128)
        # all_statics = np.concatenate([np.load(self.dir_static+f"/{i}.npy")[:,:,:,::8] for i in range(NUM_DATA)]) 
        # #(4541, 64, 1024) -> (4541, 64, 128)
        # all_masks = np.concatenate([np.load(self.dir_mask+f"/0{i}dy-mask.npy")[:,:,::8] for i in range(NUM_DATA)])
         
        self.A = torch.from_numpy(dy1)
        self.B = torch.from_numpy(st1)
        
        
    def __getitem__(self, index):
        
        A = self.A[index]
        B = self.B[index]


        # A = np.concatenate([np.load(self.dir_lidar+f"/s{i}.npy")[:,5:45] for i in range(16)])
        # B = torch.tensor(np.concatenate([np.load(self.dir_lidar+f"/d{i}.npy")[:,5:45] for i in range(16)]))
        # C_ = torch.tensor(cv2.dilate(C_.view(256, 256).numpy(), np.ones((3, 3), dtype=np.uint8), 2)).view(1, 256, 256)
        return {'label': A,
                'image': B}

    def __len__(self):
        return self.A.shape[0]

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')