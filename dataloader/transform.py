from torchvision.transforms import Compose, ToTensor, Lambda

def diffusion_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
        )
