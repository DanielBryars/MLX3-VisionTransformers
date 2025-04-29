from torchvision import datasets, transforms
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)

mnist_test = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

if __name__ == '__main__':
    
    print(f"Number of training samples: {len(mnist_train)}")
    print(f"Shape of one sample: {mnist_train[0][0].shape}")  # Should be (1, 28, 28)

    image, label = mnist_train[0]
    plt.imshow(image.squeeze(), #squeeze removes the first dimension
               cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()
    
 