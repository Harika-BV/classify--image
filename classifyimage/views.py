from django.shortcuts import render
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from classifyimage import utils
import base64
from .forms import ImageUploadForm
    

def home(request):
    return render(request, 'index.html')


def getClassOfImage(image):
    
    net = utils.Net()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH="classifyimage/models/cifar_net.pth"

    net.load_state_dict(torch.load(PATH))
    net.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    #img = Image.open("D:/HARIKA_BV/2021 July-Aug/Photo Enhancement Using AI/data2/archive/DIV2K_train_HR/DIV2K_train_HR/0064.png")
    img = Image.open(image)
    img = img.resize((32,32))
    input = transform(img)
    input = input.unsqueeze(0)

    output = net(input)
    print(output)
    _, predicted = torch.max(output, 1)
    print('Predicted: ', classes[predicted[0]])
    return classes[predicted[0]]


def index(request):
    image_uri = None
    predicted_label = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            # get predicted label
            try:
                predicted_label = getClassOfImage(image)
            except RuntimeError as re:
                print(re)

    else:
        form = ImageUploadForm()
        

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }


    return render(request, 'index.html', context)

    
    

    