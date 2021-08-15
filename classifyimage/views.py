from django.shortcuts import render
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import base64
from .forms import ImageUploadForm
    

def home(request):
    return render(request, 'index.html')



def getClassOfImage(image):
    
    net = torchvision.models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs,100)

    classes = ('apple', 'aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom','oak_tree', 'orange','orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf', 'woman','worm')

    PATH="classifyimage/models/ckpt.pth"
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    
    net.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    img = Image.open(image)
    img = img.resize((32,32))
    input = transform(img)
    input = input.unsqueeze(0)

    output = net(input)
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

