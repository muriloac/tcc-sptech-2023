from __future__ import print_function, division

# pytorch imports
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils

# image / graphics imports
from skimage import io, transform
from PIL import Image
from pylab import *
import seaborn as sns
from matplotlib.pyplot import show 
from sklearn.metrics import confusion_matrix

# data science
import numpy as np
import scipy as sp
import pandas as pd

# import other modules
from copy import deepcopy
import cxr_dataset as CXR
import eval_model as E

def calc_cam(x, label, model):
    """
    função para gerar um mapa de ativação de classe correspondente a um tensor de imagem torch

    Args:
        x: o arquivo tensor pytorch 1x3x224x224 que representa o NIH CXR
        label: rótulo fornecido pelo usuário que você deseja obter o mapa de ativação de classe; deve estar na lista FINDINGS
        model: densenet121 treinado em dados NIH CXR

    Returns:
        cam_torch: tensor torch 224x224 contendo mapa de ativação
    """
    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']

    if label not in FINDINGS:
        raise ValueError(
            str(label) +
            "is an invalid finding - please use one of " +
            str(FINDINGS))

    # encontra o índice para o rótulo; isso corresponde ao índice da saída da rede
    label_index = next(
        (x for x in range(len(FINDINGS)) if FINDINGS[x] == label))

    # define densenet_last_layer class so we can get last 1024 x 7 x 7 output
    # of densenet for class activation map
    # instancia a classe densenet_last_layer para que possamos obter a última saída 1024 x 7 x 7
    # do densenet para o mapa de ativação de classe
    class densenet_last_layer(torch.nn.Module):
        def __init__(self, model):
            super(densenet_last_layer, self).__init__()
            self.features = torch.nn.Sequential(
                *list(model.children())[:-1]
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            return x

    # instantiate cam model and get output
    # instancia o modelo cam e obtém a saída
    model_cam = densenet_last_layer(model)
    x = torch.autograd.Variable(x)
    y = model_cam(x)
    y = y.cpu().data.numpy()
    y = np.squeeze(y)

    # pull weights corresponding to the 1024 layers from model
    # puxa os pesos correspondentes às 1024 camadas do modelo
    weights = model.state_dict()['classifier.0.weight']
    weights = weights.cpu().numpy()
    
    bias = model.state_dict()['classifier.0.bias']
    bias = bias.cpu().numpy()

    # create 7x7 cam
    # cria uma cam 7x7
    cam = np.zeros((7, 7, 1))
    for i in range(0, 7):
        for j in range(0, 7):
            for k in range(0, 1024):
                cam[i, j] += y[k, i, j] * weights[label_index, k]
    cam+=bias[label_index]

    #make cam into local region probabilities with sigmoid
    # transforma a cam em probabilidades de região local com sigmoid
    cam=1/(1+np.exp(-cam))
    
    label_baseline_probs={
        'Atelectasis':0.103,
        'Cardiomegaly':0.025,
        'Effusion':0.119,
        'Infiltration':0.177,
        'Mass':0.051,
        'Nodule':0.056,
        'Pneumonia':0.012,
        'Pneumothorax':0.047,
        'Consolidation':0.042,
        'Edema':0.021,
        'Emphysema':0.022,
        'Fibrosis':0.015,
        'Pleural_Thickening':0.03,
        'Hernia':0.002
    }
    
    #normalize by baseline probabilities
    # normaliza pelas probabilidades de linha de base
    cam = cam/label_baseline_probs[label]
    
    #take log
    # tira o log
    cam = np.log(cam)
    
    return cam

def load_data(
        PATH_TO_IMAGES,
        LABEL,
        PATH_TO_MODEL,
        POSITIVE_FINDINGS_ONLY,
        STARTER_IMAGES):
    """
    Loads dataloader and torchvision model

    Args:
        PATH_TO_IMAGES: path to NIH CXR images
        LABEL: finding of interest (must exactly match one of FINDINGS defined below or will get error)
        PATH_TO_MODEL: path to downloaded pretrained model or your own retrained model
        POSITIVE_FINDINGS_ONLY: dataloader will show only examples + for LABEL pathology if True, otherwise shows positive
                                and negative examples if false

    Returns:
        dataloader: dataloader with test examples to show
        model: fine tuned torchvision densenet-121
    """

    checkpoint = torch.load(PATH_TO_MODEL, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    del checkpoint
    model.cpu()

    # build dataloader on test
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not POSITIVE_FINDINGS_ONLY:
        finding = "any"
    else:
        finding = LABEL

    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='test',
        transform=data_transform,
        finding=finding,
        starter_images=STARTER_IMAGES)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1)
    
    return iter(dataloader), model


def show_next(dataloader, model, LABEL):
    """
    Plota a imagem do raio-x, o mapa de ativação do raio-x e mostra as probabilidades do modelo para cada achado.

    Args:
        dataloader: dataloader dos raio-x de teste
        model: torchvision densenet-121 ajustado
        LABEL: achado que estamos interessados em ver o mapa de calor
    Returns:
        None (plots output)
    """
    FINDINGS = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']
    
    label_index = next(
        (x for x in range(len(FINDINGS)) if FINDINGS[x] == LABEL))

    # obtém o próximo iterador do dataloader
    try:
        inputs, labels, filename = next(dataloader)
    except StopIteration:
        print("Todos os exemplos foram esgotados - execute as células acima para gerar novos exemplos para revisar")
        return None
        
    # obtém o mapa de calor
    original = inputs.clone()
    raw_cam = calc_cam(inputs, LABEL, model)
    
    # cria previsões para o achado de interesse e todos os achados
    pred = model(torch.autograd.Variable(original.cpu())).data.numpy()[0]
    predx = ['%.3f' % elem for elem in list(pred)]
    
    fig, (showcxr,heatmap) =plt.subplots(ncols=2,figsize=(14,5))
    
    hmap = sns.heatmap(raw_cam.squeeze(),
            cmap = 'viridis',
            alpha = 0.3, # todo o mapa de calor é translúcido
            annot = True,
            zorder = 2,square=True,vmin=-5,vmax=5
            )
    
    cxr=inputs.numpy().squeeze().transpose(1,2,0)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cxr = std * cxr + mean
    cxr = np.clip(cxr, 0, 1)
        
    hmap.imshow(cxr,
          aspect = hmap.get_aspect(),
          extent = hmap.get_xlim() + hmap.get_ylim(),
          zorder = 1) # coloca o mapa sob o mapa de calor
    hmap.axis('off')
    hmap.set_title("P("+LABEL+")="+str(predx[label_index]))
    
    showcxr.imshow(cxr)
    showcxr.axis('off')
    showcxr.set_title(filename[0])
    plt.savefig(str(LABEL+"_P"+str(predx[label_index])+"_file_"+filename[0]))
    plt.show()
    
    preds_concat=pd.concat([pd.Series(FINDINGS),pd.Series(predx),pd.Series(labels.numpy().astype(bool)[0])],axis=1)
    preds = pd.DataFrame(data=preds_concat)
    preds.columns=["Achado","Probabilidade Prevista","Verdadeiro Positivo"]
    preds.set_index("Achado",inplace=True)
    preds.sort_values(by='Probabilidade Prevista',inplace=True,ascending=False)
    
    return preds