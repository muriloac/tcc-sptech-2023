import torch
import pandas as pd
import warnings
import cxr_dataset as CXR
from torch.utils.data import DataLoader
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES):
    """
    Faz previsões para o conjunto de teste e calcula as AUCs usando um modelo previamente treinado.

    Args:
        data_transforms: transformações do torchvision para pré-processar as imagens brutas; as mesmas transformações de validação
        model: densenet-121 do torchvision previamente ajustado ao conjunto de treinamento
        PATH_TO_IMAGES: caminho onde as imagens do NIH podem ser encontradas
    Returns:
        pred_df: dataframe contendo previsões individuais e verdade absoluta para cada imagem de teste
        auc_df: dataframe contendo AUCs agregadas por tuplas de treinamento/teste
    """

    # calcular previsões em lotes de 16, pode ser reduzido se sua GPU tiver menos RAM
    BATCH_SIZE = 16

    # definir o modelo para o modo de avaliação; necessário para previsões adequadas dada a utilização do batchnorm
    model.train(False)

    # criar dataloader
    dataset = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['val'])
    dataloader = DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # criar dataframes vazios
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterar sobre o dataloader
    for i, data in enumerate(dataloader, start=0):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # obter previsões e valores verdadeiros para cada item no lote
        for j, (actual, pred) in enumerate(zip(true_labels, probs)):

            thisrow = {"Image Index": dataset.df.index[BATCH_SIZE * i + j]}
            truerow = {"Image Index": dataset.df.index[BATCH_SIZE * i + j]}

            # iterar sobre cada entrada no vetor de previsão; cada uma corresponde a uma etiqueta individual
            for k, label in enumerate(dataset.PRED_LABEL):
                thisrow[f"prob_{label}"] = pred[k]
                truerow[label] = actual[k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        if i % 10 == 0:
            print(str(i * BATCH_SIZE))

    auc_df = pd.DataFrame(columns=["label", "auc"])

    # calcular AUCs
    for column in true_df:

        if column not in [
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
                'Hernia']:
            continue
        actual = true_df[column]
        pred = pred_df[f"prob_{column}"]
        thisrow = {"label": column, "auc": np.nan}
        try:
            thisrow["auc"] = sklm.roc_auc_score(
                actual.values.astype(int), pred.values)
        except BaseException:
            print("não é possível calcular a AUC para " + str(column))
            import traceback
            traceback.print_exc()
        auc_df = auc_df.append(thisrow, ignore_index=True)

    with open("results/preds.csv", "w") as f:
        pred_df.to_csv(f, index=False)
    with open("results/aucs.csv", "w") as f:
        auc_df.to_csv(f, index=False)
    return pred_df, auc_df