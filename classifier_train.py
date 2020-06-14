'''
classifier_train.py

Author - Max Elliott

Training routine for proposed auxiliary classifier, when training only the
classifier. Can also train the dimensional classifier which was never used in
the thesis.

Command line arguments:
    --checkpoint -c : Directory to load load model checkpoint from if desired
    --num_emos -n   : Number of emotional categories to classify (only for
                      categorical classifier)
    --evaluate -e   : Run the loaded model in testing mode instead
'''

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import yaml
import argparse
import librosa

from utils import audio_utils
import stargan.my_dataset as my_dataset
import stargan.classifiers as classifiers
from stargan.my_dataset import get_filenames
from train_main import make_weight_vector

import torchvision
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device used: ", device)

def save_checkpoint(state, filename='./checkpoints/cls_checkpoint.ckpt'):

    print("Saving a new best model")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)  # save checkpoint


def load_checkpoint(model, optimiser, filename='./checkpoints/cls_checkpoint.ckpt'):

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    epoch = checkpoint['epoch']

    return epoch

def train_model(model, optimiser, train_data_loader, val_data_loader, loss_fn,
                model_type='cls', epochs=1, print_every=1, var_len_data = False, start_epoch = 1):

    model = model.to(device=device) # move the model parameters to CPU/GPU

    print("Training model type: ", model_type)
    best_model_score = 0.  #best f1_score for saving checkpoints

    for e in range(start_epoch, epochs+1):

        total_loss = 0

        for t, (x, y) in enumerate(train_data_loader):
            model.train()  # put model to training mode

            if(var_len_data):
                x_real = x[0].to(device = device).unsqueeze(1)
                x_lens = x[1].to(device = device)
            else:
                x_real = x.to(device=device, dtype=torch.float)

            y = y[:, 0].to(device=device, dtype=torch.float)

            optimiser.zero_grad()
            # print(x_real.size())
            predictions = model(x_real, x_lens)

            if(model_type == 'dim'):
                y_val = y[:,0].long()
                y_aro = y[:,1].long()
                y_dom = y[:,2].long()
#                 print(y_val.size())
#                 print("predictions[0]:", predictions[0].size())
#                 print("predictions[1]:", predictions[1].size())
                loss_val = loss_fn(predictions[0].float(), y_val)
                loss_aro = loss_fn(predictions[1].float(), y_aro)
                loss_dom = loss_fn(predictions[2].float(), y_aro)

                loss = loss_val + loss_aro + loss_dom

            else:
                loss = loss_fn(predictions.float(), y.long())

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

            # print("Epoch ", e, ", iteration", t, " done.")

        if t % print_every == 0:

            print(f'| Epoch: {e:02} | Train Loss: {total_loss:.3f}')

            acc, f1, UAR = test_model(model, val_data_loader,
                                 var_len_data=var_len_data,
                                 model_type=model_type)

#             log_writer.add_scalar('f1', f1)
#             log_writer.add_scalar('lr', optimiser.state_dict()['param_groups'][0]['lr'])

            print("Accuracy = ",acc*100,"%")
            print(f"Macro-f1 score =", f1)
#             print(f"UA-Recall =", UAR)
            print()

            if model_type == 'cls':

                if f1 > best_model_score:

                    print(f"######################## New best model. f1 = {f1: .3f} ########################")
                    best_model_score = f1

                    state = {
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                            'loss_fn': loss_fn}
                    save_checkpoint(state)


def test_model(model, test_loader, var_len_data=False, model_type='cls'):

    model = model.to(device=device)
    model.eval()

    actual_preds = torch.rand(0).to(device = device, dtype = torch.long)
    actual_preds_val = torch.rand(0).to(device = device, dtype = torch.long)
    actual_preds_aro = torch.rand(0).to(device = device, dtype = torch.long)
    # actual_preds_dom = torch.rand(0).to(device = device, dtype = torch.long)

    total_y = torch.rand(0).to(device = device, dtype = torch.long)
    total_y_val = torch.rand(0).to(device = device, dtype = torch.long)
    total_y_aro = torch.rand(0).to(device = device, dtype = torch.long)
    # total_y_dom = torch.rand(0).to(device = device, dtype = torch.long)

    for i, (x, y) in enumerate(test_loader):

        if var_len_data:
            x_real = x[0].to(device = device).unsqueeze(1)
            x_lens = x[1].to(device = device)
        else:
            x_real = x.to(device=device, dtype=torch.float)


        y = y[:,0].to(device=device, dtype=torch.long)

        preds = model(x_real, x_lens)

        if model_type == 'dim':
            y_val = y[:,0].long()
            y_aro = y[:,1].long()
            y_dom = y[:,2].long()

            preds_val = torch.max(preds[0], dim = 1)[1]
            preds_aro = torch.max(preds[1], dim = 1)[1]
            # preds_dom = torch.max(preds[2], dim = 1)[1]

            actual_preds_val = torch.cat((actual_preds_val, preds_val), dim=0)
            actual_preds_aro = torch.cat((actual_preds_aro, preds_aro), dim=0)
            # actual_preds_dom = torch.cat((actual_preds_dom, preds_dom), dim=0)

            total_y_val = torch.cat((total_y_val, y_val), dim=0)
            total_y_aro = torch.cat((total_y_aro, y_aro), dim=0)
            # total_y_dom = torch.cat((total_y_dom, y_dom), dim=0)

        else:
            preds = torch.max(preds, dim = 1)[1]

            actual_preds = torch.cat((actual_preds, preds), dim=0)
            total_y = torch.cat((total_y, y), dim=0)



    if model_type == 'dim':

        # print(actual_preds_val[0:100])
        # print(actual_preds_aro[0:100])
        # print(actual_preds_dom[0:100])
        print(actual_preds_val.size()[0], "total validation predictions.")

        acc_val = accuracy_score(total_y_val.cpu(), actual_preds_val.cpu())
        acc_aro = accuracy_score(total_y_aro.cpu(), actual_preds_aro.cpu())
        # acc_dom = accuracy_score(total_y_dom.cpu(), actual_preds_dom.cpu())

        UAR_val = recall_score(total_y_val.cpu(), actual_preds_val.cpu(), average = 'weighted')
        UAR_aro = recall_score(total_y_aro.cpu(), actual_preds_aro.cpu(), average = 'weighted')
        # UAR_dom = recall_score(total_y_dom.cpu(), actual_preds_dom.cpu(), average = 'weighted')
#         print(f"UAR_val = {UAR_val: .3f}, UAR_aro = {UAR_aro: .3f}")

        f1_val = f1_score(total_y_val.cpu(), actual_preds_val.cpu(), average = 'macro')
        f1_aro = f1_score(total_y_aro.cpu(), actual_preds_aro.cpu(), average = 'macro')
        # f1_dom = f1_score(total_y_dom.cpu(), actual_preds_dom.cpu(), average = 'macro')

        return [acc_val, acc_aro], [f1_val,f1_aro], [UAR_val, UAR_aro]

    else:

        print(actual_preds.size()[0], "total validation predictions.")
        print(actual_preds[0:100])

        acc = accuracy_score(total_y.cpu(), actual_preds.cpu())
        f1 = f1_score(total_y.cpu(), actual_preds.cpu(), average = 'macro')
        UAR = recall_score(total_y.cpu(), actual_preds.cpu(), average = 'weighted')

    return acc, f1, UAR


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Training loop for classifier only.')
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Directory of checkpoint to resume training from")
    parser.add_argument("-n", "--num_emos", type=int, default=3,
                        help="Number of emotions to classify")
    parser.add_argument("-e", "--evaluate", action='store_true',
                        help="False = train, True = evaluate model")
    parser.add_argument("--epochs", type=int, help='Number epochs of training.', default=50)
    args = parser.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # num_classes = 2
    n_epochs = args.epochs
    hidden_size = 128
    input_size = 36
    num_layers = 2

    config = yaml.load(open('./config.yaml', 'r'))

    # MAKE TRAIN + TEST SPLIT
    mel_dir = os.path.join(config['data']['dataset_dir'], "world")

    files = get_filenames(mel_dir)
    num_emos = args.num_emos
    label_dir = os.path.join(config['data']['dataset_dir'], 'labels')
    files = [f for f in files if np.load(label_dir + "/" + f + ".npy")[0] < num_emos]
    # files = [f for f in files if np.load(label_dir + "/" + f + ".npy")[1] in [9, 8, 7, 6]]
    files = my_dataset.shuffle(files)

    train_test_split = config['data']['train_test_split']
    split_index = int(len(files)*train_test_split)
    train_files = files[:split_index]
    test_files = files[split_index:]

    print(len(train_files))
    print(len(test_files))

    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)

    batch_size = 16

    train_loader, test_loader = my_dataset.make_variable_dataloader(train_dataset,
                                                                    test_dataset,
                                                                    batch_size=batch_size)

    # torch.Tensor([4040./549, 4040./890,
                # 4040./996, 4040./1605]).to(device)
    emo_loss_weights = make_weight_vector(files, config['data']['dataset_dir']).to(device)

    print("Making model")

    model = nn.DataParallel(classifiers.Emotion_Classifier(input_size, hidden_size,
                     num_layers = num_layers, num_classes = num_emos, bi = True))
    # model = nn.DataParallel(classifiers.Dimension_Classifier(input_size, hidden_size,
                     # num_layers = num_layers, bi = True))
    optimiser = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.000001)

    loss_fn = nn.CrossEntropyLoss(weight = emo_loss_weights)
    epoch = 1

    if args.checkpoint is not None:
        epoch = load_checkpoint(model, optimiser, args.checkpoint)
        print("Model loaded, resuming from epoch", epoch, ".")


    if not args.evaluate:
        print("Training model.")
        train_model(model, optimiser, train_loader, test_loader, loss_fn, model_type='cls',
                    epochs = n_epochs, var_len_data = True, start_epoch=epoch)
    else:

        test_model(model, test_loader, model_type='cls', var_len_data=True)
        print("No training. Model loaded in evaluation mode.")
