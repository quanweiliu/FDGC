import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import classification_report,cohen_kappa_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import json
from datetime import datetime
import time
from models import FDGC 
from loadData import data_pipe

parser = argparse.ArgumentParser(description='FDGC')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='E:\HSI_Classification\data_preprocess\DataPipe\config\config_1.yaml')
parser.add_argument('--save-name-pre', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('--print-config', action='store_true', default=False)
parser.add_argument('--print-data-info', action='store_true', default=False)
parser.add_argument('--plot-loss-curve', action='store_true', default=False)
parser.add_argument('--show-results', action='store_true', default=False)
parser.add_argument('--save-results', action='store_true', default=True)
args = parser.parse_args()  # running in command line

if args.save_name_pre == '':
    args.results_dir = datetime.now().strftime("%Y%m%d-%H%M")
# args.results_dir

config = yaml.load(open("E:\HSI_Classification\ZZ_FDGC\config\config.yaml", "r"), 
                        Loader=yaml.FullLoader)
# config
dataset_name = config["data_input"]["dataset_name"]
patch_size = config["data_input"]["patch_size"]

num_components = config["data_transforms"]["num_components"]
batch_size = config["data_transforms"]["batch_size"]
remove_zero_labels = config["data_transforms"]["remove_zero_labels"]

max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
lb_smooth = config["network_config"]["lb_smooth"]
num_nodes = config["network_config"]["num_nodes"]

log_interval = config["result_output"]["log_interval"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]

# data_pipe.set_deterministic(seed = 666)
train_loader, test_loader, train_label, test_label = data_pipe.get_data(model_name="FDGC", 
                            path_config=args.path_config, print_config=args.print_config, 
                            print_data_info=args.print_data_info)

net = FDGC(input_channels=num_components, num_nodes=(np.max(test_label)+1)*num_nodes, num_classes=np.max(test_label)+1, patch_size=patch_size).to(args.device)
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothingCrossEntropy(smoothing=lb_smooth)
# criterion = SoftTargetCrossEntropy()
# criterion = nn.MultiLabelSoftMarginLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[max_epoch // 2, (5 * max_epoch) // 6], gamma=0.1)

def train(net, max_epoch, criterion, optimizer, scheduler):
  best_loss = 9999
  train_losses = []
  net.train()

  for epoch in range(1, max_epoch+1):
    correct = 0
    for data, target in train_loader:
      data = data.to(args.device)
      target = target.to(args.device)

      optimizer.zero_grad()
      output = net(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
    scheduler.step()
    train_losses.append(loss.cpu().detach().item())
    
    if epoch % log_interval == 0:
      print('Train Epoch: {}\tLoss: {:.6f} \tAccuracy: {:.6f}'.format(epoch,  loss.item(),  correct / len(train_loader.dataset)))
    if loss.item() < best_loss:
      best_loss = loss.item()
      torch.save(net.state_dict(), path_weight + 'model.pth')
      torch.save(optimizer.state_dict(), path_weight + 'optimizer.pth')
  return train_losses

tic1 = time.time()
train_losses = train(net, max_epoch, criterion, optimizer, scheduler)
toc1 = time.time()

def test(net):
  net.eval()
  test_losses = []
  test_preds = []
  test_loss = 0
  correct = 0
  net.load_state_dict(torch.load(path_weight + 'model.pth'))

  with torch.no_grad():
    for data, target in test_loader:
      data = data.to(args.device)
      target = target.to(args.device)
      output = net(data)
      
      test_loss += criterion(output, target).item()
      test_pred = output.data.max(1, keepdim=True)[1]
      correct += test_pred.eq(target.data.view_as(test_pred)).sum()

      test_label = torch.argmax(output, dim=1)
      test_preds.append(test_label.cpu().numpy().tolist())
  test_losses.append(test_loss)
  
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} \
        ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  return test_losses, test_preds

tic2 = time.time()
test_losses, test_preds = test(net)
toc2 = time.time()

if args.plot_loss_curve:
    fig = plt.figure()
    plt.plot(range(max_epoch), train_losses, color='blue')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

# show results
y_pred_test = [j for i in test_preds for j in i]
classification = classification_report(test_label, y_pred_test, digits=4)
kappa = cohen_kappa_score(test_label, y_pred_test)

training_time = toc1 - tic1
testing_time = toc2 - tic2
# print(training_time, testing_time)

if args.show_results:
    print(classification, "kappa", kappa)
    
if args.save_results:
    end_result = {"classification":[], "kappa":[], "training_time":[], "testing_time":[]}

    end_result["classification"] = classification
    end_result["kappa"] = kappa
    end_result["training_time"] = training_time
    end_result["testing_time"] = testing_time

    # create a new file
    if not os.path.exists(path_result):
        os.mkdir(path_result)

    # dump args
    with open(path_result + args.results_dir + "-" + dataset_name + '.json', 'w') as fid:
        config.update(args.__dict__)
        config.update(end_result)
        json.dump(config, fid, indent=2)



