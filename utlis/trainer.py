import os 
import time
import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt



# train
def train(net, max_epoch, train_loader, test_loader, criterion, optimizer, scheduler, args):
  best_loss = 9999
  best_acc = 0
  train_losses = []


  for epoch in range(1, max_epoch+1):
    net.train()
    train_correct = 0
    tic1 = time.time()

    for data, target in train_loader:
      data = data.to(args.device)
      target = target.to(args.device)
      optimizer.zero_grad()
      output = net(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      pred = output.data.max(1, keepdim=True)[1]
      train_correct += pred.eq(target.data.view_as(pred)).sum()
      
    scheduler.step()
    train_losses.append(loss.cpu().detach().item())
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    epoch_time = time.time() - tic1

    if epoch % args.log_interval == 0:
        test_accuracy, test_time = validation(net, test_loader, args)

    print("epoch", epoch,
        "train_accuracy", round(train_accuracy.item(), 4), 
        "test_accuracy", round(test_accuracy.item(), 4)
        )

    if loss.item() < best_loss:
      best_loss = loss.item()
      torch.save({"epoch": epoch,
                  "model": net.state_dict(),
                  "optimizer": optimizer.state_dict()},
                  args.result_dir + "/best_model_loss.pth")
      # print("save best loss weights at epoch", epoch)

    # # 按照道理说，这里应该用train_acc, 但是为了精度，我选择 test_acc
    # if test_accuracy.item() > best_acc:
    #   best_acc = train_accuracy.item()
    #   torch.save({"epoch": epoch,
    #               "model": net.state_dict(),
    #               "optimizer": optimizer.state_dict()},
    #               args.result_dir + "/best_model_acc.pth")
      # print("save best acc weights at epoch", epoch)
    
    # save
    # with open(os.path.join(args.result_dir, "log.csv"), 'a+', encoding='gbk') as f:
    #     row=[["epoch", epoch, 
    #           "train num", args.train_num,
    #             "loss", round(loss.item(), 4), 
    #             "train acc", round(train_accuracy.item(), 4),
    #             "test acc", round(test_accuracy.item(), 6),
    #             "training time", round(epoch_time, 4),
    #             "test time", round(test_time, 4)
    #             ]]
        # write=csv.writer(f)
        # for i in range(len(row)):
        #     write.writerow(row[i])
  return train_losses, train_accuracy, test_accuracy, epoch_time, test_time

# validation 
def validation(net, test_loader, args):
    net.eval()
    test_preds = []
    test_correct = 0
    tic2 = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target= data.to(args.device), target.to(args.device)
            output = net(data)
            test_pred = output.data.max(1, keepdim=True)[1]
            test_correct += test_pred.eq(target.data.view_as(test_pred)).sum()
            test_label = torch.argmax(output, dim=1)
            test_preds.append(test_label.cpu().numpy().tolist())
            test_accuracy = 100. * test_correct / len(test_loader.dataset)
    test_time = time.time() - tic2

    return test_accuracy, test_time


