import os 
import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt


def draw(label, name, scale: float = 4.0, dpi: int = 400, save_img=True):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)

def test(net, criterion, data_loader, args, groundTruth=None, visulation=False):
    net.eval()
    test_losses = []
    test_preds = []
    correct = 0
    # net.load_state_dict(torch.load(args.path_weight + 'model.pth'))

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(args.device)
            target = target.to(args.device)
            output = net(data)
            
            test_loss = criterion(output, target).item()
            # test_pred = output.data.max(1, keepdim=True)[1]
            test_pred = torch.argmax(output, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
        
    test_accuracy = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                      np.mean(test_losses), correct, len(data_loader.dataset), test_accuracy))

  
    if visulation and groundTruth.any() != None:
        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0)
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy.item(), 4)) + "_knn_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if groundTruth[i][j] == 0:
                    predict_labels[i][j] = 0

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy.item(), 4)) + "_knn_label"))    


    return test_losses, test_preds, test_accuracy