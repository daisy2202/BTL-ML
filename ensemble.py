import numpy as np
import DenseNet
import DenseNet_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train, valid, test = DenseNet_input.getEmotionImage()

model = DenseNet.DenseNet(train, valid, test_data=valid, input_size=42, num_class=7, init_lnr=1e-1, depth=3 * 12 + 4,
                          bc_mode=True, reduction=0.5, weight_decay=1e-4, total_blocks=3,
                          growth_rate=12, current_save_folder='./save/DenseNet-BC/', reduce_lnr=[70, 143, 180],
                          logs_folder='./summary/WideDenseNet-BC', valid_save_folder='./save/DenseNet-BC/valid/',
                          max_to_keep=0, snapshot_test=True)
score, label = model.test_snapshot_ensemble(batch_size=1, save_file=[0.6883, 0.6872, 0.6867])


def PlotConfusionMatrix(conf_arr):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(norm_conf[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '0123456789JKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()


y_pred = []
y_true = []
wrong = np.zeros([7])
for i in range(len(score)):
    true_label = np.argmax(label[i])
    pred = np.argmax(score[i])
    y_true.append(true_label)
    y_pred.append(pred)

conf_matrix = confusion_matrix(y_true, y_pred)
PlotConfusionMatrix(conf_matrix)
