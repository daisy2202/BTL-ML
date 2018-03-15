import DenseNet
import DenseNet_input
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

train, valid, test = DenseNet_input.getEmotionImage()

model = DenseNet.DenseNet(train, valid, test_data=train, input_size=42, num_class=7, init_lnr=1e-1, depth=40,
                          bc_mode=True, reduction=0.5, weight_decay=1e-4, total_blocks=3,
                          growth_rate=32, current_save_folder='./save/WideDenseNet-BC/', reduce_lnr=[70, 143, 180],
                          logs_folder='./summary/WideDenseNet-BC', valid_save_folder='./save/WideDenseNet-BC/valid/',
                          max_to_keep=0, snapshot_test=True)
score, label = model.get_score(batch_size=1, save_file=[0.7059, 0.7042, 0.7017])
label = np.argmax(np.array(label), axis=1)
score = TSNE(n_components=2).fit_transform(score)
score0 = score[label == 0, :]
score1 = score[label == 1, :]
score2 = score[label == 2, :]
score3 = score[label == 3, :]
score4 = score[label == 4, :]
score5 = score[label == 5, :]
score6 = score[label == 6, :]

# X_embedded = TSNE(n_components=2).fit_transform(score)
# print(X_embedded.shape)
plt.plot(score0[:,0], score0[:,1], 'b.')
plt.plot(score1[:,0], score1[:,1], 'g.')
plt.plot(score2[:,0], score2[:,1], 'r.')
plt.plot(score3[:,0], score3[:,1], 'c.')
plt.plot(score4[:,0], score4[:,1], 'm.')
plt.plot(score5[:,0], score5[:,1], 'y.')
plt.plot(score6[:,0], score6[:,1], 'k.')
plt.show()
