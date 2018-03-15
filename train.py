from __future__ import print_function
import DenseNet
import DenseNet_input

train, valid, test = DenseNet_input.getEmotionImage()

model = DenseNet.DenseNet(train, valid, test_data=test, input_size=42, num_class=7, init_lnr=1e-1, depth=3 * 12 + 4,
                          bc_mode=True, reduction=0.5, weight_decay=1e-5, total_blocks=3,
                          growth_rate=32, current_save_folder='./save/WideDenseNet-BC2/', reduce_lnr=[70, 143, 180],
                          logs_folder='./summary/WideDenseNet-BC2', valid_save_folder='./save/WideDenseNet-BC2/valid/',
                          max_to_keep=0, snapshot_test=False)
model.train(num_epoch=200, batch_size=32)

