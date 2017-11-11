#encoding=utf-8
'''
Created on Nov 6, 2017

@author: kohill


Test a model wheather it works.
'''

import mxnet as mx
import numpy as np

save_prefix = "model/vggpose"
import cv2
import matplotlib.pyplot as plt
from modelCPM import CPMModel_test
class DataBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = [data]
        self.label = 0
        self.pad = pad
epoch =3217
batch_size = 1
sym  = CPMModel_test()
# mx.visualization.plot_network(CPMModel(),
# shape = {"data":(1,3,368,368),
#          "heatmaplabel":(1,15,46,46),
#          "heatweight":(1,15,46,46),
#          "partaffinityglabel":(1,26,46,46),
#          "vecweight":(1,26,46,46),
#          }                              
#                               
#                               ).view()
sym_load, newargs,aux_args = mx.model.load_checkpoint(save_prefix, epoch)
model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in [6]],                        
                    label_names=None)
model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],for_training = True)
model.init_params(arg_params=newargs, aux_params=aux_args, allow_missing=False,allow_extra=False)

img = cv2.imread("sample_image/test2.jpg")
img = cv2.resize(img,(368,368))
imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1))/256 - 0.5
imgs_batch = DataBatch(mx.nd.array([imgs_transpose[:,:,:]]), 0)
model.forward(imgs_batch)
result = model.get_outputs()
heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
heatmap = cv2.resize(heatmap, (368, 368), interpolation=cv2.INTER_CUBIC)   

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(heatmap[:,:,-1])
plt.show()
