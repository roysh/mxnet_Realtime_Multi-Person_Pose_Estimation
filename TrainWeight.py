from modelCPM import *
from config.config import config

class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, prefix, carg_params={}, caux_params = {},begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, 19, 46, 46)),
        ('partaffinityglabel', (batch_size, 38, 46, 46)),
        ('heatweight', (batch_size, 19, 46, 46)),
        ('vecweight', (batch_size, 38, 46, 46))])
   
        self.init_params(arg_params = carg_params, aux_params=caux_params,
                         allow_missing=True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))    
        print (prefix)
        for i in range(num_epoch):    
            train_data.reset()    
            self.save_checkpoint(prefix+ "final",i)
            for nbatch_index_,next_data_batch in enumerate(train_data):
                self.forward(next_data_batch, is_train=True)       # compute predictions  
                self.backward()   
                self.update()           
                if nbatch_index_ % 100 == 0:
                    self.save_checkpoint(prefix,nbatch_index_)
                if nbatch_index_ % 10 == 0:
                    prediction=cmodel.get_outputs()
                    sumerror = 0
                    print 'iteration: ', nbatch_index_            
                    for i in range(int(len(prediction)/2)):
                        lossiter = prediction[i*2 + 1].asnumpy()              
                        cls_loss = np.sum(lossiter)/batch_size
                        sumerror = sumerror + cls_loss
                        print cls_loss,
                    print ""
                    for i in range(int(len(prediction)/2)):
                        lossiter = prediction[i*2 + 0].asnumpy()              
                        cls_loss = np.sum(lossiter)/batch_size
                        sumerror = sumerror + cls_loss
                        print cls_loss,
                    print(nbatch_index_,"N/C",sumerror)
                    print ""            
        
        
batch_size = 10
    
cocodata = cocoIterweightBatch('pose_io/data.json',
                               'data', (batch_size, 3, 368,368),
                               ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                               [(batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                                (batch_size, 19, 46, 46), (batch_size, 38, 46, 46)],
                               batch_size
                             )
# cocodata = mx.io.PrefetchingIter(cocodata)

sym = poseSymbol()
cmodel = poseModule(symbol=sym, context=[mx.gpu(1),mx.gpu(4)],
                    label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight'])

prefix = 'model/vggpose'
testsym, newargs, aux_params = mx.model.load_checkpoint("model/vggposefinal", 6)


starttime = time.time()
cmodel.fit(cocodata, num_epoch = config.TRAIN.num_epoch, batch_size = batch_size, prefix = prefix, carg_params = newargs,caux_params = aux_params)
cmodel.save_checkpoint(prefix, config.TRAIN.num_epoch)
endtime = time.time()

print 'cost time: ', (endtime-starttime)/60
