from channels.generic.websocket import WebsocketConsumer,JsonWebsocketConsumer
from channels.exceptions import StopConsumer

import cooler
import pandas as pd
import numpy as np

# from rest_framework.renderers import JSONRenderer
# import pickle

import tensorflow as tf
import keras
from keras import optimizers
from label.model import train_discriminative_model,get_TAD_model,mirrorTAD
from label.query_methods import DiscriminativeRepresentationSamplingInteractive,RandomSampling

randomSeed = 444

#class Consumer(WebsocketConsumer):
class Consumer(JsonWebsocketConsumer):
    def websocket_connect(self, message):
        self.ready = False
        self.labelMode = False
        self.accept()

    # def websocket_receive(self, message):
    def receive_json(self, content):
        if content.get('type') == 'pre':
            mode = content.get('mode')
            self.tag = content.get('tag')
            initModel = content.get('initModel')
            testSet = content.get('testSet')
            sharefirstround = content.get('sharefirstround')
            sharefirstroundmodel = content.get('sharefirstroundmodel')
            self.mode = mode
            self.ready = True
            self.testSet = False
            coolstr1 = content.get('mcool') + "::/resolutions/" + str(content.get('resolution'))
            coolstr2 = content.get('mcool')
            if bool(testSet):
                testSetData = np.load(testSet, allow_pickle=True)
                self.testSet = True
                self.testSetData = testSetData
                self.testSetX = testSetData['x']
                self.testSetY = keras.utils.to_categorical(testSetData['y'], num_classes = 2)

            if not initModel:
                self.model = get_TAD_model()
            else:
                print("load initModel")
                from keras.models import load_model
                self.model = load_model(initModel)

            try:
                self.mcool = cooler.Cooler(coolstr1)
            except:
                self.mcool = cooler.Cooler(coolstr2)




            if(mode != 'check'):
                featureDict = pd.read_pickle(content.get('feature'))
                self.features = np.nan_to_num(featureDict['features'])
                self.tads  = featureDict['tads']
                self.next = None
                self.currentTad = None
                self.nextTad = None
                self.labeled_tad = []
                # self.label = []
                label = np.empty(self.features.shape[0])
                label[:] = np.nan
                self.label = label
                self.round_ = 0

                # np.random.seed(1)
                np.random.seed(randomSeed)
                labeled_idx = np.random.choice(self.features.shape[0], 50, replace=False)
                # k-means clustering to chose the best first 20 sample
                # labeled_idx = initSampling(self.features,n=50)
                new_labeled_idx = labeled_idx
                self.new_labeled_idx = new_labeled_idx
                self.tadIndex = np.nditer(new_labeled_idx)
                self.labeled_idx = labeled_idx 
            else:
                self.getInconsistentRes()# update self.tads,label and self.consistentTads
                self.labeled_tad = []
                self.labelMode = True
                self.labeled_idx = []
                self.updateCurrent()
                self.updateNext()

            if  bool(sharefirstround) and bool(sharefirstroundmodel) :
                data = np.load(sharefirstround, allow_pickle=True)
                firstroundLabel = data['y'][:50]
                firstroundTad = data['tad'][:50]
                from keras.models import load_model
                self.model = load_model(sharefirstroundmodel)
                self.label[self.labeled_idx] = firstroundLabel
                for i in firstroundTad:
                    self.labeled_tad.append(i)


            if mode == 'random':
                # self.round_ = 1
                self.query = RandomSampling(self.model, input_shape = 28, num_labels = 2)

                # _first,labeled_idx = self.query.query(self.features, self.label, np.array([]), amount = 50, RandomSeed=randomSeed)
                # labeled_idx = labeled_idx.astype(int)

                # self.labeled_idx = labeled_idx
                # self.new_labeled_idx = labeled_idx
                # self.tadIndex = np.nditer(labeled_idx)

                self.updateCurrentA()
                self.updateNextA()

            if mode == 'active':
                print('DAL mode')
                self.query = DiscriminativeRepresentationSamplingInteractive(self.model, input_shape = 28, num_labels = 2)
                self.updateCurrentA()
                self.updateNextA()
            if mode == 'label':
                print('label mode')
                self.tads  = featureDict['tads'].iterrows()
                self.labeled_tad = []

                self.labelMode = True
                self.labeled_idx = []
                self.blacklist = {chrom:[] for chrom in self.mcool.chromnames}
                self.updateCurrent()
                self.updateNext()
                # self.currentTad = None
                # self.nextTad = None
                
            if sharefirstround and sharefirstroundmodel:
                print('share first round mode')
                # save model
                self.model.save(self.tag + '_' + self.mode + '_model_round' + str(self.round_) + '.h5')
                self.query.update_model(self.model)
                self.round_ += 1

                labeled_idx,new_labeled_idx = self.query.query(self.features, self.label, self.labeled_idx, amount = 50)
                labeled_idx = labeled_idx.astype(int)
                new_labeled_idx = new_labeled_idx.astype(int)
                self.labeled_idx = labeled_idx
                self.new_labeled_idx = new_labeled_idx
                self.tadIndex = np.nditer(new_labeled_idx)
                self.updateCurrentA()
                self.updateNextA()
            # labeled_idx,new_labeled_idx = self.query.query(self.features, self.label, self.labeled_idx, amount = 100)


        if content.get('type') == 'label' and self.ready and not self.labelMode:
                
            label_ = content.get('label')
            # self.label.append(label_)
            self.label[self.currentTadIndex] = label_
            self.labeled_tad.append(self.currentTad)
            if self.next == None:
                self.trainModel()
                # for each round, save model
                self.model.save(self.tag + '_' + self.mode + '_model_round' + str(self.round_) + '.h5')
                self.query.update_model(self.model)
                self.round_ += 1

                labeled_idx,new_labeled_idx = self.query.query(self.features, self.label, self.labeled_idx, amount = 50)
                labeled_idx = labeled_idx.astype(int)
                new_labeled_idx = new_labeled_idx.astype(int)
                self.labeled_idx = labeled_idx
                self.new_labeled_idx = new_labeled_idx
                self.tadIndex = np.nditer(new_labeled_idx)
                self.updateCurrentA()
            else:
                self.send_json(self.next)
                self.currentTad = self.nextTad
                self.currentTadIndex = self.nextTadIndex
                self.next = None
                self.nextTad = None
                self.nextTadIndex = None

            self.updateNextA()

        if content.get('type') == 'label' and self.ready and self.labelMode:
            label_ = content.get('label')
            self.label[self.currentTadIndex] = label_
            self.labeled_idx.append(self.currentTadIndex)
            self.labeled_tad.append(self.currentTad)
            if self.next == None and self.mode == 'check':
                # correctedTads = pd.DataFrame(self.labeled_tad).append(pd.DataFrame(self.consistentTads)) 
                correctedTads = self.finalTads
                correctedLabel = np.concatenate((self.label,self.consistentLabel))
                np.savez(self.tag + 'correctedData.npz',x=self.newX,y=correctedLabel,tad=correctedTads)
                raise StopConsumer()

            self.send_json(self.next)
            self.currentTad = self.nextTad
            self.currentTadIndex = self.nextTadIndex

            self.updateNext()

        if content.get('type') == 'skip' and self.labelMode:
            self.label[self.currentTadIndex] = 0
            self.labeled_idx.append(self.currentTadIndex)
            self.labeled_tad.append(self.currentTad)
            side = content.get('side')
            tad = self.currentTad
            if side == 'left':
                print('skip left border!')
                self.blacklist[tad.chrom].append(tad.bin1)
            if side == 'right':
                self.blacklist[tad.chrom].append(tad.bin2)
            if side == 'both':
                self.blacklist[tad.chrom].append(tad.bin1)
                self.blacklist[tad.chrom].append(tad.bin2)
            nextTad = self.nextTad
            if nextTad.bin1 in self.blacklist[nextTad.chrom] or nextTad.bin2 in self.blacklist[nextTad.chrom]:
                print('skip nextTad')
                self.label[self.nextTadIndex] = 0
                self.labeled_idx.append(self.nextTadIndex)
                self.labeled_tad.append(self.nextTad)

                self.updateCurrent()
                self.updateNext()
            else:
                print('direct next')
                self.send_json(self.next)
                self.currentTad = self.nextTad
                self.currentTadIndex = self.nextTadIndex
                self.updateNext()

        if content.get('type') == 'close':
            x = self.features[self.labeled_idx]
            y = self.label[self.labeled_idx]
            tad = self.labeled_tad
            # np.savez(self.tag + 'labeledData.npz',x=x,y=y,tad=tad)
            np.savez(self.tag + '_' + self.mode + '_labeledData.npz',x=x,y=y,tad=tad)

            if not self.labelMode:
                # self.model.save('my_model.h5')
                # self.model.save(self.tag + '_finalmodel.h5')
                self.model.save(self.tag + '_' + self.mode + '_finalmodel.h5')
            raise StopConsumer()
            

    def websocket_disconnect(self, message):
        x = self.features[self.labeled_idx]
        y = self.label[self.labeled_idx]
        tad = self.labeled_tad
        # np.savez(self.tag + 'labeledData.npz',x=x,y=y,tad=tad)
        np.savez(self.tag + '_' + self.mode + '_labeledData.npz',x=x,y=y,tad=tad)

        if not self.labelMode:
            self.model.save(self.tag + '_' + self.mode + '_finalmodel.h5')
        raise StopConsumer()

        # # self.model.save('my_model.h5')
        # self.model.save(self.tag + '_finalmodel.h5')

        # x = self.features[self.labeled_idx]
        # y = self.label[self.labeled_idx]
        # np.savez(self.tag + 'labeledData.npz',x=x,y=y)
        # # result = {'features':self.features[:len(self.label)],'labels':self.label}
        # # with open("labeledData.pkl", "wb") as fp:
        # #     pickle.dump(result,fp)
        # raise StopConsumer()

    def updateCurrent(self):
        while(True):
            index, tad = next(self.tads)
            try:
                check = tad.bin1 in self.blacklist[tad.chrom] or tad.bin2 in self.blacklist[tad.chrom] 
            except AttributeError:
                check = False
            if check:
                self.label[index] = 0
                continue
            mat = self.fetchMat(tad)
            if self.mode == 'check':
                mat['label'] = self.inconsistentLabel[index]
            if not mat:
                self.label[index] = 0
            else:
                self.send_json(mat)
                self.currentTad = tad
                self.currentTadIndex = index
                break
    def updateNext(self):
        self.next = None
        self.nextTad = None
        self.nextTadIndex = None
        while(True):
            try:
                index, tad = next(self.tads)
            except StopIteration:
                return False
            try:
                check = tad.bin1 in self.blacklist[tad.chrom] or tad.bin2 in self.blacklist[tad.chrom] 
            except AttributeError:
                check = False
            if check:
                self.label[index] = 0
                continue
            mat = self.fetchMat(tad)
            if self.mode == 'check':
                mat['label'] = self.inconsistentLabel[index]
            if not mat:
                self.label[index] = 0
            else:
                self.next = mat
                self.nextTad = tad
                self.nextTadIndex = index
                break

    def updateCurrentA(self):
        try:
            index = next(self.tadIndex)
        except StopIteration:
            return False
        tad = self.tads.iloc[index]
        self.currentTadIndex = index
        mat = self.fetchMat(tad)
        self.currentTad = tad
        # send current model prediction
        feature = self.features[index:(index+1),:]
        predict = np.argmax(self.model.predict(feature),axis=1)[0]
        mat['predict'] = int(predict)
        self.send_json(mat)

    def updateNextA(self):
        try:
            index = next(self.tadIndex)
        except StopIteration:
            return False

        tad = self.tads.iloc[index]
        self.nextTadIndex = index
        mat = self.fetchMat(tad)
        # send current model prediction
        feature = self.features[index:(index+1),:]
        predict = np.argmax(self.model.predict(feature),axis=1)[0]
        mat['predict'] = int(predict)

        self.nextTad = tad
        self.next = mat
        return True
        
    def trainModel(self):
        print("train model")
        x = self.features[self.labeled_idx]
        y = self.label[self.labeled_idx]
        # add mirror tad
        mirror_x,mirror_y = mirrorTAD(x,y)
        x = np.vstack((x,mirror_x))
        y = np.append(y,mirror_y)
        # if self.mode == 'random': 
        #     mirror_x,mirror_y = mirrorTAD(x,y)
        #     x = np.vstack((x,mirror_x))
        #     y = np.append(y,mirror_y)
        print(y)
        y = keras.utils.to_categorical(y, num_classes = 2)

        self.model.fit(x, y,
                epochs=1000,
                batch_size=20,
                shuffle=True,
                class_weight={0 : float(x.shape[0]) / y[y==0].shape[0],
                                1 : float(x.shape[0]) / y[y==1].shape[0]},
                verbose=2)
        if self.testSet:
            evaluation = self.model.evaluate(self.testSetX, self.testSetY)
            print('Test loss:', evaluation[0])
            print('Test accuracy:', evaluation[1])
            if evaluation[1] > 0.95:
                print('accuracy reached 0.95!')
    # def encode_json(self,content):
        # test = JSONRenderer().render(content)
        # print(test)
        # return test
    def fetchMat(self,tad):
        res = self.mcool.binsize
        chromsize = self.mcool.chromsizes[tad.chrom] // res 
        start = int(tad.bin1)
        end   = int(tad.bin2)
        size = end - start
        wingSize = int((300 - size)/2)
        if(start - wingSize < 0):
            tadStart = start
            tadEnd = end
            end = end + 2*wingSize
            start = 0
        else:
            tadStart = wingSize
            tadEnd = wingSize + size
            start = start - wingSize
            end   = (end + wingSize) if (end + wingSize) < chromsize else chromsize

        #mat = self.mcool.matrix(balance=True,sparse=True).fetch((tad.chrom,start*res, end * res))
        try:
            mat = self.mcool.matrix(balance=True,sparse=False).fetch((tad.chrom,start*res, end * res))
        except:
            mat = self.mcool.matrix(balance=False,sparse=False).fetch((tad.chrom,start*res, end * res))
        mat = np.nan_to_num(mat)
        shape = mat.shape[0]
        # nonzeroRatio = np.count_nonzero(mat)/(shape*shape)
        # print(nonzeroRatio)
        # if mat.sum()==0 or nonzeroRatio < 0.3:
        #     return False
        # data = np.array([mat.row,mat.col,mat.data]).T.tolist()
        data = mat.flatten().tolist()
        return {
            # should add tadPos
            'data': data,
            'shape': shape,
            'tadPos': [tadStart,tadEnd]
        }

    def getInconsistentRes(self):
        # debug
        self.features = None

        predicts = self.model.predict(self.testSetX)
        model_predictions = np.argmax(predicts,axis=1)==1
        test_output = self.testSetData['y']
        incorrect_index = tf.math.not_equal(model_predictions,test_output)
        correct_index = np.logical_not(incorrect_index)
        print(f'there is {incorrect_index.numpy().sum()} inconsistent TADs')
        tads = pd.DataFrame(self.testSetData['tads'][incorrect_index],columns=['chrom','bin1','bin2'])
        rest_tads = pd.DataFrame(self.testSetData['tads'][correct_index],columns=['chrom','bin1','bin2'])
        self.tads = tads.iterrows()
        self.finalTads = pd.concat([tads,rest_tads])
        self.consistentLabel = test_output[correct_index]
        self.inconsistentLabel = test_output[incorrect_index]
        self.newX = np.concatenate((self.testSetX[incorrect_index],self.testSetX[correct_index]))
        label = np.empty(tads.shape[0])
        label[:] = np.nan
        self.label = label


    # def trainModel(self,):
    # def updateModel(self,):
