import numpy as np
import cv2
from imgUtil import *


class YDEnv(object):
    action_bound = [-1, 1]
    action_dim = 3
    state_dim = 12

    def __init__(self, ):
        simgSrc = cv2.imread("imgSrc.bmp")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kimgSrc = cv2.dilate(simgSrc, kernel)  # 膨胀
        kimgSrc[:, :, 1:3] = 0
        kimgSrc[:, :, 0] = kimgSrc[:, :, 0] / 255.0
        simgSrc[:, :, 0:2] = 0
        simgSrc[:, :, 2] = simgSrc[:, :, 2] / 255.0

        # self.env_info = np.zeros([12], dtype=np.float32)
        self.env_info = np.array([77.,44.,  140.,  139.,   76.,   27.,   62.,   32., - 113.,    1.,    1.,    0.])

        self.simgSrc = simgSrc
        self.kimgSrc = kimgSrc

    def rest(self,):
        lab = np.array([77.,44.,  140.,  139.,   76.,   27.,   62.,   32., - 113.,    1.,    1.,    0.])

        lab[0] = np.random.randint(30, 224 - 30)
        lab[1] = np.random.randint(30, 224 - 30)
        lab[2] = np.random.randint(0, 180)

        lab[3] = np.random.randint(30, 224 - 30)
        lab[4] = np.random.randint(30, 224 - 30)
        lab[5] = np.random.randint(0, 180)


        lab[6:9] = lab[3:6] - lab[0:3]
        lab[9:12][lab[6:9] > 0] = 1

        self.env_info = lab

    def step(self, action):
        # action = np.clip(action, *self.action_bound)
        self.env_info[0:3] += action
        if self.env_info[2] > 178:
            self.env_info[2] = 178
        if self.env_info[2] < 0:
            self.env_info[2] = 0

        self.env_info[6:9] = self.env_info[3:6] - self.env_info[0:3]
        self.env_info[9:12] = 0
        self.env_info[9:12][self.env_info[6:9] > 0] = 1

        return self.env_info.copy()

    def getimg(self):
        blank = np.zeros(shape=[224, 224, 3])
        lab = self.env_info

        tem = makeSpinImage(self.kimgSrc, lab[2])
        startx = int(lab[0] - tem.shape[0] / 2)
        starty = int(lab[1] - tem.shape[1] / 2)
        blank[startx:startx + tem.shape[0], starty:starty + tem.shape[1], :] += tem

        tem = makeSpinImage(self.simgSrc, lab[5])
        startx = int(lab[3] - tem.shape[0] / 2)
        starty = int(lab[4] - tem.shape[1] / 2)
        blank[startx:startx + tem.shape[0], starty:starty + tem.shape[1], :] += tem
        return blank

class trainMemory(object):

    def __init__(self, ):

        # np.unique(a,axis=0)
        capacity = 300
        self.pointer = 0
        self.capacity = capacity
        self.data = np.zeros([capacity, 12], dtype=np.float32)
        self.imgs = np.zeros([capacity, 224,224,3], dtype=np.float32)

    def store_transition(self, lab, img):
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = lab
        self.imgs[index, :] = img
        self.pointer += 1

    def sample(self, n):
        indices = np.random.choice(min(self.capacity,self.pointer), size=n)


        return self.data[indices, :],self.imgs[indices, :]


if __name__ == '__main1__':
    env = YDEnv()
    action = np.array([3,0,0])
    n = 1
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'):
            action = np.array([-n, 0, 0])
        elif key == ord('s'):
            action = np.array([n, 0, 0])
        elif key == ord('a'):
            action = np.array([0, -n, 0])
        elif key == ord('d'):
            action = np.array([0, n, 0])
        elif key == ord('q'):
            action = np.array([0, 0, -n])
        elif key == ord('e'):
            action = np.array([0, 0, n])
        print(env.step(action))
        cv2.imshow('123', env.getimg())


if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_hub as hub
    modelTest = tf.keras.models.load_model('oldModel2.h5',custom_objects={'KerasLayer':hub.KerasLayer})

    env = YDEnv()
    action = np.array([3, 0, 0])
    n = -1
    for j in range(200):
        env.rest()
        for i in range(150):
            img = env.getimg()[np.newaxis,]
            lab = modelTest(img)
            print(lab)
            if lab[0,0] > 0.8:
                action[0] = -n
            elif lab[0,0] < 0.2:
                action[0] = n
            if lab[0, 1] > 0.8:
                action[1] = -n
            elif lab[0, 1] < 0.2:
                action[1] = n
            if lab[0, 2] > 0.8:
                action[2] = -n
            elif lab[0, 2] < 0.2:
                action[2] = n

            env.step(action)
            action = np.array([0, 0, 0])
            try:
                cv2.imshow('123', env.getimg())
            except:
                break
            cv2.waitKey(1)

if __name__ == '__main1__':
    from VRDatasetC import NpDataset

    env = YDEnv()

    train_dateset = NpDataset('train')
    for imgs, lables in train_dateset:
        # print(lables[3])
        print(imgs.shape)
        print(lables.shape)
        # print(np.unique(imgs[0]))
        cv2.imshow('1', imgs[3])
        env.env_info = lables[3]
        env.step(0)
        img = env.getimg()
        cv2.imshow('123', img)

        print(env.env_info)
        print(lables[3])
        cv2.waitKey(0)


if __name__ == '__main1__':

    import tensorflow as tf
    from VRDatasetC import NpDataset

    trainset = NpDataset('train')
    modelTest = tf.keras.models.load_model('oldModel3.h5')
    optimizer = tf.keras.optimizers.Adam(0.00003)
    tm = trainMemory()
    env = YDEnv()
    action = np.array([3, 0, 0])
    n = -1

    for j in range(200):
        env.rest()
        for i in range(150):

            img = env.getimg()[np.newaxis,]
            lab = modelTest(img)

            if lab[0,0] > 0.5:
                action[0] = -n
            elif lab[0,0] < 0.5:
                action[0] = n
            if lab[0, 1] > 0.5:
                action[1] = -n
            elif lab[0, 1] < 0.5:
                action[1] = n
            if lab[0, 2] > 0.5:
                action[2] = -n
            elif lab[0, 2] < 0.5:
                action[2] = n



            labTem = env.step(action)
            try:
                img = env.getimg()
            except :
                break
            print(i)
            print(lab)
            print(labTem)
            tm.store_transition(labTem, img)
            cv2.imshow('123', img)
            cv2.waitKey(1)

        tem = 30
        for i in range(tem):
            target, image_data = tm.sample(8)
            Simage_data, Starget = next(iter(trainset))

            target = np.vstack((Starget, target))
            image_data = np.vstack((Simage_data, image_data))

            # envTem=YDEnv()
            # envTem.env_info = target[3]
            # envTem.step(0)
            # imgTem = envTem.getimg()
            # cv2.imshow('123', imgTem)
            # cv2.imshow('12', image_data[3])
            # cv2.waitKey(0)

            with tf.GradientTape() as tape:

                lab = modelTest(image_data, training=True)

                t = target[:, 9:12]  # 0,1

                meanLoss = tf.reduce_mean(
                    t * tf.square(tf.nn.relu(0.8 - lab)) + (1 - t) * tf.square(tf.nn.relu(lab - 0.2)))
                # meanLoss = tf.reduce_mean(tf.keras.losses.mean_squared_error(lab, target[:, 9:12]))

                loss_regularization = []
                for p in modelTest.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

                total_loss = meanLoss
                gradients = tape.gradient(total_loss, modelTest.trainable_variables)
                optimizer.apply_gradients(zip(gradients, modelTest.trainable_variables))
                tf.print(meanLoss, loss_regularization)

        modelTest.save("oldModel2.h5")
        tf.print("saveModel")