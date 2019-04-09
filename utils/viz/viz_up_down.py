import matplotlib.pyplot as plt
import numpy as np
import tqdm

up_data = np.load('/data3/zyx/project/HAR/data/up_down/up.npy')
down_data = np.load('/data3/zyx/project/HAR/data/up_down/down.npy')
jog_data = np.load('/data3/zyx/project/HAR/data/up_down/jog.npy')
walk_data = np.load('/data3/zyx/project/HAR/data/up_down/walk.npy')
for i in tqdm.tqdm(range(0,1000)):
    plt.figure(figsize=(16,16))
    # x = [j for j in range(256)]
    plt.subplot(221)
    plt.plot(up_data[i,:,0:3])
    plt.title('up')

    plt.subplot(222)
    plt.plot(down_data[i, :, 0:3])
    plt.title('down')

    plt.subplot(223)
    plt.plot(jog_data[i, :, 0:3])
    plt.title('jog')

    plt.subplot(224)
    plt.plot(walk_data[i, :, 0:3])
    plt.title('walk')

    plt.savefig('/data3/zyx/project/HAR/data/up_down/draw/'+str(i)+'.png')
    plt.close()
    # plt.show()