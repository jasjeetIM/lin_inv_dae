import numpy as np
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

def estimate_ipt_celebA(A, A_T, y, autoencoder, iterations, visualize=False, x=None): 
        
        if visualize and x is not None: 
            plt.figure(figsize=(20, 2))
 
        x_t = np.random.normal(0., 1, (12288,))
        x_final = np.zeros((64,64,3))
        
        start_time = time.time()
        # WARNING: Visualization logic is hardcoded to the value 10, 
        #  if you change iterations , visualization logic will need to be changed too
        for j in range(iterations):
            v_t = x_t - np.dot(A_T, np.dot(A,x_t) - y)
            x_t = autoencoder.predict(v_t.reshape(1,64,64,3))
            
            if visualize and x is not None:
            #Print original image
                if j == 0:
                    ax = plt.subplot(1, 20, j / 1 + 1)
                    plt.imshow(x)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                #Show recovered image at iteration j
                if j % 1== 0 and j > 1:
                    ax = plt.subplot(1, 20, j/1 + 1)
                    plt.imshow(x_t.reshape(64,64,3))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

            x_t = x_t.reshape((12288,))
            if j == iterations -1:
                time_diff = time.time() - start_time #time to run 10 iterations 
                x_final = x_t.reshape((64,64,3))
        
        if visualize:
            plt.show() 
        
        return x_final, time_diff


def get_super_resol_A(factor):
    A = np.zeros(shape=(int(64/factor)**2*3, 64*64*3))
    l = 0
    for i in range(int(64/factor)):
        for j in range(int(64/factor)):
            for k in range(3):
                a = np.zeros(shape=(64, 64, 3))
                a[factor*i:factor*(i+1), factor*j:factor*(j+1), k] = 1.0/factor**2
                A[l, :] = np.reshape(a, [1, -1])
                l += 1
    return A

def get_inpaint_A(mask_size):
    A = np.ones((3, 64, 64))
    A[:, int(64/2 - mask_size/2):int(64/2 + mask_size/2), int(64/2 - mask_size/2):int(64/2 + mask_size/2)] = 0
    A = A.reshape(-1)
    A = np.diag(A)
    return A
