import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class ring_data:
    def __init__(self,n_images,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.n_images = n_images
        self.data = []

    def point(self):
        x = 0.0
        y = 0.0
        test = 0
        while test == 0:
            #x = 2 * numpy.random.rand() - 1
            #y = 2 * numpy.random.rand() - 1
            x = 2.0*self.max_val*numpy.random.rand() - self.max_val
            y = 2.0*self.max_val*numpy.random.rand() - self.max_val
            if self.min_val < math.sqrt(x*x+y*y) and math.sqrt(x*x+y*y) < self.max_val:
                test = 1

        return x,y

    def ring(self):

        points=[]

        for i in range(100): points.append(self.point())

        points=numpy.transpose(numpy.array(points))

        return points

    def ring_to_image(self):

        ring=self.ring()

        h, xedges, yedges, Image = plt.hist2d(ring[0],ring[1],bins=25,norm=LogNorm(),normed=True,range=((-1.5,1.5),(-1.5,1.5)))

        plt.close()

        return h

    def get_data(self):

        for image in range(self.n_images):

            if image%(self.n_images*.1) == 0 and image !=0: print(str(100*image/self.n_images))

            self.data.append(self.ring_to_image())

    def save_data(self):

        if len(self.data) == 0:

            print("No data to save!")

        else:

            numpy.savez_compressed("test.npz",data=self.data)

r=ring_data(1000,.75,1.0)
r.get_data()
r.save_data()
