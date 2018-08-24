import numpy

class layer:
    def __init__(self,nodes,filter_d=2,stride=1,padding=0):
        self.nodes = nodes
        self.filter_d = filter_d
        self.stride = stride
        self.padding = padding

    def convolution_2d(self,X):

        W=numpy.random.rand(self.filter_d,self.filter_d)
        W=numpy.flip(numpy.flip(W,axis=0),axis=1)
        #print(X)
        dim=(numpy.shape(X)[0]-numpy.shape(W)[0]+2*self.padding)/self.stride+1
        c=numpy.zeros((dim,dim))
        m=(self.padding-1)/2
        n=m
        def conv(data,filter,p,q):
            val=0
            for i in range(self.filter_d):
                for j in range(self.filter_d):
                    val=data[p+i,j+q]*W[i,j]+val

            return val

        l=[]
        #f=lambda i,j:conv(X,W,i,j)
        for num1 in range(dim):
            for num2 in range(dim):
                l.append(conv(X,W,num2,num1))

        l=numpy.round(numpy.reshape(numpy.array(l),(dim,dim)),1)
        #print(numpy.shape(l))
        #class convolution_2d:
        #    def __init__(self,array,filter_dim_1,filter_dim_2):
        #self.data=array
        return l

    def evaluate(self,data):

        conv_X=[]
        print("test")
        for i in range(len(X)):
            conv_X.append(self.convolution_2d(X[i]))
        conv_X=numpy.array(conv_X)
        print(numpy.shape(conv_X))
        return conv_X

if __name__ == '__main__':
    import numpy
    X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
    conv_1=layer(10)
    output=conv_1.evaluate(X)
