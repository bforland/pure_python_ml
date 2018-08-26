import numpy

class layer:
    def __init__(self,filters,filter_d=2,stride=1,padding=0):
        self.filters = filters
        self.filter_d = filter_d
        self.stride = stride
        self.padding = padding
        self.output_dim=0

    def conv(self,data,W,p,q):
        val=0
        for i in range(self.filter_d):
            for j in range(self.filter_d):
                val=data[p+i,j+q]*W[i,j]+val

        return val

    def convolution_2d(self,X):

        W=numpy.random.rand(self.filter_d,self.filter_d)
        W=numpy.flip(numpy.flip(W,axis=0),axis=1)
        #print(X)
        dim=(numpy.shape(X)[0]-numpy.shape(W)[0])/self.stride+1
        #c=numpy.zeros((dim,dim))

        l=numpy.empty(0)
        #f=lambda i,j:conv(X,W,i,j)
        for num1 in range(dim):
            for num2 in range(dim):
                l=numpy.append(l,self.conv(X,W,num2,num1))

        l=numpy.reshape(numpy.array(l),(dim,dim))
        self.output_dim=dim
        #print(numpy.shape(l))
        #class convolution_2d:
        #    def __init__(self,array,filter_dim_1,filter_dim_2):
        #self.data=array
        return l

    def single_filter(self,conv_X,data):
        dims=numpy.shape(data[0])
        padded_d=numpy.zeros((dims[0]+self.padding*2,dims[1]+self.padding*2))
        for i in range(len(data)):
            padded_d[(self.padding):(self.padding+dims[0]),(self.padding):(self.padding+dims[1])]=data[i]
            conv_X=numpy.append(conv_X,self.convolution_2d(padded_d))
            padded_d=padded_d*0.0
        return conv_X

    def evaluate(self,data):
        conv_X=numpy.empty(0)
        for i in range(self.filters):
            conv_X=numpy.append(conv_X,self.single_filter(numpy.empty(0),data))
        return numpy.reshape(conv_X,(self.filters*len(data),self.output_dim,self.output_dim))

if __name__ == '__main__':
    import numpy
    X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
    conv_1=layer(10)
    output=conv_1.evaluate(X)
