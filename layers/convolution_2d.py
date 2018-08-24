import numpy

class layer:
    def __init__(self,filters,filter_d=2,stride=1,padding=0,input_layer=0):
        self.filters = filters
        self.filter_d = filter_d
        self.stride = stride
        self.padding = padding
        self.input_layer = input_layer

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

    def single_filter(self,data):

        conv_X=[]

        for i in range(len(data)):
            conv_X.append(self.convolution_2d(data[i]))
        conv_X=numpy.array(conv_X)

        return conv_X

    def evaluate(self,data,input_layer=0):
        if self.input_layer == 1:
            filters=[]
            for i in range(self.filters):
                filters.append(self.single_filter(data))
            filters=numpy.array(filters)
            return filters
        else:
            maps=[]
            for j in range(len(data)):
                filters=[]
                for i in range(self.filters):
                    filters.append(self.single_filter(data))
                filters=numpy.array(filters)
                maps.append(filters)
            maps=numpy.array(maps)
            return(maps)

if __name__ == '__main__':
    import numpy
    X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
    conv_1=layer(10)
    output=conv_1.evaluate(X)
    print(numpy.shape(output))
