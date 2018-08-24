import numpy

def convolution_2d(X,filter_d=2,stride=1,padding=0):
    #X=numpy.random.rand(25,25)

    #filter_d=18
    #stride=1
    #padding=0

    W=numpy.random.rand(filter_d,filter_d)
    W=numpy.flip(numpy.flip(W,axis=0),axis=1)
    #print(X)
    dim=(numpy.shape(X)[0]-numpy.shape(W)[0]+2*padding)/stride+1
    c=numpy.zeros((dim,dim))
    m=(filter_d-1)/2
    n=m
    def conv(data,filter,p,q):
        val=0
        for i in range(filter_d):
            for j in range(filter_d):
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

if __name__ == "__main__":
    numpy.random.seed(0)
    X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
    print(numpy.shape(X))
    conv_X=[]
    for i in range(len(X)):
        conv_X.append(convolution_2d(X[i],filter_d=2,stride=2,padding=0))
    conv_X=numpy.array(conv_X)
    print(numpy.shape(conv_X))
