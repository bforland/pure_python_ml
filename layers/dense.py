import numpy

class layer:
    def __init__(self,nodes):
        self.nodes = nodes

    def evaluate(self,data):
        weights=numpy.random.rand(len(data),self.nodes)
        return numpy.transpose(weights).dot(data)

if __name__ == '__main__':
    import numpy
    X = numpy.floor(10.0 * (numpy.random.rand(10,6,6)))
    conv_1=layer(10)
    output=conv_1.evaluate(X)
