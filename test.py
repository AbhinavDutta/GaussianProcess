from gaussianprocessregression  import  GaussianProcessRegression
from matern52kernel import Matern52Kernel
import numpy
x=numpy.array([1,2,3,4,5])
y=numpy.array([10,20,31,43,55])
kern = Matern52Kernel(1,1)
gpr = GaussianProcessRegression(kern,X=x,y=y)
xt = [7,8,9]
r=gpr.get_predictions(xt)
print(r)