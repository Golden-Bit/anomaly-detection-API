
"""
source: https://medium.com/@sahil_g/cpu-vs-gpu-pytorch-tensor-operations-50e215ff764a
"""

#importing required libraries
import torch
import time

#Initialisation of tensors
dim=8000

start_time = time.time()
x=torch.randn(dim,dim)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
x=torch.randn((dim,dim), device=torch.device("cuda:0"))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

# Matrix Multiplication
dim=8000

x=torch.randn(dim,dim)
y=torch.randn(dim,dim)
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)


x=torch.randn(dim,dim,device=torch.device("cuda:0"))
y=torch.randn(dim,dim,device=torch.device("cuda:0"))
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)


#Broadcasting
dim=8000

device=torch.device("cuda:0")

start_time = time.time()
torch.add(torch.randn(dim,1), torch.randn(dim))
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
torch.add(torch.randn(dim,1,device=device), torch.randn(dim,device=device))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)


#Outer Product of tensors

dim=8000

device=torch.device("cuda:0")

start_time = time.time()
torch.outer(torch.randn(dim), torch.randn(dim))
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)

start_time = time.time()
torch.outer(torch.randn(dim,device=device), torch.randn(dim,device=device))
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

"""
In all the above tensor operations, the GPU is faster as compared to the CPU. But if we reduce the dimension of the 
tensors a lot, then the computation required to do the above operations will be small, and a very small number of cores 
will be required. As we know, the individual cores of the CPU are more powerful than that of the GPU. Hence, the CPU 
will be equally fast or faster than the GPU in case of low dimensions of tensors, as parallel computation of the GPU 
will be compensated by the strength of individual cores of the CPU.
"""

#Element Wise operations (multiplication,addition,subtraction)
dim=4

x=torch.randn(dim,dim)
y=torch.randn(dim,dim)
start_time = time.time()
a=x*y
b=x+y
c=x-y
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)


x=torch.randn(dim,dim,device=torch.device("cuda:0"))
y=torch.randn(dim,dim,device=torch.device("cuda:0"))
start_time = time.time()
a=x*y
b=x+y
c=x-y
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

"""
Now, if some task involves recurrence. It must be done sequentially. 
Hence, the CPU will be more efficient in such cases.
"""

dim=4

x=torch.randn(dim,dim)
y=torch.randn(dim,dim)

start_time = time.time()
for i in range(100000):
   x+=x
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)


x=torch.randn(dim,dim,device=torch.device("cuda:0"))
y=torch.randn(dim,dim,device=torch.device("cuda:0"))

start_time = time.time()
for i in range(100000):
   x+=x
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)

"""
We have 10⁵ operations, but due to the structure of the code, it is impossible to parallelize much of these computations 
because to compute the next x, you need to know the value of the previous (or current) x. We only can parallelize 16 
operations (additions) per iteration. As the CPU has few but much more powerful cores, it is faster for the given example!
"""