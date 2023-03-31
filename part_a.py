'''
Creating Tensors with different shapes, sizes, and values
Vector manipulation of tensors of various types
Matrix manipulation of tensors of various types
'''

import torch

'''
We can create various types of tensors of different sizes and different default values
depending on the command that we use and the arguments we input.
'''
print("""Tensors are created by using torch.type(dimensons), where type
determines the default value of the values inside the tensor.
For example, here are some tensors of dimensions 2x4 with all zeroes,
then all ones, then random values between 0 and 1. 
Dimensions 2x4 means an array of 2 arrays with 4 values in each.""")
print(torch.empty(2, 4))
print(torch.ones(2, 4))
print(torch.rand(2, 4))
print("""The tensor full of ones isn't set as just ones because 
      tensors are by default floating point values, I'll go into this later 
      about how to change default types and actually make 1 as default integers""")


print("""\nThe seed can be set for random tensor generation to control the 
randomness. The next two tensors should be the same.""")
torch.manual_seed(1000)
print(torch.rand(2, 3))
torch.manual_seed(1000)
print(torch.rand(2, 3))

print('\nHere are some tensors with higher dimensions, all with random values.')
print(torch.rand(3, 4, 5))
print(torch.rand(2, 2, 4, 3))
print(torch.rand(2, 5, 3, 2, 3))

print("\nTensors of 1 dimension are sometimes referred to as vectors, e.g.")
print(f'{torch.rand(3)}\n{torch.rand(5)}')

print("\nTensors of 2 dimensions are sometimes referred to as matrices, e.g.")
print(f'{torch.rand(2, 3)}\n{torch.rand(4, 5)}')

print("""\nTensors of any size can be multiplied by a constant, and as a result
all the values in the tensor will be multiplied by that constant""")
print("\nMultiplying a random tensor by 3 will give all values between 0 and 3")
print(torch.rand(2, 3, 2) * 3)
print("""\nSimilarly, Tensors of any size can be added by a constant, and as
a result all the values in the tensor will be added by that ocnstant""")
print("\nAdded 10 to a random tensor will give all values between 10 and 11")
print(torch.rand(2, 3) + 10)


print("""\nIf tensors are of the same size, they can be added, multiplied,
divided, and subtracted from each other. The mathematical operation will
simply apply to every pair of values that are in the same position in
the tensor""")
print('\nSubtracting two random tensors will give all values between -1 and 1')
print(torch.rand(2, 2) - torch.rand(2, 2))
print('\nMultiplying two random tensors will give all values between 0 and 1')
print(torch.rand(2, 3) * torch.rand(2, 3))
print("""\nAdding two random tensors that have been multiplied first by 5 and
3 respectively will give a tensor with all values between 0 and 15""")
print((torch.rand(3, 3) * 5) + (torch.rand(3, 3) * 3))

print("""\nSo far all values in the tensors have been 32bit floating point
numbers, as this is the default, however we can specify a range of different
types when the tensor is created using the dtype=torch.type argument """)
print('\nA tensor of zeroes as 32bit ints, then a tensor of ones as 64bit ints')
print(f'{torch.empty(2, 2, dtype=torch.int32)}\n{torch.ones(2, 2, dtype=torch.int64)}')
print("""\nThere is a chance each of the 0s in the first tensor will be cast to
a very large negative number, instead of 0. This is because of how floating
point numbers are stored. 0 as a float will generally be slightly above or slightly
below 0. Therefore adding something like 0.1 to all values before changing to int
would fix this, leading us to our next part:""")

print('\nWe can also change the type of a tensor after creating using tensor.to')
print('\nLets make a tensor of actual 0 int values')
print((torch.empty(2, 2) + 0.1).to(torch.int32))
print('\nLets make a random tensor and multiply it by 10')
t = torch.rand(2, 2) * 10
print(t)
print('\nChanging the type from float32 to int16 will give int values from 0 to 9')
t = t.to(torch.int16)
print(t)

print("""We can use the same process to create a tesnor where roughly half the values
will be True, roughly half the values will be False, however it depends on the
initial random generation.""")
print('\nWe create a random tensor, add 0.5 to it, change to int, then change to bool')
print((torch.rand(2, 2) + 0.5).to(torch.int32).to(torch.bool))

