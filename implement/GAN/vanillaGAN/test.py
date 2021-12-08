# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# # GPU
# torch.cuda.set_device(3)
# print(torch.cuda.current_device())

# import pickle
#
# lossD = [1, 2.0]
# lossG = [2.0, 1]
#
# with open('./data/lossD.txt', 'wb') as f:
#     pickle.dump(lossD, f)
#
# with open('./data/lossG.txt', 'wb') as f:
#     pickle.dump(lossG, f)
#
# with open('./data/lossD.txt', 'rb') as f:
#     D = pickle.load(f)
#
# with open('./data/lossG.txt', 'rb') as f:
#     G = pickle.load(f)
#
# print(D)
# print(G)

