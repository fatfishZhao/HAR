
import torch
from models.FCN import FCN
import torch.autograd as autograd

model = FCN(num_classes=6)

print(model)
resume = '/data3/zyx/project/HAR/trained_model/FCN_0219/weights-9-2795-[0.8717].pth'
resume_dict = torch.load(resume)
model.load_state_dict(resume_dict)
x = autograd.Variable(torch.randn(1, 6, 256))
torch_out = torch.onnx._export(model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "FCN_8717.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file