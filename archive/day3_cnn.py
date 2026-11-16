from torchvision import datasets, transforms
import torch
import torch.nn.functional as F

# train_data = datasets.MNIST(root="./data", train=True, download=True,
#                             transform=transforms.ToTensor())
# print(train_data[0][0].shape)   # 应该是 torch.Size([1, 28, 28])
# print(len(train_data))
#
# img, label = train_data[0]
#
# for row in img[0]:
#     print("".join("█" if v > 0.5 else ("+" if v > 0.2 else ".") for v in row))
#
# print("这张图的标签是:", label)
# # 60000

# img, label = train_data[0]              # [1, 28, 28]
# batch = img.unsqueeze(0)                # [1, 1, 28, 28] 最前面加一个"批次"
#
# k = torch.tensor([[[[-1., 0., 1.],
#                     [-1., 0., 1.],
#                     [-1., 0., 1.]]]])   # [1, 1, 3, 3]
#
# out = F.conv2d(batch, k)
# print("卷积后:", out.shape)              # 应该是 [1, 1, 26, 26]

# for row in out[0][0]:
#     print("".join("█" if v > 2 else ("▄" if v < -2 else ".") for v in row))



# 造一张 8x8 的图，画两条竖线
pic = torch.zeros(1, 1, 8, 8)
for i in range(8):
    pic[0, 0, i, i] = 1.0    # 斜线：第 i 行的第 i 列


# 一把尺子：专门找"左暗右亮"
k = torch.tensor([[[[-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.]]]])

out = F.conv2d(pic, k)

for row in out[0][0]:
    print(" ".join(f"{v:3.0f}" for v in row))

