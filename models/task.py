import torch.nn as nn
import torch
import pdb

# 这是一个用于学习任务不变特征的模块。它包含了一系列的1x1卷积层和ReLU激活函数，
# 用来增强特征的表示能力。该模块接收输入特征，并对其执行一系列卷积操作和非线性变换，输出加强了特征表达的结果。


class TaskInvariantFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TaskInvariantFeatures, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#这是一个层次解耦模块。它接收一个特征张量作为输入，并根据给定的任务数，使用注意力机制对特征进行加权处理。
# 针对每个任务，该模块计算对应的注意力权重，将这些权重应用于输入特征，然后对加权后的特征进行平均池化，以生成任务特定的特征表示。

class HierarchicalDecoupling(nn.Module):
    def __init__(self, in_channels, num_tasks):
        super(HierarchicalDecoupling, self).__init__()
        self.num_tasks = num_tasks  #对 self.attention_weights 进行 softmax 操作，以确保每个任务的注意力权重在通道维度上的总和为 1。得到 att_weights，形状为 (num_tasks, in_channels)。
        self.attention_weights = nn.Parameter(torch.randn(num_tasks, in_channels)) #并使用torch.randn函数对其进行随机初始化。这些随机初始化的值将作为模型学习的初始注意力权重，在训练过程中会通过反向传播进行调整，以最小化定义的损失函数。

    def forward(self, x):
        att_weights = torch.softmax(self.attention_weights, dim=1)
        task_specific_features = []  #将每个任务生成的特定特征张量 weighted_features 添加到 task_specific_features 列表中。
        for i in range(self.num_tasks):

            attention_weight = att_weights[i].view(1, -1, 1, 1)  #  torch.Size([1, 1024, 1, 1])它的形状表示为 (batch_size=1, channels=1024, height=1, width=1)。
            weighted_features = x * attention_weight.expand_as(x) #torch.Size([384, 1024, 1, 1])  #模块通过self.attention_weights参数定义了每个任务的注意力权重。这些权重用于加权输入特征 x，以产生任务特定的特征表示。
            task_specific_features.append(weighted_features)        #您可以添加一个正交化损失函数，以鼓励注意力权重之间的正交性。这可以通过在训练过程中迫使权重矩阵的转置与自身的乘积为单位矩阵来实现。这种方法可以使得权重之间的关系更加独立，
        stacked_features = torch.stack(task_specific_features, dim=0)  #torch.Size([2, 384, 1024, 1, 1]) #使用 torch.stack 在 task_specific_features 列表中的特征张量上进行堆叠，沿着维度 dim=0，得到 stacked_features，形状为 (num_tasks, batch_size, in_channels, height, width)。
        return stacked_features.mean(dim=0)  # Taking mean across the tasks #使用 stacked_features.mean(dim=0) 计算沿着任务维度 (num_tasks 维度) 的平均值，得到一个张量，其形状与输入张量 x 相同，表示了跨所有任务的任务特定特征的聚合。

# 在进行特征解耦时，需要对权重进行正交化的目的在于让不同任务（如 ReID 和检测）之间的注意力权重更加独立和正交，
# 以便确保模型能够专注于不同任务所需的特征表示。
# 正交化权重的想法是使得不同任务的注意力权重在特征空间中更具有差异性，这有助于确保模型学习到的特征对于不同任务是更具区分性和独立性的。
# 在这个过程中，通过鼓励不同任务的注意力权重正交化，可以使得模型更专注于每个任务所需的特征表示，而不是混淆或共享特征。
#
#     def compute_orthogonality_loss(self):
#         attention_weights = self.attention_weights
#         orthogonality_loss = torch.norm(
#             torch.matmul(attention_weights, attention_weights.t()) - torch.eye(attention_weights.shape[0]))
#         return orthogonality_loss



# 在深度学习中，正交化损失用于促使模型学习到具有更好特性的参数或表示。当应用于注意力权重向量时，正交化损失的目的是增加注意力权重之间的独立性，
# 从而改善模型的泛化能力和学习表示的有效性。正交化损失背后的主要思想是，通过强制模型学习到的参数或表示之间具有较小的相关性或共线性，能够更好地泛化到未见过的数据上。
# 在注意力机制中，如果不同任务的注意力权重之间存在很高的相关性或共线性，可能会导致模型过度依赖少数任务或特定模式，从而降低模型在新任务或不同数据分布下的性能。
# 通过引入正交化损失，可以促使注意力权重之间更加独立，减少它们之间的相关性，使得模型更能够灵活地适应不同的任务或数据分布。这有助于提高模型的泛化能力，
# 减少过拟合的风险，并且可能改善模型在多任务学习中的效果。在实践中，正交化损失可以通过在模型的训练过程中，向损失函数中添加一项来实现。
# 这一项通常是基于注意力权重矩阵的转置与自身相乘得到单位矩阵的损失项。通过最小化这个损失项， 模型被迫学习到不同任务之间更加独立的注意力权重，以便更好地学习任务间的共享特征，并且对不同任务之间保持更好的区分度。
#因此，正交化权重可以有助于减少任务间的相关性，使模型学习到更加独立和泛化的任务表示。这种独立性有助于模型更好地学习到不同任务之间的共享特征，提高模型的泛化能力，同时减少不必要的信息冗余。





#这是一个整体模型，由TaskInvariantFeatures和HierarchicalDecoupling组成。该模型包含一个任务不变特征学习模块列表（task_invariant_modules），
# 用于学习输入特征的任务不变表示。此外，它还包含一个HierarchicalDecoupling模块，用于将任务特定的特征与特定任务进行解耦和加权处理。在前向传播中，
# 模型接收一个字典类型的特征数据 box_features，并对每个特征执行任务不变特征学习操作和层次解耦操作，然后输出处理后的特征字典 output。

#这个模型的目的是从输入的特征中学习任务不变表示，并根据任务数以一种解耦的方式对特征进行处理，以便在多任务学习中使用。

def compute_orthogonality_loss(hierarchical_decoupling_before_trans,hierarchical_decoupling_after_trans):
        # 获取两个不同任务的注意力权重
        attention_weights_before_trans = hierarchical_decoupling_before_trans.attention_weights
        attention_weights_after_trans = hierarchical_decoupling_after_trans.attention_weights
        # 如果需要对第一个注意力权重矩阵进行转置
        attention_weights_before_trans = attention_weights_before_trans.transpose(0, 1)
        # 调整第二个注意力权重矩阵的大小为 (1024, 2)
        attention_weights_after_trans = attention_weights_after_trans.transpose(0, 1)[:1024, :]
        # 计算两个注意力权重向量的正交化损失
        orthogonality_loss = torch.norm(torch.matmul(attention_weights_before_trans,
                                                        attention_weights_after_trans.t()))  # 此处为两个权重向量之间的点积
        return orthogonality_loss




#对两个同时解耦
class TaskInvariantModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_tasks):
        super(TaskInvariantModel, self).__init__()
        self.task_invariant_modules = nn.ModuleList([TaskInvariantFeatures(in_channels[i], out_channels[i]) for i in range(len(in_channels))])  #这个列表中的每个元素实际上都是一个独立的 1x1 卷积层。
        self.hierarchical_decoupling_before_trans = HierarchicalDecoupling(out_channels[0], num_tasks)
        self.hierarchical_decoupling_after_trans = HierarchicalDecoupling(out_channels[1], num_tasks)  # Assuming 'after_trans' uses different output channels

    # 定义计算两个任务注意力权重向量的正交化损失

    def forward(self, box_features):
        output = {}
        for i, key in enumerate(box_features.keys()):
            feature = box_features[key]
            processed_feature = self.task_invariant_modules[i](feature)
            if key == 'before_trans':
                processed_feature = self.hierarchical_decoupling_before_trans(processed_feature)
            elif key == 'after_trans':
                processed_feature = self.hierarchical_decoupling_after_trans(processed_feature)
            output[key] = processed_feature

        return output





#只对第一个特征进行解耦
# class TaskInvariantModel(nn.Module):
#     def __init__(self, in_channels, out_channels, num_tasks):
#         super(TaskInvariantModel, self).__init__()
#         self.task_invariant_modules = nn.ModuleList([TaskInvariantFeatures(in_channels[i], out_channels[i]) for i in range(len(in_channels))])
#         self.hierarchical_decoupling = HierarchicalDecoupling(out_channels[0], num_tasks)
#
#     def forward(self, box_features):
#         output = {}
#         for i, key in enumerate(box_features.keys()):
#             feature = box_features[key]
#             processed_feature = self.task_invariant_modules[i](feature)
#             if i == 0:
#                 processed_feature = self.hierarchical_decoupling(processed_feature)
#             output[key] = processed_feature
#
#         # # 输出信息
#         # for key, value in output.items():
#         #     print(f"Key: {key}, Shape: {value.shape}")  # 输出每个键对应的张量形状
#
#         return output




#只对第二个特征进行解耦
# class TaskInvariantModel(nn.Module):
#     def __init__(self, in_channels, out_channels, num_tasks):
#         super(TaskInvariantModel, self).__init__()
#         self.task_invariant_modules = nn.ModuleList([TaskInvariantFeatures(in_channels[i], out_channels[i]) for i in range(len(in_channels))])
#         self.hierarchical_decoupling_after_trans = HierarchicalDecoupling(out_channels[1], num_tasks)  # Assuming 'after_trans' uses different output channels
#
#     def forward(self, box_features):
#         output = {}
#         for i, key in enumerate(box_features.keys()):
#             feature = box_features[key]
#             processed_feature = self.task_invariant_modules[i](feature)
#             if key == 'after_trans':
#                 processed_feature = self.hierarchical_decoupling_after_trans(processed_feature)
#             output[key] = processed_feature
#
#         return output



# # 示例使用  #那么就表示你希望创建三个卷积层，每一层的输入和输出通道数分别为这些值。
# in_channels = [1024, 2048, 1024]  # 输入通道数  设置的长度就是 需要几个卷积！！！
# out_channels = [1024, 2048, 1024]  # 输出通道数
# num_tasks = 2  #需要处理的任务数量 reid和 cls
#
# # 创建模型
# model = TaskInvariantModel(in_channels, out_channels, num_tasks)
#
# # 示例数据，假设 box_features 是一个字典，包含 'before_trans' 和 'after_trans' 键对应的特征数据
# box_features = {'before_trans': torch.randn(384, 1024, 1, 1),
#                 'after_trans': torch.randn(384, 2048, 1, 1)}
#
# # 前向传播
# output = model(box_features)
# print(box_features["before_trans"].shape)
# print(box_features["after_trans"].shape)




# 这段代码实现了一个模型 TaskInvariantModel，其中包含了两个 HierarchicalDecoupling 模块，用于对两个不同的特征（'before_trans' 和 'after_trans'）
# 执行解耦操作。这个模型旨在学习任务不变的特征表示，并根据任务数对每个特征进行解耦和加权处理，以适应多任务学习的需求。
# 解读如下：
# TaskInvariantFeatures: 这是一个任务不变特征学习模块，由一个卷积层和ReLU激活函数组成。它用于学习输入特征的任务不变表示。
# HierarchicalDecoupling: 这是一个层次解耦模块。它接收一个特征张量作为输入，并根据给定的任务数，在特征上使用注意力机制进行加权处理。每个任务都有对应的注意力权重，
# 这些权重被应用于输入特征，并对加权后的特征进行平均池化，以生成任务特定的特征表示。
# TaskInvariantModel: 这是整体模型，由 TaskInvariantFeatures 和两个 HierarchicalDecoupling 组成。TaskInvariantFeatures 用于学习任务不变特征，
# 而两个 HierarchicalDecoupling 模块分别针对两个不同的特征进行解耦操作。在前向传播过程中，模型接收一个字典类型的特征数据 box_features，
# 并对每个特征执行任务不变特征学习和层次解耦操作，最后输出处理后的特征字典 output。
# 在这个模型中，根据输入特征的键名，before_trans 特征会经过 hierarchical_decoupling_before_trans 进行解耦处理，
# 而 after_trans 特征会经过 hierarchical_decoupling_after_trans 进行独立的解耦处理。这种设计允许模型为不同的特征学习到不同的任务不变表示，
# 并且根据需要对特征进行任务相关的解耦处理，以适应不同任务的需求。