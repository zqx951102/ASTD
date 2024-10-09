#coat-swinT.py

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils

from loss.oim import OIMLoss
from models.resnet import build_resnet
import pdb
from models.task import TaskInvariantModel,compute_orthogonality_loss
from models.swin import build_swin



class COAT(nn.Module):
    def __init__(self, cfg):
        super(COAT, self).__init__()
        #换seresnet骨干网络时候
        #backbone, _ = build_resnet(name="resnet50", pretrained=True)
        #backbone, _ = build_resnet(name="seresnet50", pretrained=True)
        semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT
        backbone_name = cfg.MODEL.BONE
        backbone, box_head, feat_len = build_swin(name=backbone_name,
                                                  semantic_weight=semantic_weight)  # SWIN时候 需要计算给出 feat_len的长度
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )        
       # _, box_head = build_resnet(name="resnet50", pretrained=True) #全部都在第二个位置  说明都是经过Res5Head(resnet_model)的特征 要是第一个位置是backbone的输出
       #
       # _, box_head_2nd = build_resnet(name="resnet50", pretrained=True)  #Res5Head(resnet_model) 也就是图中经过SAT之后的特征！！！
       #
       # _, box_head_3rd = build_resnet(name="resnet50", pretrained=True)


        _, box_head, _ = build_swin(name=backbone_name, semantic_weight=semantic_weight)  # 全部都在第二个位置  说明都是经过Res5Head(resnet_model)的特征 要是第一个位置是backbone的输出
        _, box_head_2nd, _ = build_swin(name=backbone_name, semantic_weight=semantic_weight)  # Res5Head(resnet_model) 也就是图中经过SAT之后的特征！！！
        _, box_head_3rd, _ = build_swin(name=backbone_name, semantic_weight=semantic_weight)



        # 每个box_head 变量似乎表示一个独立的ResNet - 50模型，可能用于不同的目的或任务，例如多任务学习、模型集成
        # 或者在物体检测的不同阶段使用不同的特征表示。这种设计常见于某些物体检测框架中，其中一个模型可能用于提取区域特征，另一个用于分类或回归头部任务。




  #这是和COAT区别的地方 写法 不同的
        # ########相比与seqnet增加的地方 也就是三个 tansform的级连头  这个地方是定义
        # box_head = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_1ST,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_1ST,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_1ST,
        # )
        # box_head_2nd = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_2ND,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_2ND,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_2ND,
        # )
        # box_head_3rd = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_3RD,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_3RD,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_3RD,
        # )
        # #########
        faster_rcnn_predictor = FastRCNNPredictor(feat_len, 2)  # 要是resnet结构就是2048
     
#        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
#        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
#        box_predictor2 = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)  #多加一个 预测头

        box_predictor = BBoxRegressor(feat_len, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)  # 要是resnet结构就是2048
        box_predictor2 = BBoxRegressor(feat_len, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)  # 要是resnet结构就是2048
        roi_heads = CascadedROIHeads(
            cfg=cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            box_head_2nd=box_head_2nd,
            box_head_3rd=box_head_3rd,
            # parent class
            box_predictor2=box_predictor2,  #相比于coat 多加的头
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
            feat_len=feat_len,  # 新添加的
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        self.eval_feat = cfg.EVAL_FEATURE

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_rcnn_reg_1st = cfg.SOLVER.LW_RCNN_REG_1ST
        self.lw_rcnn_cls_1st = cfg.SOLVER.LW_RCNN_CLS_1ST
        self.lw_rcnn_reg_2nd = cfg.SOLVER.LW_RCNN_REG_2ND
        self.lw_rcnn_cls_2nd = cfg.SOLVER.LW_RCNN_CLS_2ND
        self.lw_rcnn_reg_3rd = cfg.SOLVER.LW_RCNN_REG_3RD
        self.lw_rcnn_cls_3rd = cfg.SOLVER.LW_RCNN_CLS_3RD
        self.lw_rcnn_reid_2nd = cfg.SOLVER.LW_RCNN_REID_2ND
        self.lw_rcnn_reid_3rd = cfg.SOLVER.LW_RCNN_REID_3RD
        self.lw_orthogonality_loss = cfg.SOLVER.LW_ORTHOGONALITY_LOSS

    def inference(self, images, targets=None, query_img_as_gallery=False):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)
            embeddings_2nd, _ = self.roi_heads.embedding_head_2nd(box_features_2nd)
            box_features_3rd = self.roi_heads.box_head_3rd(box_features)
            embeddings_3rd, _ = self.roi_heads.embedding_head_3rd(box_features_3rd)
            if self.eval_feat == 'concat':
                embeddings = torch.cat((embeddings_2nd, embeddings_3rd), dim=1)
            elif self.eval_feat == 'stage2':
                embeddings = embeddings_2nd
            elif self.eval_feat == 'stage3':
                embeddings = embeddings_3rd
            else:
                raise Exception("Unknown evaluation feature name")
            return embeddings.split(1, 0)
        else:
            # gallery
            boxes, _ = self.rpn(images, features, targets)
            detections = self.roi_heads(features, boxes, images.image_sizes, targets, query_img_as_gallery)[0]
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        boxes, rpn_losses = self.rpn(images, features, targets)

        _, rcnn_losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd = self.roi_heads(features, boxes, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        rpn_losses["loss_rpn_reg"] = rpn_losses.pop("loss_rpn_box_reg")
        rpn_losses["loss_rpn_cls"] = rpn_losses.pop("loss_objectness")

        losses = {}
        losses.update(rcnn_losses)
        losses.update(rpn_losses)
#        losses.update({"orthogonality_loss": self.roi_heads.orthogonality_loss})
        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_rcnn_reg_1st"] *= self.lw_rcnn_reg_1st
        losses["loss_rcnn_cls_1st"] *= self.lw_rcnn_cls_1st
        losses["loss_rcnn_reg_2nd"] *= self.lw_rcnn_reg_2nd
        losses["loss_rcnn_cls_2nd"] *= self.lw_rcnn_cls_2nd
        losses["loss_rcnn_reg_3rd"] *= self.lw_rcnn_reg_3rd
        losses["loss_rcnn_cls_3rd"] *= self.lw_rcnn_cls_3rd
        losses["loss_rcnn_reid_2nd"] *= self.lw_rcnn_reid_2nd
        losses["loss_rcnn_reid_3rd"] *= self.lw_rcnn_reid_3rd
#        losses["orthogonality_loss"] *= self.lw_orthogonality_loss
        return losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd

class CascadedROIHeads(RoIHeads):
    '''
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    '''
    def __init__(
        self,
        cfg,
        faster_rcnn_predictor,
        box_head_2nd,
        box_head_3rd,
        box_predictor2,  #新添加的头
        feat_len,
        *args,
        **kwargs
    ):
        super(CascadedROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh=cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST
        self.nms_thresh_2nd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.nms_thresh_3rd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD
        self.fg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND
        self.fg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD
        self.bg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD

        # Regression head
        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor
        self.box_predictor_3rd = box_predictor2   #deepcopy(self.box_predictor)  #self.box_predictor_3rd = deepcopy(self.box_predictor) 原来coat里面是这样


        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = box_head_2nd
        self.box_head_3rd = box_head_3rd

        # feature mask
        self.use_feature_mask = cfg.MODEL.USE_FEATURE_MASK
        self.feature_mask_size = cfg.MODEL.FEATURE_MASK_SIZE

        # Feature embedding
        embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.embedding_head_2nd = NormAwareEmbedding(in_channels=[int(feat_len / 2), feat_len])
        self.embedding_head_2nd_class = NormAwareEmbedding(in_channels=[int(feat_len / 2), feat_len])
        self.embedding_head_3rd = NormAwareEmbedding(in_channels=[int(feat_len / 2), feat_len])
        self.embedding_head_3rd_class = NormAwareEmbedding(in_channels=[int(feat_len / 2), feat_len])

#        self.embedding_head_2nd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)
#        self.embedding_head_3rd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)#deepcopy(self.embedding_head_2nd)
#        self.embedding_head_2nd_class = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)#这两行也是多出来的  目的是去 定义新的self.embedding_head_2nd_class和self.embedding_head_3rd_class
#        self.embedding_head_3rd_class = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)#deepcopy(self.embedding_head_2nd)

      #原来的coat的写法
        # embedding_dim = cfg.MODEL.EMBEDDING_DIM        #_C.MODEL.EMBEDDING_DIM = 256  #2-3阶段是 经过NAE得到特征嵌入
        # self.embedding_head_2nd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)
        # self.embedding_head_3rd = deepcopy(self.embedding_head_2nd)

        # OIM
        num_pids = cfg.MODEL.LOSS.LUT_SIZE
        num_cq_size = cfg.MODEL.LOSS.CQ_SIZE
        oim_momentum = cfg.MODEL.LOSS.OIM_MOMENTUM
        oim_scalar = cfg.MODEL.LOSS.OIM_SCALAR
        self.reid_loss_2nd = OIMLoss(embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.reid_loss_3rd = OIMLoss(embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar)#deepcopy(self.reid_loss_2nd)    # 原来coat写法 self.reid_loss_3rd = deepcopy(self.reid_loss_2nd)


#N个卷积层，每一层的输入和输出通道数分别为这些值  3个卷积效果好一点
        in_channels = [1024, 2048, 1024]
        out_channels = [1024, 2048, 1024]
        # in_channels = [1024, 2048]
        # out_channels = [1024, 2048]
        num_tasks = 2  #几个任务
        self.taskNAE = TaskInvariantModel(in_channels, out_channels, num_tasks)
#        self.orthogonality_loss = compute_orthogonality_loss(self.taskNAE.hierarchical_decoupling_before_trans,self.taskNAE.hierarchical_decoupling_after_trans)

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

        # evaluation        
        self.eval_feat = cfg.EVAL_FEATURE

    def forward(self, features, boxes, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        cws = True
        gt_det_2nd = None
        gt_det_3rd = None
        feats_reid_2nd = None
        feats_reid_3rd = None
        targets_reid_2nd = None
        targets_reid_3rd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False)
            boxes, _, box_pid_labels_1st, box_reg_targets_1st = self.select_training_samples(
                boxes, targets
            )

        # ------------------- The first stage ------------------ #
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
#        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(box_features_1st["after_trans"])
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(box_features_1st["feat_res5"])

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False)
            boxes, _, box_pid_labels_2nd, box_reg_targets_2nd = self.select_training_samples(boxes, targets)
        else:
            orig_thresh = self.nms_thresh # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.box_head_2nd(gt_box_features)
            embeddings, _ = self.embedding_head_2nd(gt_box_features)
            gt_det_2nd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det_2nd["boxes"] if gt_det_2nd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            if self.eval_feat == 'concat':
                embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_2nd["embeddings"]), dim=1) if gt_det_2nd else torch.zeros(0, 512)
            elif self.eval_feat == 'stage2' or self.eval_feat == 'stage3':
                embeddings = gt_det_2nd["embeddings"] if gt_det_2nd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- The second stage -------------------- #
#这是一个 Roi-Align 操作，用于从特征图中提取感兴趣区域（Region of Interest，ROI）对应的特征。features 是特征图，
# boxes 是感兴趣区域的边界框坐标，image_shapes 是输入图像的尺寸。
        box_features = self.box_roi_pool(features, boxes, image_shapes)  #Roi-Align操作
#该行将 Roi-Align 操作得到的区域特征传递给名为 box_head_2nd 的模型。通常这是物体检测网络的“头部”，用于从提取的区域特征中进一步提取更高级的特征表示。
        box_features = self.box_head_2nd(box_features)  #也就是res5 的输出 backbone的第二个输出
#        box_regs_2nd = self.box_predictor_2nd(box_features["after_trans"])  #边界框预测faster_rcnn_predictor 这个部分执行边界框预测的操作。
        box_regs_2nd = self.box_predictor_2nd(box_features["feat_res5"])
             #print(box_features["before_trans"].shape) torch.Size([384, 1024, 1, 1])
             #print(box_features["after_trans"].shape) torch.Size([384, 2048, 1, 1])

        # class 'dict'
        # Key: before_trans, Shape: torch.Size([384, 1024, 1, 1])
        # Key: after_trans, Shape: torch.Size([384, 2048, 1, 1])

        #box_features = self.taskNAE(box_features)

        # print(type(box_features))
        # for key, value in box_features.items():
        #     print(f"Key: {key}, Shape: {value.shape}")
        #pdb.set_trace()
        box_embeddings_2nd, _ = self.embedding_head_2nd(box_features)  #这个地方也就是用两个NAE去分别生成  box_embeddings_2nd 和 box_cls_scores_2nd
        _, box_cls_scores_2nd = self.embedding_head_2nd_class(box_features)  # self.embedding_head_2nd_class = NormAwareEmbedding 专门用一个NAE去结偶

#测试输出
# print(box_features.keys())
# dict_keys(['before_trans', 'after_trans'])

        #和coat不一样的地方
# box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(box_features)   也就是把这个统一输出的时候 分别输出成两个步骤了  以前是在一个 函数里面输出两个结果 现在是两个不同的函数 虽然都是NAE 去分别计算这两个值

        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        if self.training:
            boxes = self.get_boxes(box_regs_2nd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_3rd,
                    self.bg_iou_thresh_3rd,
                    allow_low_quality_matches=False)
            boxes, _, box_pid_labels_3rd, box_reg_targets_3rd = self.select_training_samples(boxes, targets)
        else:
            self.nms_thresh = self.nms_thresh_2nd
            if self.eval_feat != 'stage2':
                boxes, scores, _, _ = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=cws,
                )

        if not self.training and query_img_as_gallery and self.eval_feat != 'stage2':
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.box_head_3rd(gt_box_features)
            embeddings, _ = self.embedding_head_3rd(gt_box_features)
            gt_det_3rd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0 and self.eval_feat != 'stage2':
            assert not self.training
            boxes = gt_det_3rd["boxes"] if gt_det_3rd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            if self.eval_feat == 'concat':
                embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_3rd["embeddings"]), dim=1) if gt_det_3rd else torch.zeros(0, 512)
            elif self.eval_feat == 'stage3':
                embeddings = gt_det_2nd["embeddings"] if gt_det_3rd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- The third stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)

        if not self.training:
            box_features_2nd = self.box_head_2nd(box_features)
            box_embeddings_2nd, _ = self.embedding_head_2nd(box_features_2nd)
        box_features = self.box_head_3rd(box_features)
#        box_regs_3rd = self.box_predictor_3rd(box_features["after_trans"])
        box_regs_3rd = self.box_predictor_3rd(box_features["feat_res5"])
        #box_features = self.taskNAE(box_features)  # 任务不变 解耦NAE操作 分两个分支

        box_embeddings_3rd, _ = self.embedding_head_3rd(box_features)
        _, box_cls_scores_3rd = self.embedding_head_3rd_class(box_features)  #也是进行了拆分 多出来的 embedding_head_3rd_class函数 其实也就是NAE函数
        #原来coat的写法
        # box_embeddings_3rd, box_cls_scores_3rd = self.embedding_head_3rd(box_features)

        if box_cls_scores_3rd.dim() == 0:
            box_cls_scores_3rd = box_cls_scores_3rd.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            box_labels_3rd = [y.clamp(0, 1) for y in box_pid_labels_3rd]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                box_cls_scores_2nd,
                box_regs_2nd,
                box_labels_2nd,
                box_reg_targets_2nd,
                box_cls_scores_3rd,
                box_regs_3rd,
                box_labels_3rd,
                box_reg_targets_3rd,
            )

            loss_rcnn_reid_2nd, feats_reid_2nd, targets_reid_2nd = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd = self.reid_loss_3rd(box_embeddings_3rd, box_pid_labels_3rd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(loss_rcnn_reid_3rd=loss_rcnn_reid_3rd)
        else:
            if self.eval_feat == 'stage2':
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_2nd,
                    cws=cws,
                )
            else:
                self.nms_thresh = self.nms_thresh_3rd
                _, _, embeddings_2nd, _ = self.postprocess_boxes(
                    box_cls_scores_3rd,
                    box_regs_3rd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_2nd,
                    cws=cws,
                )
                boxes, scores, embeddings_3rd, labels = self.postprocess_boxes(
                    box_cls_scores_3rd,
                    box_regs_3rd,
                    box_embeddings_3rd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_3rd,
                    cws=cws,
                )
                # set to original thresh after finishing postprocess
                self.nms_thresh = orig_thresh

            num_images = len(boxes)
            for i in range(num_images):
                if self.eval_feat == 'concat':
                    embeddings = torch.cat((embeddings_2nd[i],embeddings_3rd[i]), dim=1)
                elif self.eval_feat == 'stage2':
                    embeddings = embeddings_2nd[i]
                elif self.eval_feat == 'stage3':
                    embeddings = embeddings_3rd[i]
                else:
                    raise Exception("Unknown evaluation feature name")
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings
                    )
                )

        return result, losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    box_cls_scores_1st,
    box_regs_1st,
    box_labels_1st,
    box_reg_targets_1st,
    box_cls_scores_2nd,
    box_regs_2nd,
    box_labels_2nd,
    box_reg_targets_2nd,
    box_cls_scores_3rd,
    box_regs_3rd,
    box_labels_3rd,
    box_reg_targets_3rd,
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st)    

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(box_labels_1st > 0).squeeze(1)
    labels_pos = box_labels_1st[sampled_pos_inds_subset]
    N = box_cls_scores_1st.size(0)
    box_regs_1st = box_regs_1st.reshape(N, -1, 4)

    loss_rcnn_reg_1st = F.smooth_l1_loss(
        box_regs_1st[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_1st[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_1st = loss_rcnn_reg_1st / box_labels_1st.numel()

    # --------------------- The second stage -------------------- #
    box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
    box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
    loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, box_labels_2nd.float())

    sampled_pos_inds_subset = torch.nonzero(box_labels_2nd > 0).squeeze(1)
    labels_pos = box_labels_2nd[sampled_pos_inds_subset]
    N = box_cls_scores_2nd.size(0)
    box_regs_2nd = box_regs_2nd.reshape(N, -1, 4)

    loss_rcnn_reg_2nd = F.smooth_l1_loss(
        box_regs_2nd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_2nd[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_2nd = loss_rcnn_reg_2nd / box_labels_2nd.numel()

    # --------------------- The third stage -------------------- #
    box_labels_3rd = torch.cat(box_labels_3rd, dim=0)
    box_reg_targets_3rd = torch.cat(box_reg_targets_3rd, dim=0)
    loss_rcnn_cls_3rd = F.binary_cross_entropy_with_logits(box_cls_scores_3rd, box_labels_3rd.float())

    sampled_pos_inds_subset = torch.nonzero(box_labels_3rd > 0).squeeze(1)
    labels_pos = box_labels_3rd[sampled_pos_inds_subset]
    N = box_cls_scores_3rd.size(0)
    box_regs_3rd = box_regs_3rd.reshape(N, -1, 4)

    loss_rcnn_reg_3rd = F.smooth_l1_loss(
        box_regs_3rd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_3rd[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_3rd = loss_rcnn_reg_3rd / box_labels_3rd.numel()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
        loss_rcnn_cls_3rd=loss_rcnn_cls_3rd,
        loss_rcnn_reg_3rd=loss_rcnn_reg_3rd,
    )
