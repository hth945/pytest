import tensorflow as tf
from detection.models import resnet,rpn_head,fpn,roi_align,bbox_head
from detection.core.bbox import bbox_target

class FasterRCNN(tf.keras.Model):

    def __init__(self, num_classes, **kwags):
        super(FasterRCNN, self).__init__(**kwags)
       
        self.NUM_CLASSES = num_classes
        # RPN configuration
        # Anchor attributes
        self.ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)
        self.ANCHOR_FEATURE_STRIDES = (4, 8, 16, 32, 64)

        # Bounding box refinement mean and standard deviation
        self.RPN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)

        # RPN training configuration
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5
        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        # ROIs kept configuration
        self.PRN_PROPOSAL_COUNT = 2000
        self.PRN_NMS_THRESHOLD = 0.7

        # RCNN configuration
        # Bounding box refinement mean and standard deviation
        self.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)

        # ROI Feat Size
        self.POOL_SIZE = (7, 7)

        # RCNN training configuration
        self.RCNN_BATCH_SIZE = 256
        self.RCNN_POS_FRAC = 0.25
        self.RCNN_POS_IOU_THR = 0.5
        self.RCNN_NEG_IOU_THR = 0.5

        # Boxes kept configuration
        self.RCNN_MIN_CONFIDENCE = 0.7
        self.RCNN_NME_THRESHOLD = 0.3
        self.RCNN_MAX_INSTANCES = 100


        self.backbone = resnet.ResNet(depth=50, name='res_net')
        self.neck = fpn.FPN(name='fpn')
        self.rpn_head = rpn_head.RPNHead(
            anchor_scales=self.ANCHOR_SCALES,
            anchor_ratios=self.ANCHOR_RATIOS,
            anchor_feature_strides=self.ANCHOR_FEATURE_STRIDES,
            proposal_count=self.PRN_PROPOSAL_COUNT,
            nms_threshold=self.PRN_NMS_THRESHOLD,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rpn_deltas=self.PRN_BATCH_SIZE,
            positive_fraction=self.RPN_POS_FRAC,
            pos_iou_thr=self.RPN_POS_IOU_THR,
            neg_iou_thr=self.RPN_NEG_IOU_THR,
            name='rpn_head')

        self.roi_align = roi_align.PyramidROIAlign(pool_shape=self.POOL_SIZE, name='pyramid_roi_align')

        self.bbox_head = bbox_head.BBoxHead(
            num_classes=self.NUM_CLASSES,
            pool_size=self.POOL_SIZE,
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RCNN_TARGET_STDS,
            min_confidence=self.RCNN_MIN_CONFIDENCE,
            nms_threshold=self.RCNN_NME_THRESHOLD,
            max_instances=self.RCNN_MAX_INSTANCES,
            name='b_box_head')

        # Target Generator for the second stage.
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=self.RCNN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR,
            neg_iou_thr=self.RCNN_NEG_IOU_THR)


    def call(self, inputs, training=True):

        ###################back one###################
        if training: # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
        else: # inference
            imgs, img_metas = inputs
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5],training=training)
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]

        ###################rpn + proposals###################
        # [1, 369303, 2] [1, 369303, 2], [1, 369303, 4],
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        # [369303, 4] => [215169, 4], valid => [6000, 4], performance =>[2000, 4],  NMS
        proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas)

        ###################roi pooling###################
        if training: # get target value for these proposal target label and target delta
            rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list =self.bbox_target.build_targets(proposals_list, gt_boxes, gt_class_ids, img_metas)
            # proposal中选出来的框,再根据标签中的数据计算iou,获得iou超过多少的正标签,和小于多少的负标签
            # rois_list选出来的roi(归一化了).
            # rcnn_target_matchs_list 选出来的框的标签, 0为背景
            # rcnn_target_deltas_list 选出来的框的偏移偏差(归一化了)(正标签有效)

            # positive_roi_ix = tf.where(rcnn_target_matchs_list[0] > 0)
            # print('1  ',tf.shape(positive_roi_ix),tf.shape(rcnn_target_matchs_list), tf.shape(rcnn_target_deltas_list))

        else:
            rois_list = proposals_list


        ####################self RPN########################
        # if training:
        #     rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)
        #     return [rpn_class_loss, rpn_bbox_loss]
        # else:
        #     proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas, with_probs=True)
        #     return proposals_list



        # rois_list only contains coordinates, rcnn_feature_maps save the 5 features data=>[192,7,7,256] # [2000,7,7,256]
        pooled_regions_list = self.roi_align((rois_list, rcnn_feature_maps, img_metas), training=training)

        ################### class ###################
        # [192, 81], [192, 81], [192, 81, 4]
        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = self.bbox_head(pooled_regions_list, training=training)

        if training:
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas)

            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(rcnn_class_logits_list, rcnn_deltas_list, rcnn_target_matchs_list, rcnn_target_deltas_list)

            # return [rpn_class_loss, rpn_bbox_loss,rcnn_class_loss, rcnn_bbox_loss],rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list
            return [rpn_class_loss, rpn_bbox_loss, rcnn_class_loss,rcnn_bbox_loss]
        else:
            # print(rcnn_probs_list)
            # proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas,with_probs=True)
            #
            # return proposals_list

            detections_list = self.bbox_head.get_bboxes(rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)
            return detections_list

        return 0

