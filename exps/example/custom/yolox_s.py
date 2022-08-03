#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self, num_epochs):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "../forklift_data_coco"
        self.train_ann = "../forklift_data_coco/train/_annotations.coco.json"
        self.val_ann = "../forklift_data_coco/valid/_annotations.coco.json"
        self.test_ann = "../forklift_data_coco/test/_annotations.coco.json"

        self.num_classes = 3

        self.max_epoch = num_epochs
        self.data_num_workers = 4
        self.eval_interval = 1
