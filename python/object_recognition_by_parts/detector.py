#!/usr/bin/env python
"""
Module defining the parts based detector to find objects in a scene
"""

from object_recognition_core.db import ObjectDb, Models
from object_recognition_core.pipelines.detection import DetectorBase
from object_recognition_core.utils import json_helper
from ecto_image_pipeline.conversion import MatToPointCloudXYZOrganized
from object_recognition_by_parts_cells import Detector
import object_recognition_core
import object_recognition_by_parts_cells
import ecto
from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward

class PartsBasedDetector(ecto.BlackBox, DetectorBase):
  
    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @staticmethod
    def declare_cells(_p):
        return {'detector': CellInfo(Detector), 'mat_to_cloud': MatToPointCloudXYZOrganized()}

    @staticmethod
    def declare_forwards(_p):
        p = {'detector': 'all'}
        i = {'detector': [Forward('image'), Forward('depth'), Forward('K')],
             'mat_to_cloud': [Forward('points', 'points3d')]}
        o = {'detector': 'all', 'mat_to_cloud': [Forward('point_cloud', 'cloud_out')]}

        return (p,i,o)

    def connections(self, _p):
        return [ self.mat_to_cloud['point_cloud'] >> self.detector['input_cloud'] ]
