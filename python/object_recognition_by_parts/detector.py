#!/usr/bin/env python
"""
Module defining the parts based detector to find objects in a scene
"""

from object_recognition_core.db import ObjectDb, Models
from object_recognition_core.pipelines.detection import DetectionPipeline
from object_recognition_core.utils import json_helper
from ecto_image_pipeline.conversion import MatToPointCloudXYZOrganized
from object_recognition_by_parts_cells import Detector
import object_recognition_core
import object_recognition_by_parts_cells
import ecto
from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward

class PBDDetector(ecto.BlackBox):
  
    def __init__(self, object_db, object_ids, visualize, *args, **kwargs):
        self.object_db = object_db
        self.object_ids = object_ids
        self.visualize = visualize
        ecto.BlackBox.__init__(self, *args, **kwargs)

    def declare_cells(self, _p):
        return {'detector': CellInfo(Detector), 'mat_to_cloud': MatToPointCloudXYZOrganized()}

    def declare_forwards(self, _p):
        p = {'detector': 'all'}
        i = {'detector': [Forward('image'), Forward('depth'), Forward('K')],
             'mat_to_cloud': [Forward('points', 'points3d')]}
        o = {'detector': 'all', 'mat_to_cloud': [Forward('point_cloud', 'cloud_out')]}

        return (p,i,o)

    def connections(self, _p):
        return [ self.mat_to_cloud['point_cloud'] >> self.detector['input_cloud'] ]


#####################################################################################################################

class PartsBasedDetectionPipeline(DetectionPipeline):

    @classmethod
    def config_doc(cls):
        return  """
                    # The subtype can be any YAML that will help differentiate
                    # between TOD methods: we usually use the descriptor name
                    # but anything like parameters could be used
                    subtype:
                        type: ""
                    # TOD requires several parameters
                    parameters:
                        # TODO
                """

    @classmethod
    def type_name(cls):
        return 'by_parts'

    @classmethod
    def detector(self, *args, **kwargs):
        visualize = kwargs.pop('visualize', False)
        submethod = kwargs.pop('subtype')
        parameters = kwargs.pop('parameters')
        object_ids = parameters['object_ids']
        object_db = ObjectDb(parameters['db'])
        model_documents = Models(object_db, object_ids, self.type_name(), json_helper.dict_to_cpp_json_str(submethod))
        return PBDDetector(object_db, object_ids, visualize, *args, **parameters['extra'])
