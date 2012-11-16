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

class PBDDetector(ecto.BlackBox):
    mat_to_cloud = MatToPointCloudXYZOrganized
    detector = Detector
  
  
    def __init__(self, object_db, object_ids, model_file, visualize, max_overlap, remove_planes, *args, **kwargs):
        self.object_db = object_db
        self.object_ids = object_ids
        self.model_file = model_file
        self.visualize = visualize
        self.max_overlap = max_overlap
        self.remove_planes = remove_planes
        ecto.BlackBox.__init__(self, *args, **kwargs)
    
    def declare_params(self, p):
        #p.forward("model_file", cell_name="detector")
        #p.forward("visualize", cell_name="detector")
        #p.forward("max_overlap", cell_name="detector")
        #p.forward("db", cell_name="detector")
        pass
      
    def declare_io(self, _p, i, o):
        i.forward("image", cell_name="detector")
        i.forward("depth", cell_name="detector")
        i.forward("K", cell_name="detector")
        i.forward("points3d", cell_name="mat_to_cloud", cell_key="points")
        
        o.forward("pose_results", cell_name="detector")
        o.forward("image", cell_name="detector")
        o.forward("point3d_clusters", cell_name="detector")
        o.forward("cloud_out", cell_name="mat_to_cloud", cell_key="point_cloud")
    
    def configure(self, p, _i, _o):
        params = { 'db': self.object_db, 'model_file' : self.model_file, 'visualize': self.visualize, 'max_overlap': self.max_overlap , 'remove_planes': self.remove_planes }
        self.detector = Detector("PB detector", **params)
        pass
  
    def connections(self):
        return [ self.mat_to_cloud['point_cloud'] >> self.detector['input_cloud'] ]


#####################################################################################################################

class PartsBasedDetectionPipeline(DetectionPipeline):
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
        model_file = parameters['extra'].get('model_file')
        max_overlap = parameters['extra'].get('max_overlap', 0.1)
        remove_planes = parameters['extra'].get('remove_planes', False)
        return PBDDetector(object_db, object_ids, model_file, visualize, max_overlap, remove_planes, *args, **kwargs)
