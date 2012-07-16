#!/usr/bin/env python
"""
Module defining the transparent objects detector to find objects in a scene
"""

from object_recognition_core.db import ObjectDb, Models
from object_recognition_core.pipelines.detection import DetectionPipeline
from object_recognition_core.utils import json_helper
import object_recognition_by_parts_cells

########################################################################################################################

class PartsBasedDetectionPipeline(DetectionPipeline):
    @classmethod
    def type_name(cls):
        return 'by_parts'

    @classmethod
    def detector(self, *args, **kwargs):
        visualize = kwargs.pop('visualize', False)
        submethod = kwargs.pop('submethod')
        parameters = kwargs.pop('parameters')
        object_ids = parameters['object_ids']
        object_db = ObjectDb(parameters['db'])
        model_documents = Models(object_db, object_ids, self.type_name(), json_helper.dict_to_cpp_json_str(submethod))
        model_file = parameters['extra'].get('model_file')
        return object_recognition_by_parts_cells.Detector(model_file=model_file, visualize=visualize)
