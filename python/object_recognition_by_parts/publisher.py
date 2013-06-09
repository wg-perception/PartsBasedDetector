"""
Module defining the Table Publisher
"""

from ecto_ros import Mat2Image
from ecto_ros.ecto_sensor_msgs import Publisher_Image
from object_recognition_core.io.sink import SinkBase
import ecto
from ecto import BlackBoxForward as Forward

########################################################################################################################

class Publisher(ecto.BlackBox, SinkBase):
    """
    Class publishing some outputs from the part based detection
    """
    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        #SinkBase.__init__(self)

    @staticmethod
    def declare_cells(p):
        return {'image_converter': Mat2Image(swap_rgb=False)}

    @staticmethod
    def declare_forwards(_p):
        return ({},{'image_converter': [Forward('image')]},{})

    def configure(self, p, _i, _o):
        self.image_publisher = Publisher_Image(topic_name = 'by_parts_result')

    def connections(self, _p):
        connections = [self.image_converter['image'] >> self.image_publisher['input'] ]

        return connections
