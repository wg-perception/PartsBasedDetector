"""
Module defining the Table Publisher
"""

from ecto_ros import Mat2Image
from ecto_ros.ecto_sensor_msgs import Publisher_Image
from object_recognition_core.io.sink import Sink
import ecto
from ecto import BlackBoxForward as Forward

########################################################################################################################

class Publisher(ecto.BlackBox):
    """
    Class publishing some outputs from the part based detection
    """
    def declare_cells(self, p):
        return {'image_converter': Mat2Image(swap_rgb=False)}

    def declare_forwards(self, _p):
        return ({},{'image_converter': [Forward('image')]},{})

    def configure(self, p, _i, _o):
        self.image_publisher = Publisher_Image(topic_name = 'by_parts_result')

    def connections(self, _p):
        connections = [self.image_converter['image'] >> self.image_publisher['input'] ]

        return connections

########################################################################################################################

class PublisherSink(Sink):

    @classmethod
    def type_name(cls):
        return 'by_parts_publisher'

    @classmethod
    def sink(self, *args, **kwargs):
        return Publisher()
