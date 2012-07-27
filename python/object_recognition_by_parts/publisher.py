"""
Module defining the Table Publisher
"""

from ecto_ros import Mat2Image
from ecto_ros.ecto_sensor_msgs import Publisher_Image
from object_recognition_core.io.sink import Sink
import ecto

########################################################################################################################

class Publisher(ecto.BlackBox):
    """
    Class publishing some outputs from the part based detection
    """
    _image_converter = Mat2Image
    _image_publisher = Publisher_Image

    def declare_params(self, p):
        #p.declare('marker_hull_topic', 'The ROS topic to use for the table message.', 'marker_table')
        pass

    def declare_io(self, _p, i, _o):
        i.forward('image', cell_name='_image_converter', cell_key='image')

    def configure(self, p, _i, _o):
        self._image_converter = Mat2Image(swap_rgb=True)
        self._image_publisher = Publisher_Image(topic_name = 'by_parts_result')

    def connections(self):
        connections = [self._image_converter['image'] >> self._image_publisher['input'] ]

        return connections

########################################################################################################################

class PublisherSink(Sink):

    @classmethod
    def type_name(cls):
        return 'by_parts_publisher'

    @classmethod
    def sink(self, *args, **kwargs):
        return Publisher()
