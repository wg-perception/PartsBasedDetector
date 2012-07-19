"""
Module defining the Table Publisher
"""

from object_recognition_core.ecto_cells.io_ros import Publisher_Marker, Publisher_MarkerArray
from object_recognition_core.io.sink import Sink
from object_recognition_msgs.ecto_cells.ecto_object_recognition_msgs import Publisher_TableArray
import ecto

MarkerPub = Publisher_Marker
MarkerArrayPub = Publisher_MarkerArray

########################################################################################################################

class TablePublisher(ecto.BlackBox):
    """
    Class publishing the different results of tabletop
    """
    _table_msg_assembler = TableMsgAssembler
    _table_visualization_msg_assembler = TableVisualizationMsgAssembler
    _marker_array_hull_ = MarkerArrayPub
    _marker_array_origin_ = MarkerArrayPub
    _marker_array_table_ = MarkerArrayPub
    _marker_array_delete = MarkerArrayPub
    _marker_array_clusters = MarkerArrayPub
    _table_array = Publisher_TableArray

    def declare_params(self, p):
        p.declare('marker_hull_topic', 'The ROS topic to use for the table message.', 'marker_table')
        p.declare('marker_origin_topic', 'The ROS topic to use for the table message.', 'marker_table')
        p.declare('marker_table_topic', 'The ROS topic to use for the table message.', 'marker_table')
        p.declare('marker_array_delete', 'The ROS topic to use for the markers to remove.', 'marker_table')
        p.declare('marker_array_clusters', 'The ROS topic to use for the markers of the clusters.', 'marker_array_clusters')
        p.declare('table_array', 'The array of found tables.', 'table_array')
        p.declare('latched', 'Determines if the topics will be latched.', True)

    def declare_io(self, _p, i, _o):
        self.passthrough = ecto.PassthroughN(items=dict(image_message='The original imagemessage',
                                                        pose_results='The final results'))

        i.forward(['clouds', 'clouds_hull'], cell_name='_table_msg_assembler',
                  cell_key=['clouds', 'clouds_hull'])
        i.forward(['clusters', 'table_array_msg'], cell_name='_table_visualization_msg_assembler',
                  cell_key=['clusters', 'table_array_msg'])
        i.forward('image_message', cell_name='passthrough', cell_key='image_message')
        i.forward('pose_results', cell_name='passthrough', cell_key='pose_results')
        

    def configure(self, p, _i, _o):
        self._table_msg_assembler = TablePublisher._table_msg_assembler()
        self._table_visualization_msg_assembler = TablePublisher._table_visualization_msg_assembler()
        self._marker_array_hull_ = TablePublisher._marker_array_hull_(topic_name=p.marker_hull_topic, latched=p.latched)
        self._marker_array_origin_ = TablePublisher._marker_array_origin_(topic_name=p.marker_origin_topic, latched=p.latched)
        self._marker_array_table_ = TablePublisher._marker_array_table_(topic_name=p.marker_table_topic, latched=p.latched)
        self._marker_array_delete = TablePublisher._marker_array_delete(topic_name=p.marker_array_delete)
        self._marker_array_clusters = TablePublisher._marker_array_clusters(topic_name=p.marker_array_clusters)
        self._table_array = TablePublisher._table_array(topic_name=p.table_array)

    def connections(self):
        connections = [self.passthrough['image_message'] >> self._table_msg_assembler['image_message'],
                       self.passthrough['image_message'] >> self._table_visualization_msg_assembler['image_message'], 
                       self.passthrough['pose_results'] >> self._table_msg_assembler['pose_results'],
                       self.passthrough['pose_results'] >> self._table_visualization_msg_assembler['pose_results'] ]

        connections += [ self._table_msg_assembler['table_array_msg'] >> self._table_array[:],
                        self._table_msg_assembler['table_array_msg'] >> self._table_visualization_msg_assembler['table_array_msg'] ]
        connections += [self._table_visualization_msg_assembler['marker_array_hull'] >> self._marker_array_hull_[:],
                self._table_visualization_msg_assembler['marker_array_origin'] >> self._marker_array_origin_[:],
                self._table_visualization_msg_assembler['marker_array_table'] >> self._marker_array_table_[:],
                self._table_visualization_msg_assembler['marker_array_delete'] >> self._marker_array_delete[:],
                self._table_visualization_msg_assembler['marker_array_clusters'] >> self._marker_array_clusters[:] ]
        return connections

########################################################################################################################

class TablePublisherSink(Sink):

    @classmethod
    def type_name(cls):
        return 'publisher'

    @classmethod
    def sink(self, *args, **kwargs):
        return Publisher()
