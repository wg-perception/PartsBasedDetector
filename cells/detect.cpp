#include <boost/scoped_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/ModelReader.h>
#include "PartsBasedDetector.hpp"
#include "MatlabIOModel.hpp"
#include "Visualize.hpp"

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
//using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDb;

namespace parts_based_detector
{
  struct PartsBasedDetectorCell
  {

    void
    ParameterCallback(const object_recognition_core::db::Documents& db_documents)
    {
    }

    static void
    declare_params(tendrils& params)
    {
      params.declare(&PartsBasedDetectorCell::visualize_, "visualize", "Visualize results", false);
      params.declare(&PartsBasedDetectorCell::model_file_, "model_file", "The path to the model file").required(true);

    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare(&PartsBasedDetectorCell::color_, "image", "An rgb full frame image.");
      inputs.declare(&PartsBasedDetectorCell::depth_, "depth", "The 16bit depth image.");
      //outputs.declare(&PartsBasedDetectorCell::pose_results_, "pose_results", "The results of object recognition");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
        // create the model object and deserialize it
        MatlabIOModel model;
        model.deserialize(*model_file_);

        // create the visualizer
        visualizer_.reset(new Visualize(model.name()));

        // create the PartsBasedDetector and distribute the model parameters
        detector_.reset(new PartsBasedDetector<double>);
        detector_->distributeModel(model);
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "detector: process" << std::endl;

      std::vector<Candidate> candidates;
      try 
      {
          detector_->detect(*color_, *depth_, candidates);
      } catch(const cv::Exception& e)
      {
      }

      if (*visualize_ && candidates.size() > 0)
      { 
        Candidate::sort(candidates);
        visualizer_->candidates(*color_, candidates, 1, true);
        cv::waitKey(30);
      }

      //pose_results_->clear();
      return ecto::OK;
    }


    // Parameters
    spore<bool> visualize_;
    spore<std::string> model_file_;

    // I/O
    spore<cv::Mat> color_, depth_;
    //ecto::spore<std::vector<PoseResult> > pose_results_;

    // the detector classes
    boost::scoped_ptr<Visualize> visualizer_;
    boost::scoped_ptr<PartsBasedDetector<double> > detector_;    

  };
}

ECTO_CELL(parts_based_cells, object_recognition_core::db::bases::ModelReaderBase<parts_based_detector::PartsBasedDetectorCell>, "Detector",
  "Detection of objects by parts");
