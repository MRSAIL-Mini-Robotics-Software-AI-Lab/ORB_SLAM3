#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<vector>

#include<opencv2/core/core.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include<System.h>

using namespace std;

namespace py = pybind11;


class OrbSlam3Py {
public:
    OrbSlam3Py(const string &strVocFile, const string &strSettingsFile, const string &cameraType){
        if(cameraType == "stereo"){
            slam_system = new ORB_SLAM3::System(strVocFile,strSettingsFile,ORB_SLAM3::System::STEREO,true);
        }
    }
    std::vector<double> TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp){
        Sophus::SE3f pose = slam_system->TrackStereo(imLeft, imRight, timestamp);
        float* rot_quaternion = pose.data();
        Eigen::Matrix<float, 3, 1> translation = pose.translation();
        return {translation[0], translation[1], translation[2], rot_quaternion[0], rot_quaternion[1], rot_quaternion[2], rot_quaternion[3]};
    }
    ~OrbSlam3Py(){
        delete slam_system;
    }
private:
    ORB_SLAM3::System* slam_system;
};


PYBIND11_MODULE(orbslam3_py, m) {
    py::class_<OrbSlam3Py>(m, "OrbSlam3Py")
        .def(py::init<const string&, const string&, const string&>())
        .def("TrackStereo", [](OrbSlam3Py& instance, py::array_t<uint8_t> imLeft, py::array_t<uint8_t> imRight, double timestamp){
            py::buffer_info infoLeft = imLeft.request();
            int heightLeft = infoLeft.shape[0];
            int widthLeft = infoLeft.shape[1];
            uint8_t* dataLeft =static_cast<uint8_t*>(infoLeft.ptr);

            cv::Mat cvImgLeft(heightLeft, widthLeft, CV_8UC3, dataLeft);

            py::buffer_info infoRight = imRight.request();
            int heightRight = infoRight.shape[0];
            int widthRight = infoRight.shape[1];
            uint8_t* dataRight =static_cast<uint8_t*>(infoRight.ptr);

            cv::Mat cvImgRight(heightRight, widthRight, CV_8UC3, dataRight);

            py::gil_scoped_release release;
            return instance.TrackStereo(cvImgLeft, cvImgRight, timestamp);
        });
}