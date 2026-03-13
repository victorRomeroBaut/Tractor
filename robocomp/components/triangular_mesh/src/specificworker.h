/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <Eigen/Core>
#include <QMainWindow>
#include <QLabel>
#include "navmap/core/NavMap.hpp"
#include "navmap/core/Geometry.hpp"


/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
Q_OBJECT
public:
    /**
     * \brief Constructor for SpecificWorker.
     * \param configLoader Configuration loader for the component.
     * \param tprx Tuple of proxies required for the component.
     * \param startup_check Indicates whether to perform startup checks.
     */
	SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

	/**
     * \brief Destructor for SpecificWorker.
     */
	~SpecificWorker();


public slots:

	/**
	 * \brief Initializes the worker one time.
	 */
	void initialize();

	/**
	 * \brief Main compute loop of the worker.
	 */
	void compute();

	/**
	 * \brief Handles the emergency state loop.
	 */
	void emergency();

	/**
	 * \brief Restores the component from an emergency state.
	 */
	void restore();

    /**
     * \brief Performs startup checks for the component.
     * \return An integer representing the result of the checks.
     */
	int startup_check();

private:

	/**
     * \brief Flag indicating whether startup checks are enabled.
     */
	bool startup_check_flag;

	/**
	 * \brief Processes camera image: converts from BGR to RGB and displays it.
	 */
	void processCameraImage();
	/**
	\brief Retrieves and processes Lidar color cloud data.
	 */
	void processLidarColorCloud();
	/**
	\brief Displays up to 100 lidar points in a 2D top-down view.
	 */
	void displayLidarPoints();
	/**
	\brief Builds a mesh from lidar point cloud data using NavMap library.
	 */
	void buildMeshFromLidar();

	/**
	\brief Displays the constructed mesh in 2D top-down view.
	 */
	void displayMesh();

	/**
	 \brief Builds a colored point cloud from RGBD camera data.
	 */
	void buildPointCloudFromRGBD();

	/**
	 \brief Sets camera intrinsic parameters for proper 3D-to-2D projection.
	 \param fx Focal length in X (pixels)
	 \param fy Focal length in Y (pixels)
	 \param cx Principal point X (pixels)
	 \param cy Principal point Y (pixels)
	 \param width Image width
	 \param height Image height
	 */
	void setCameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height);

	/**
	 \brief Displays the colored RGBD point cloud in a Qt window with 3D visualization.
	 */
	void displayPointCloudQt();

	/**
	 \brief Constructs a mesh from RGBD point cloud data using NavMap library.
	 \note The point cloud must be built first using buildPointCloudFromRGBD().
	 */
	void buildMeshFromRGBD();

private:
	/// NavMap mesh container
	navmap::NavMap mesh_;
	/// Colored point cloud from RGBD (XYZ + RGB)
	struct ColoredPoint {
		float x, y, z;      // Position
		uint8_t r, g, b;    // Color (BGR from OpenCV)
	};
	std::vector<ColoredPoint> rgbd_point_cloud_;
	
	/// Camera intrinsic parameters
	struct CameraIntrinsics {
		float fx{320.0f};   // Focal length X (pixels)
		float fy{320.0f};   // Focal length Y (pixels)
		float cx{320.0f};   // Principal point X (pixels)
		float cy{240.0f};   // Principal point Y (pixels)
		int width{640};     // Image width
		int height{480};    // Image height
	} camera_intrinsics_;	
	/// Qt window for point cloud visualization
	QMainWindow* pointcloud_window_{nullptr};signals:
	//void customSignal();
};

#endif
