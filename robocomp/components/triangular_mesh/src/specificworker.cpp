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

#include "specificworker.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <QImage>
#include <QPixmap>


SpecificWorker::SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check) : GenericWorker(configLoader, tprx)
{
	this->startup_check_flag = startup_check;
	if(this->startup_check_flag)
	{
		this->startup_check();
	}
	else
	{
		#ifdef HIBERNATION_ENABLED
			hibernationChecker.start(500);
		#endif
		
		// Example statemachine:
		/***
		//Your definition for the statesmachine (if you dont want use a execute function, use nullptr)
		states["CustomState"] = std::make_unique<GRAFCETStep>("CustomState", period, 
															std::bind(&SpecificWorker::customLoop, this),  // Cyclic function
															std::bind(&SpecificWorker::customEnter, this), // On-enter function
															std::bind(&SpecificWorker::customExit, this)); // On-exit function
		//Add your definition of transitions (addTransition(originOfSignal, signal, dstState))
		states["CustomState"]->addTransition(states["CustomState"].get(), SIGNAL(entered()), states["OtherState"].get());
		states["Compute"]->addTransition(this, SIGNAL(customSignal()), states["CustomState"].get()); //Define your signal in the .h file under the "Signals" section.
		//Add your custom state
		statemachine.addState(states["CustomState"].get());
		***/
		statemachine.setChildMode(QState::ExclusiveStates);
		statemachine.start();
		auto error = statemachine.errorString();
		if (error.length() > 0){
			qWarning() << error;
			throw error;
		}
	}
}

SpecificWorker::~SpecificWorker()
{
	std::cout << "Destroying SpecificWorker" << std::endl;
	if (pointcloud_window_ != nullptr)
	{
		pointcloud_window_->close();
		delete pointcloud_window_;
	}
}


void SpecificWorker::initialize()
{
    std::cout << "initialize worker" << std::endl;
	GenericWorker::initialize();
	setCameraIntrinsics(320.0f, 320.0f, 320.0f, 320.0f, 640, 640);
    //initializeCODE
    /////////GET PARAMS, OPEND DEVICES....////////
    //int period = configLoader.get<int>("Period.Compute") //NOTE: If you want get period of compute use getPeriod("compute")
    //std::string device = configLoader.get<std::string>("Device.name") 
}

void SpecificWorker::compute()
{
    // std::cout << "Compute worker" << std::endl;
	this->setCameraIntrinsics(320.0f, 320.0f, 320.0f, 320.0f, 640, 640);
	this->processCameraImage();
	this->buildPointCloudFromRGBD();
	this->displayPointCloudQt();
	//this->processLidarColorCloud();
	//this->buildMeshFromLidar();
	//this->displayMesh();
	//computeCODE
	//Main loop here
}

void SpecificWorker::processCameraImage()
{
	//? Get the image data from the camera proxy
	auto image_data = this->camerargbdsimple_proxy->getImage("camera1");
	RoboCompCameraRGBDSimple::TPoints points_data = this->camerargbdsimple_proxy->getPoints("camera1");
	//? Convert the image data to an OpenCV Mat and display it
	cv::Mat img(image_data.height, image_data.width, CV_8UC3, (void*)image_data.image.data());
	cv::Mat img_rgb;
	cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
	cv::imshow("RGB image", img_rgb);
	std::cout << "data: " << points_data.points.size() << std::endl;
	std::cout << "xyz: " << points_data.points[0].x << ", " << points_data.points[0].y << ", " << points_data.points[0].z << std::endl;
	cv::waitKey(1);
}

void SpecificWorker::processLidarColorCloud()
{
	try
	{
		// RoboCompLidar3D::TColorCloudData color_cloud_data = this->lidar3d_proxy->getColorCloudData();
		RoboCompLidar3D::TData lidar_data = this->lidar3d_proxy->getLidarData("lidar1", 0.0, 1.0, -1); // 0.0, 1.0, 1
		// std::cout << "Lidar Color Cloud Data retrieved successfully" << std::endl;
		// std::cout << "Number of points: " << color_cloud_data.numberPoints << std::endl;
		std::cout << "lidar" << lidar_data.points.size() << std::endl;
		std::cout << "xyz: " << lidar_data.points[0].x << ", " << lidar_data.points[0].y << ", " << lidar_data.points[0].z << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error getting Lidar color cloud data: " << e.what() << std::endl;
	}
}


void SpecificWorker::buildMeshFromLidar()
{
	try
	{
		// Get lidar data
		RoboCompLidar3D::TData lidar_data = this->lidar3d_proxy->getLidarData("lidar1", 0.0, 1.0, 1);
		
		if (lidar_data.points.size() < 3)
		{
			std::cerr << "Not enough lidar points to build mesh (need at least 3)" << std::endl;
			return;
		}
		
		std::cout << "Building mesh from " << lidar_data.points.size() << " lidar points..." << std::endl;
		
		// Clear previous mesh
		mesh_ = navmap::NavMap();
		
		// Add all points as vertices to the mesh
		std::vector<uint32_t> vertex_indices;
		vertex_indices.reserve(lidar_data.points.size());
		
		for (const auto& point : lidar_data.points)
		{
			Eigen::Vector3f pos(point.x, point.y, point.z);
			uint32_t vid = mesh_.add_vertex(pos);
			vertex_indices.push_back(vid);
		}
		
		std::cout << "Added " << vertex_indices.size() << " vertices" << std::endl;
		
		// Create a surface
		size_t surface_id = mesh_.create_surface("lidar_mesh");
		
		// Simple greedy triangulation: connect every 3 consecutive points
		// For better results, you could implement Delaunay or other algorithms
		size_t num_triangles = 0;
		for (size_t i = 0; i + 2 < vertex_indices.size(); i += 3)
		{
			// Create triangle from 3 consecutive vertices
			navmap::NavCelId tri_id = mesh_.add_navcel(
				vertex_indices[i],
				vertex_indices[i + 1],
				vertex_indices[i + 2]
			);
			
			// Add triangle to surface
			mesh_.add_navcel_to_surface(surface_id, tri_id);
			num_triangles++;
		}
		
		std::cout << "Created " << num_triangles << " triangles" << std::endl;
		std::cout << "Mesh built successfully!" << std::endl;
		std::cout << "Total vertices: " << mesh_.positions.size() << std::endl;
		std::cout << "Total triangles: " << mesh_.navcels.size() << std::endl;
		std::cout << "Total surfaces: " << mesh_.surfaces.size() << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error building mesh from lidar: " << e.what() << std::endl;
	}
}


void SpecificWorker::buildMeshFromRGBD()
{
	try
	{
		if (rgbd_point_cloud_.empty())
		{
			std::cerr << "RGBD point cloud is empty. Build it first using buildPointCloudFromRGBD()" << std::endl;
			return;
		}
		
		if (rgbd_point_cloud_.size() < 3)
		{
			std::cerr << "Not enough RGBD points to build mesh (need at least 3)" << std::endl;
			return;
		}
		
		std::cout << "Building mesh from " << rgbd_point_cloud_.size() << " RGBD points..." << std::endl;
		
		// Clear previous mesh
		mesh_ = navmap::NavMap();
		
		// Add all colored points as vertices to the mesh
		std::vector<uint32_t> vertex_indices;
		vertex_indices.reserve(rgbd_point_cloud_.size());
		
		for (const auto& point : rgbd_point_cloud_)
		{
			Eigen::Vector3f pos(point.x, point.y, point.z);
			uint32_t vid = mesh_.add_vertex(pos);
			vertex_indices.push_back(vid);
		}
		
		std::cout << "Added " << vertex_indices.size() << " colored vertices" << std::endl;
		
		// Create a surface for RGBD mesh
		size_t surface_id = mesh_.create_surface("rgbd_mesh");
		
		// Simple greedy triangulation: connect every 3 consecutive points
		// For better results, you could implement Delaunay or other algorithms
		size_t num_triangles = 0;
		for (size_t i = 0; i + 2 < vertex_indices.size(); i += 3)
		{
			// Create triangle from 3 consecutive vertices
			navmap::NavCelId tri_id = mesh_.add_navcel(
				vertex_indices[i],
				vertex_indices[i + 1],
				vertex_indices[i + 2]
			);
			
			// Add triangle to surface
			mesh_.add_navcel_to_surface(surface_id, tri_id);
			num_triangles++;
		}
		
		std::cout << "Created " << num_triangles << " triangles from colored point cloud" << std::endl;
		std::cout << "RGBD Mesh built successfully!" << std::endl;
		std::cout << "Total vertices: " << mesh_.positions.size() << std::endl;
		std::cout << "Total triangles: " << mesh_.navcels.size() << std::endl;
		std::cout << "Total surfaces: " << mesh_.surfaces.size() << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error building mesh from RGBD: " << e.what() << std::endl;
	}
}


void SpecificWorker::displayMesh()
{
	try
	{
		if (mesh_.positions.size() < 3)
		{
			std::cerr << "Mesh is empty or has insufficient vertices" << std::endl;
			return;
		}
		
		std::cout << "Displaying mesh with " << mesh_.navcels.size() << " triangles..." << std::endl;
		
		// Find min/max coordinates for scaling
		float min_x = mesh_.positions.x[0], max_x = mesh_.positions.x[0];
		float min_y = mesh_.positions.y[0], max_y = mesh_.positions.y[0];
		float min_z = mesh_.positions.z[0], max_z = mesh_.positions.z[0];
		
		for (size_t i = 0; i < mesh_.positions.size(); ++i)
		{
			min_x = std::min(min_x, mesh_.positions.x[i]);
			max_x = std::max(max_x, mesh_.positions.x[i]);
			min_y = std::min(min_y, mesh_.positions.y[i]);
			max_y = std::max(max_y, mesh_.positions.y[i]);
			min_z = std::min(min_z, mesh_.positions.z[i]);
			max_z = std::max(max_z, mesh_.positions.z[i]);
		}
		
		// Create visualization canvas
		int canvas_width = 1000;
		int canvas_height = 1000;
		cv::Mat canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(255, 255, 255));
		
		// Calculate scale factors
		float x_range = (max_x - min_x > 0.1f) ? (max_x - min_x) : 1.0f;
		float y_range = (max_y - min_y > 0.1f) ? (max_y - min_y) : 1.0f;
		float scale_x = (canvas_width - 40) / x_range;
		float scale_y = (canvas_height - 40) / y_range;
		
		// Draw grid lines
		cv::line(canvas, cv::Point(20, canvas_height/2), cv::Point(canvas_width-20, canvas_height/2), cv::Scalar(200, 200, 200), 1);
		cv::line(canvas, cv::Point(canvas_width/2, 20), cv::Point(canvas_width/2, canvas_height-20), cv::Scalar(200, 200, 200), 1);
		
		// Draw all triangles
		for (size_t i = 0; i < mesh_.navcels.size(); ++i)
		{
			const auto& navcel = mesh_.navcels[i];
			
			// Get the three vertices of the triangle
			Eigen::Vector3f v0 = mesh_.positions.at(navcel.v[0]);
			Eigen::Vector3f v1 = mesh_.positions.at(navcel.v[1]);
			Eigen::Vector3f v2 = mesh_.positions.at(navcel.v[2]);
			
			// Project to 2D (top-down view: X-Y plane)
			cv::Point p0(20 + (v0.x() - min_x) * scale_x, canvas_height - 20 - (v0.y() - min_y) * scale_y);
			cv::Point p1(20 + (v1.x() - min_x) * scale_x, canvas_height - 20 - (v1.y() - min_y) * scale_y);
			cv::Point p2(20 + (v2.x() - min_x) * scale_x, canvas_height - 20 - (v2.y() - min_y) * scale_y);
			
			// Ensure points are within canvas
			p0.x = std::max(0, std::min(canvas_width - 1, p0.x));
			p0.y = std::max(0, std::min(canvas_height - 1, p0.y));
			p1.x = std::max(0, std::min(canvas_width - 1, p1.x));
			p1.y = std::max(0, std::min(canvas_height - 1, p1.y));
			p2.x = std::max(0, std::min(canvas_width - 1, p2.x));
			p2.y = std::max(0, std::min(canvas_height - 1, p2.y));
			
			// Color based on average Z height
			float avg_z = (v0.z() + v1.z() + v2.z()) / 3.0f;
			uint8_t color_value = static_cast<uint8_t>(255.0f * (avg_z - min_z) / (max_z - min_z + 0.001f));
			cv::Scalar tri_color(255 - color_value, 100 + color_value / 2, 100);  // Color gradient based on height
			
			// Draw triangle filled
			cv::Point triangle[1][3] = { {p0, p1, p2} };
			const cv::Point* ppt[1] = {triangle[0]};
			int npt[] = {3};
			cv::polylines(canvas, ppt, npt, 1, true, cv::Scalar(0, 0, 0), 1);  // Triangle outline
			cv::fillPoly(canvas, ppt, npt, 1, tri_color, cv::LINE_AA);  // Triangle fill
		}
		
		// Add vertex markers (larger points at vertices)
		int num_vertices_to_show = std::min(static_cast<int>(mesh_.positions.size()), 100);
		for (int i = 0; i < num_vertices_to_show; ++i)
		{
			cv::Point p(20 + (mesh_.positions.x[i] - min_x) * scale_x, 
					    canvas_height - 20 - (mesh_.positions.y[i] - min_y) * scale_y);
			p.x = std::max(0, std::min(canvas_width - 1, p.x));
			p.y = std::max(0, std::min(canvas_height - 1, p.y));
			cv::circle(canvas, p, 2, cv::Scalar(0, 0, 0), -1);
		}
		
		// Add text information
		std::string title = "Mesh Visualization (Top-Down View) | " + std::to_string(mesh_.navcels.size()) + " triangles";
		cv::putText(canvas, title, cv::Point(25, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
		
		char range_text[250];
		snprintf(range_text, sizeof(range_text), "X: [%.2f, %.2f]  Y: [%.2f, %.2f]  Z: [%.2f, %.2f]  Vertices: %zu",
				min_x, max_x, min_y, max_y, min_z, max_z, mesh_.positions.size());
		cv::putText(canvas, range_text, cv::Point(25, canvas_height - 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
		
		// Display the canvas
		cv::imshow("Mesh Visualization", canvas);
		cv::waitKey(1);
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error displaying mesh: " << e.what() << std::endl;
	}
}


void SpecificWorker::buildPointCloudFromRGBD()
{
	try
	{
		// Get image and points data from RGBD camera
		auto image_data = this->camerargbdsimple_proxy->getImage("camera1");
		RoboCompCameraRGBDSimple::TPoints points_data = this->camerargbdsimple_proxy->getPoints("camera1");
		
		if (points_data.points.empty())
		{
			std::cerr << "No RGBD points available" << std::endl;
			return;
		}
		
		// Convert image to OpenCV Mat for color access
		cv::Mat img(image_data.height, image_data.width, CV_8UC3, (void*)image_data.image.data());
		
		// Clear previous point cloud
		rgbd_point_cloud_.clear();
		rgbd_point_cloud_.reserve(points_data.points.size());
		
		std::cout << "Building RGBD point cloud from " << points_data.points.size() << " points..." << std::endl;
		std::cout << "Camera intrinsics: fx=" << camera_intrinsics_.fx << ", fy=" << camera_intrinsics_.fy
					<< ", cx=" << camera_intrinsics_.cx << ", cy=" << camera_intrinsics_.cy << std::endl;
		
		int valid_points = 0;
		
		// Build colored point cloud with proper 3D-to-2D projection
		for (size_t i = 0; i < points_data.points.size(); ++i)
		{
			const auto& point = points_data.points[i];
			
			// Skip invalid/zero/negative Z points
			if (point.z <= 0.001f || point.x == 0 && point.y == 0 && point.z == 0)
				continue;
			
			// Project 3D point to 2D using camera intrinsics
			// pixel_x = fx * (point.x / point.z) + cx
			// pixel_y = fy * (point.y / point.z) + cy
			float pixel_x = (camera_intrinsics_.fx * point.x / point.z) + camera_intrinsics_.cx;
			float pixel_y = (camera_intrinsics_.fy * point.y / point.z) + camera_intrinsics_.cy;
			
			// Round to nearest integer
			int px = static_cast<int>(std::round(pixel_x));
			int py = static_cast<int>(std::round(pixel_y));
			
			// Check if pixel is within image bounds
			if (px < 0 || px >= image_data.width || py < 0 || py >= image_data.height)
				continue;
			
			// Get color from image (OpenCV uses BGR format)
			cv::Vec3b bgr = img.at<cv::Vec3b>(py, px);
			
			// Create colored point
			ColoredPoint cp;
			cp.x = point.x;
			cp.y = point.y;
			cp.z = point.z;
			cp.b = bgr[0];  // Blue
			cp.g = bgr[1];  // Green
			cp.r = bgr[2];  // Red
			
			rgbd_point_cloud_.push_back(cp);
			valid_points++;
		}
		
		std::cout << "RGBD point cloud built successfully with " << valid_points << " valid colored points out of " 
					<< points_data.points.size() << " total points" << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error building RGBD point cloud: " << e.what() << std::endl;
	}
}


void SpecificWorker::setCameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height)
{
	camera_intrinsics_.fx = fx;
	camera_intrinsics_.fy = fy;
	camera_intrinsics_.cx = cx;
	camera_intrinsics_.cy = cy;
	camera_intrinsics_.width = width;
	camera_intrinsics_.height = height;
	
	std::cout << "Camera intrinsics updated:" << std::endl;
	std::cout << "  Focal length (fx, fy): (" << fx << ", " << fy << ")" << std::endl;
	std::cout << "  Principal point (cx, cy): (" << cx << ", " << cy << ")" << std::endl;
	std::cout << "  Image size: " << width << " x " << height << std::endl;
}


void SpecificWorker::displayPointCloudQt()
{
	try
	{
		if (rgbd_point_cloud_.empty())
		{
			std::cerr << "Point cloud is empty. Build it first using buildPointCloudFromRGBD()" << std::endl;
			return;
		}
		
		std::cout << "Displaying point cloud with " << rgbd_point_cloud_.size() << " points in Qt window..." << std::endl;
		
		// Create or reuse the Qt window
		if (pointcloud_window_ == nullptr)
		{
			pointcloud_window_ = new QMainWindow();
			pointcloud_window_->setWindowTitle("3D Point Cloud Viewer (RGBD)");
			pointcloud_window_->resize(900, 800);
		}
		
		// Find min/max coordinates for scaling
		float min_x = rgbd_point_cloud_[0].x, max_x = rgbd_point_cloud_[0].x;
		float min_y = rgbd_point_cloud_[0].y, max_y = rgbd_point_cloud_[0].y;
		float min_z = rgbd_point_cloud_[0].z, max_z = rgbd_point_cloud_[0].z;
		
		for (const auto& point : rgbd_point_cloud_)
		{
			min_x = std::min(min_x, point.x);
			max_x = std::max(max_x, point.x);
			min_y = std::min(min_y, point.y);
			max_y = std::max(max_y, point.y);
			min_z = std::min(min_z, point.z);
			max_z = std::max(max_z, point.z);
		}
		
		// Create visualization canvas (3 views: top-down, front, side)
		int view_width = 300;
		int view_height = 300;
		int padding = 10;
		int total_width = view_width * 3 + padding * 4;
		int total_height = view_height + padding * 2;
		
		cv::Mat canvas(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));
		
		// Calculate scale factors
		float x_range = (max_x - min_x > 0.1f) ? (max_x - min_x) : 1.0f;
		float y_range = (max_y - min_y > 0.1f) ? (max_y - min_y) : 1.0f;
		float z_range = (max_z - min_z > 0.1f) ? (max_z - min_z) : 1.0f;
		
		float scale_xy_x = (view_width - 20) / x_range;
		float scale_xy_y = (view_height - 20) / y_range;
		float scale_xz_x = (view_width - 20) / x_range;
		float scale_xz_z = (view_height - 20) / z_range;
		float scale_yz_y = (view_width - 20) / y_range;
		float scale_yz_z = (view_height - 20) / z_range;
		
		// Helper lambda to draw a point on canvas
		auto draw_point = [&](cv::Mat& img, int view_x, int view_y, int view_w, int view_h,
							float scale_u, float scale_v, float u_min, float v_min,
							const ColoredPoint& pt, int radius = 2) {
			int px = padding + view_x + 10 + (pt.x - u_min) * scale_u;
			int py = padding + view_y + 10 + (pt.y - v_min) * scale_v;
			px = std::max(padding, std::min(total_width - 1, px));
			py = std::max(padding, std::min(total_height - 1, py));
			cv::Scalar color(pt.b, pt.g, pt.r);  // Use actual RGB color
			cv::circle(img, cv::Point(px, py), radius, color, -1);
		};
		
		// View 1: Top-down (X-Y plane)
		for (const auto& pt : rgbd_point_cloud_)
		{
			draw_point(canvas, 0, 0, view_width, view_height, 
						scale_xy_x, scale_xy_y, min_x, min_y, pt, 2);
		}
		cv::rectangle(canvas, cv::Point(padding, padding), 
						cv::Point(padding + view_width, padding + view_height), 
						cv::Scalar(0, 0, 0), 2);
		cv::putText(canvas, "Top-Down (X-Y)", cv::Point(padding + 5, padding + 20), 
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
		
		// View 2: Front (X-Z plane)
		for (const auto& pt : rgbd_point_cloud_)
		{
			int px = padding + view_width + padding * 2 + 10 + (pt.x - min_x) * scale_xz_x;
			int py = padding + view_height - 10 - (pt.z - min_z) * scale_xz_z;
			px = std::max(padding + view_width + padding * 2, std::min(total_width - 1, px));
			py = std::max(padding, std::min(total_height - 1, py));
			cv::Scalar color(pt.b, pt.g, pt.r);
			cv::circle(canvas, cv::Point(px, py), 2, color, -1);
		}
		cv::rectangle(canvas, cv::Point(padding + view_width + padding * 2, padding),
					  cv::Point(padding + view_width * 2 + padding * 2, padding + view_height),
						cv::Scalar(0, 0, 0), 2);
		cv::putText(canvas, "Front (X-Z)", cv::Point(padding + view_width + padding * 2 + 5, padding + 20),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
		
		// View 3: Side (Y-Z plane)
		for (const auto& pt : rgbd_point_cloud_)
		{
			int px = padding + view_width * 2 + padding * 3 + 10 + (pt.y - min_y) * scale_yz_y;
			int py = padding + view_height - 10 - (pt.z - min_z) * scale_yz_z;
			px = std::max(padding + view_width * 2 + padding * 3, std::min(total_width - 1, px));
			py = std::max(padding, std::min(total_height - 1, py));
			cv::Scalar color(pt.b, pt.g, pt.r);
			cv::circle(canvas, cv::Point(px, py), 2, color, -1);
		}
		cv::rectangle(canvas, cv::Point(padding + view_width * 2 + padding * 3, padding),
					  cv::Point(padding + view_width * 3 + padding * 3, padding + view_height),
						cv::Scalar(0, 0, 0), 2);
		cv::putText(canvas, "Side (Y-Z)", cv::Point(padding + view_width * 2 + padding * 3 + 5, padding + 20),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
		
		// Add statistics
		char stats[300];
		snprintf(stats, sizeof(stats), "Points: %zu | X: [%.2f, %.2f] | Y: [%.2f, %.2f] | Z: [%.2f, %.2f]",
					rgbd_point_cloud_.size(), min_x, max_x, min_y, max_y, min_z, max_z);
		cv::putText(canvas, stats, cv::Point(padding, total_height - 5),
					cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
		
		// Convert OpenCV Mat to Qt pixmap and display
		cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
		QImage qimg(canvas.data, canvas.cols, canvas.rows, canvas.step, QImage::Format_RGB888);
		QPixmap pixmap = QPixmap::fromImage(qimg);
		
		QLabel* label = new QLabel();
		label->setPixmap(pixmap);
		label->setScaledContents(true);
		pointcloud_window_->setCentralWidget(label);
		pointcloud_window_->show();
		
		std::cout << "Point cloud visualization displayed successfully!" << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cerr << "Error displaying point cloud: " << e.what() << std::endl;
	}
}


void SpecificWorker::emergency()
{
    std::cout << "Emergency worker" << std::endl;
    //emergencyCODE
    //
    //if (SUCCESSFUL) //The componet is safe for continue
    //  emmit goToRestore()
}


//Execute one when exiting to emergencyState
void SpecificWorker::restore()
{
    std::cout << "Restore worker" << std::endl;
    //restoreCODE
    //Restore emergency component
}


int SpecificWorker::startup_check()
{
	std::cout << "Startup check" << std::endl;
	QTimer::singleShot(200, QCoreApplication::instance(), SLOT(quit()));
	return 0;
}



/**************************************/
// From the RoboCompCameraRGBDSimple you can call this methods:
// RoboCompCameraRGBDSimple::TRGBD this->camerargbdsimple_proxy->getAll(string camera)
// RoboCompCameraRGBDSimple::TDepth this->camerargbdsimple_proxy->getDepth(string camera)
// RoboCompCameraRGBDSimple::TImage this->camerargbdsimple_proxy->getImage(string camera)
// RoboCompCameraRGBDSimple::TPoints this->camerargbdsimple_proxy->getPoints(string camera)

/**************************************/
// From the RoboCompCameraRGBDSimple you can use this types:
//  
// RoboCompCameraRGBDSimple::TPoints
// RoboCompCameraRGBDSimple::TImage
// RoboCompCameraRGBDSimple::TDepth
// RoboCompCameraRGBDSimple::TRGBD

/**************************************/
// From the RoboCompLidar3D you can call this methods:
// RoboCompLidar3D::TColorCloudData this->lidar3d_proxy->getColorCloudData()
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarData(string name, float start, float len, int decimationDegreeFactor)
// RoboCompLidar3D::TDataImage this->lidar3d_proxy->getLidarDataArrayProyectedInImage(string name)
// RoboCompLidar3D::TDataCategory this->lidar3d_proxy->getLidarDataByCategory(TCategories categories, long timestamp)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataProyectedInImage(string name)
// RoboCompLidar3D::TData this->lidar3d_proxy->getLidarDataWithThreshold2d(string name, float distance, int decimationDegreeFactor)

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData
// RoboCompLidar3D::TDataCategory
// RoboCompLidar3D::TColorCloudData

/**************************************/
// From the RoboCompOmniRobot you can call this methods:
// RoboCompOmniRobot::void this->omnirobot_proxy->correctOdometer(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBasePose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->getBaseState(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->resetOdometer()
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometer(RoboCompGenericBase::TBaseState state)
// RoboCompOmniRobot::void this->omnirobot_proxy->setOdometerPose(int x, int z, float alpha)
// RoboCompOmniRobot::void this->omnirobot_proxy->setSpeedBase(float advx, float advz, float rot)
// RoboCompOmniRobot::void this->omnirobot_proxy->stopBase()

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

