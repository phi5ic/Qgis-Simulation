Advanced Water Leak Detection System for QGIS

This project is a high-performance spatial analysis tool designed to identify water pipeline breaches by combining Machine Learning (Random Forest), Dijkstra’s Pathfinding, and Copernicus SAR (Synthetic Aperture Radar) Imagery. By layering real-world satellite moisture data over local sensor networks, the system provides a robust "double-verification" for leak detection.
## Core Features

    Machine Learning Prediction: Uses a SimpleRandomForest classifier to analyze moisture indices, pressure drops, and flow rate anomalies.

    SAR Data Integration: Incorporates Sentinel-1 GRD imagery from the Copernicus Browser to detect surface moisture changes through satellite backscatter.

    3-Month Historical Baseline: Compares live readings against a 90-day seasonal mean to filter out environmental noise from actual leaks.

    Dijkstra’s Bypass Analysis: Automatically calculates the total length of safe vs. affected pipeline segments, aiding in rapid maintenance routing.

    Automated QGIS Layering: Dynamically generates color-coded sensor maps (MASTER_SENSORS) and high-priority danger zones (LEAK_LOCATION_ZONES).

## Technical Workflow

    Data Acquisition: User loads a pipeline/road network (LineString) and overlays a SAR raster from Copernicus.

    ML Classification: The system trains on sensor data to establish "Normal" vs "Leak" profiles based on pressure and flow.

    SAR Anomaly Detection: The script analyzes the VV polarization band from Sentinel-1 to identify high-moisture hotspots along the network.

    Spatial Correlation: Leaks are "confirmed" only when both the ML model and SAR anomaly data converge on the same coordinates.

    Bypass Calculation: Dijkstra’s algorithm measures the network impact and provides a percentage-based breakdown of safe operation zones.

## SAR Integration Guide (Copernicus)

To move from simulation to real-world monitoring, follow these steps to integrate Copernicus data:
### Recommended: Sentinel Hub QGIS Plugin

    Account: Create a free account at the Copernicus Data Space Ecosystem.

    Plugin: Install the Sentinel Hub plugin in QGIS.

    Data Selection: Select Sentinel-1 GRD with VV Polarization.

    Baseline Comparison: Use a 3-month average for the "Historical Baseline" to ensure detection accuracy.

### Analysis Parameters
Parameter	Value	Reason
Satellite	Sentinel-1	Penetrates cloud cover and works at night.
Polarization	VV	Optimal sensitivity for surface water and soil moisture.
Buffer Distance	50 Meters	Standard safety radius used for Dijkstra bypass calculations.
## Installation & Usage
### Prerequisites

    QGIS 3.x

    A LineString layer (Roads or Pipelines).

    pandas, numpy, and PyQt5 (standard in QGIS Python environment).

### Running the System

    Open your QGIS project and load your network layer.

    Open the Python Console (Plugins > Python Console).

    Paste the contents of standalone_final_fixed_Version3.py.

    Run the script. The console will output detailed Accuracy, Precision, and Network Impact metrics.

## Performance Metrics

The system outputs a comprehensive report to the console including:

    ML Metrics: Accuracy, F1-Score, and a Confusion Matrix.

    SAR Analysis: Count of high-confidence moisture anomalies (>75%).

    Route Stats: Total Network Length (km) and Safe Route Percentage (%).

    Note: This system is optimized for Emergency Deployment. For the most accurate results, ensure your road/pipeline layer is projected in a CRS that uses meters (like a local UTM zone).
    
