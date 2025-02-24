# object_detector_DL_HW4
implement an object detector from scratch.
The goal of our object detector is to find karts, bombs/projectiles, and pickup items. 

# Point-based object detection
You'll take your segmentation network and repurpose it for object detection. Specifically, you'll predict a dense heatmap of object centers, as shown below:
![image](https://github.com/user-attachments/assets/c0fc81d1-94c1-40fd-803e-27079bd2b829)

# Peak extraction
A peak in a heatmap is any point that is
-	a local maxima in a certain (rectangular) neighborhood (larger or equal to any neighboring point), and
-	has a value above a certain threshold.
