# BUILDING 3D URBAN RECONSTRUCTION
# Building3D_Urban


INTRODUCTION

Building3D is an urban-scale dataset for building roof modelling from aerial LiDAR point clouds. It consists of more than 160 thousand buildings, covering 16 cities in Estonia about 998 Km2. Besides mesh models and real-world LiDAR point clouds, it is the first time to release wireframe models. We believe that our Building3D will facilitate future research on urban modelling, mesh simplification, point cloud completion and semantic segmentation etc. please refer to our paper and website for details.

DATASET

The Building3D dataset is only available for download to users with educational email addresses and non-commercial account. Please visit our website to download the dataset. In this competition, we will not approve accounts with email addresses from free email providers, such as gmail.com, qq.com, web.de, etc. Only university or company email addresses will be eligible for prize awards.

Participants must test their methods on Tallinn City datasets. The final ranking is determined by WED.

Data Type
Image 1
Building Point Clouds
Each building point clouds are stored in XYZ format including XYZ coordinates, RGB colour, near-infrared information, intensity and reflectance.

Image 2
Roof Point Clouds
For 3D roof reconstruction, it doesn't involve facade point clouds. Thus, the roof point clouds only retain all the points representing roof structure.

Image 3
Mesh
Building mesh models are created from aerial LiDAR point clouds and building footprints by using the Terrasolid software, and then modified by hand.

Image 4
Wireframe
Wireframe models are a very simple 3D representation. It consists of vertexes and edges.



Ref. https://huggingface.co/spaces/Building3D/USM3D

