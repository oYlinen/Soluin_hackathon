Versatile Lightweight Pipeline to Segment and Label Objects from 3D Gaussian Splatting Data  

Project for the Since AI Hackathon 2026: Team Soluin  

Authors: Olli Ylinen, Jaakko Vihuri and Nokia Win  

Valmet Challenge: Segment and label objects from 3D Gaussian splatting data inside a factory  

Methods:  
Versitile lightweight pipeline to segment and label objects from 3D Gaussian splatting data, demo in a factory setting.  
1) DBSCAN (Density-based spatial clustering of applications with noise) algorithm to detect interesting point clouds  
2) Define target point classes  
3) Use pretrained 3D point cloud model PointNet++ to encode Point clouds -> Feature Vectors    
4) Calculate Cosine Similarity of feature vectors between target and query objects  
5) Interactive 3D Viewer Displays identified objects and object data such as condition and issue_code  

Setup:


We recommend to use python 3.10 and make a clean python
environment to begin with.  

INSTALL STEPS:  

1) git clone https://github.com/oYlinen/Soluin_hackathon  
cd Soluin_hackathon  
pip install -r requirements.txt  
Download point cloud data and have it on the Soluin_hackathon folder: https://drive.google.com/file/d/1MrwNjHxFqk5IfHpJQ24KlTn5nAAn40rd/view?pli=1  

2) Download the pre trained model PointNet++:  
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch  

3) Run the demo  
python main.py all  

Future Possibilities:  
The pipeline is extremely Versatile and the pieces can be changed or the scene can be easily changed and the objects are very easy and quick to add as target classes. The pretrained model can also be changed to a better one so that the object detection could be better.  




