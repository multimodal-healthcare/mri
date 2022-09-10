#  MRI
This is the repository for the Magnetic Resonance Imaging (MRI) modality of the pipeline. Together with other modality, we build a multi-modal AI model to diagnose Cardiovascular Disease (CVD). 

## Requirements
`python==3.9.13`  
`pydicom==2.3.0`  
`numpy==1.23.2`  
`pandas==1.4.4`  
`ipywidgets==8.0.2`  
`skimage==0.19.3`  
`matplotlib==3.5.3`  

## Data Processing Overview
### MRI Brain Scans 
We have in total 298 MRI scans recorded in the DICOM format (`*.dcm`). DICOM stands for Digital Imaging and Communications in Medicine. Each scan has the shape `(256, 256, 256)`, which means that there are 256 slices (indicated by the last index), each one is a 2D image with `(height, width) = (256, 256)` as illustrated in Figure 1. 

<figure>
<img align = "center" src="./images/brain_slice.png" width="200" height="200" />
<figcaption><b>Figure 1: A Slice from the MRI brain scan</b></figcaption>
</figure>

There are also slices that only contain noises, like in Figure 2 below. The noisy slice indices often range from 0-49 and 205-255. Therefore, on the  

<figure>
<img align = "center" src="./images/noisy_slice.png" width="200" height="200" />
<figcaption><b>Figure 1: Noisy Slice</b></figcaption>
</figure>