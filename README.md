## Prediction of turbulent heat transfer using convolutional neural networks
#### by Junhyuk Kim and Changhoon Lee
#### Journal of fluid mechanics, 2020, 882, A18
#### Article link : <https://doi.org/10.1017/jfm.2019.814>

![graphical abstract](graphical-abstract.jpg)

As shown in figure, we applied convolutional neural network(CNN) to the prediction of turbulent heat transfer in wall-bounded turbulence. CNN is constructed to predict the wall-normal heat flux based on the nearby other wall information including wall-shear stresses and pressure. As training results, very high prediction accuracy could be acceived even at a higher Reynolds number than the trained one. It stands for the existence of nonlinear relationship between the heat transfer and other wall information. We tried to figure out the spatial relationship between input and the heat transfer. The deep neural network is well-known as a black box method, however, through observation of a gradient map with statistical sense, we found that the trained CNN contains meaningful patterns that reveal underlying physics. In future work, our framework could be extended to other problems including turbulence reconstruction, turbulence control, and especially other turbulence physics analysis.  

Code and some data will be provided in this site soon. Our code is written in tensorflow library. In the code, minor things are omitted for simplicity. Data are composed of 2D flow fields including du/dy, dw/dy, dT/dy, and p.

If you have any questions about the article and the code, please email <junhyuk6@yonsei.ac.kr>.
