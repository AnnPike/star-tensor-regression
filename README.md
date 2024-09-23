Contains algorithms and experiment setting for my thesis work in a field of tensorial regression. Having a matrix with depth (3 dimentional tensor), for example some features over time and a target matrix, for example some target over time.
If we assume a linear connection between these features and the target we can solve it using simple Linear Regression by solving each depthwice dimenction for example, however it would't consider dependency in time domain. This is where the 
designed algorithm takes place.
![Tensoriaml Regressopn](https://github.com/AnnPike/star-tensor-regression/blob/master/2-1.png)
The proposed solution was extended with efficiency algorithm for fast and better convergence of 'tall' tensors
![Efficient solution](https://github.com/AnnPike/star-tensor-regression/blob/master/blendenpik_ill_coh.png)
