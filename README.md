# KU-ComputerVision
Computer Vision class of Korea University (Prof. Christian Wallraven)


# Assignment1

## Result 1

| Lambertian Reflectance | Projection |
|---|---|
|![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/gif/sphere2d.gif)|![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/gif/sphere3d.gif)|

## Result 2

![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/renderer.gif)

# Assianment2

## Result

| 	filter_func (mean time)	| filter_func (std time)	| opencv (mean time)	| opencv (std time)  |
|---|---|---|---|
| size3	0:00:02.006593	| 0:00:00.065378	| 0:00:00.000227	| 0:00:00.000044 |
| size5	0:00:01.966926	| 0:00:00.043482	| 0:00:00.000509	| 0:00:00.000106 |
| size9	0:00:01.972027	| 0:00:00.047833	| 0:00:00.001441	| 0:00:00.000050 |
| size15	0:00:02.054571	| 0:00:00.053821	| 0:00:00.003206	| 0:00:00.000458 |
| size23	0:00:02.176916	| 0:00:00.069011	| 0:00:00.003177	| 0:00:00.000519 |


## Discussion

The results can be interpreted from two perspectives. The first is a comparison of time changes as filter size changes in my filter function. The second is the comparison of computation time between the two functions.

First of all, There is no big difference when the filter size is 3,5,9, but it took longer from 15 in my filter function. I think the amount of time spent on the filter increases linearly. The correlation between the filter size and the total operation is closed to one for a 500 x 500 pixels gray-scale image (figure 1). The reason is that the number of the filter weights increases with $n^2$ (eq. 1), but the number of operations between the image and the filter decreases (eq. 2) as follows. 

$$the\ number\ of\ filter\ weights  = n^2 \tag{eq. 1}$$ 
$$the\ number\ of\ operation = (w - n + 1) \times (h - n + 1) \tag{eq. 2}$$

where `h` is a image height and `w` is a image weight. `n` is a filter size.

<div align='center'>
    <img src='https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment2/corr_op_f.jpg'>
    Figure 1. the correlation between the filter size and the number of total operation
</div>

As a result, the compututation time is affected by other factors

Second is that opencv `filter2D` is much faster than my filter functions. The reason is that opencv code is based on C++. Comparing Python vs C++ speed reveals which executes faster and creates more time-efficient programs. One thing to note that C++ compiles the code and Python interprets the code. The interpretation of code is always slower than the compilation. 

In shortly, I believe that using opencv is more beneficial both in time and effectively for the reasons I have mentioned above

