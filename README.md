# KU-ComputerVision
Computer Vision class of Korea University ([Prof. Christian Wallraven](https://koreauniv.pure.elsevier.com/en/persons/christian-wallraven))

# Members
- Jaehyuk Heo 2021020630 ([DSBA Lab](http://dsba.korea.ac.kr)) [[Blog](https://tootouch.github.io) | [Github](https://github.com/TooTouch)]
- Hyeongwon Kang 2021020634 ([DSBA Lab](http://dsba.korea.ac.kr)) [[Blog](https://hwk0702.github.io) | [Github](https://github.com/hwk0702)]

# Assignment1

- Answer1 [jupyter notebook](https://github.com/TooTouch/KU-ComputerVision/tree/main/Assignment1/sphere.ipynb)
- Answer2 [jupyter notebook](https://github.com/TooTouch/KU-ComputerVision/tree/main/Assignment1/renderer.ipynb)

## Result 1

| Lambertian Reflectance | Projection |
|---|---|
|![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/gif/sphere2d.gif)|![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/gif/sphere3d.gif)|

## Result 2

![](https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment1/renderer.gif)

# Assianment2

- Answer [jupyter notebook](https://github.com/TooTouch/KU-ComputerVision/tree/main/Assignment2/filtering.ipynb)

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

First of all, There is no big difference when the filter size is 3,5,9, but it took longer from 15 in my filter function. I think the amount of time spent on the filter increases linearly. The correlation between the filter size and the total operation is closed to one for a 500 x 500 pixels gray-scale image (figure 1). The reason is that the number of the filter weights increases with <img alt="formula" src="https://render.githubusercontent.com/render/math?math=n^2" /> (eq. 1), but the number of operations between the image and the filter decreases (eq. 2) as follows. 

<p align='center'>
    <img src="https://render.githubusercontent.com/render/math?math=the%5C%20number%5C%20of%5C%20filter%5C%20weights%20%20%3D%20n%5E2%20%5Ctag%7BEq.%201%7D"> (eq. 1)<br>  
    <img src="https://render.githubusercontent.com/render/math?math=the%5C%20number%5C%20of%5C%20operation%20%3D%20(w%20-%20n%20%2B%201)%20%5Ctimes%20(h%20-%20n%20%2B%201)"> (eq. 2)
</p>

where `h` is a image height and `w` is a image weight. `n` is a filter size.

<p align='center'>
    <img src='https://github.com/TooTouch/KU-ComputerVision/blob/main/Assignment2/corr_op_f.jpg'><br>
    Figure 1. the correlation between the filter size and the number of total operation
</p>

As a result, the compututation time is affected by other factors

Second is that opencv `filter2D` is much faster than my filter functions. The reason is that opencv code is based on C++. Comparing Python vs C++ speed reveals which executes faster and creates more time-efficient programs. One thing to note that C++ compiles the code and Python interprets the code. The interpretation of code is always slower than the compilation. 

In shortly, I believe that using opencv is more beneficial both in time and effectively for the reasons I have mentioned above

