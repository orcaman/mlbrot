# mlbrot

This is the companion repo for the blog post [Chaos, Prediction and Golang: Using AWS Machine Learning to Mispredict The Mandelbrot Set](https://hackernoon.com/chaos-prediction-and-golang-using-aws-machine-learning-to-mispredict-the-mandelbrot-set-45a87a72b2f#.ayoatlmtx).

![Alt text](/img/diff.jpg?raw=true "mlbrot")  

# usage
```
Usage of ./mlbrot:
  -c int
    	concurrency level (default 20)
  -method string
    	the coloring method. accepts "classic" and "ml" (default "classic")
  -n int
    	max lines to write. use 0 if you don't want to write a trainning data file
```