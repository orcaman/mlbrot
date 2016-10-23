package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"image"
	"image/color"
	"image/png"
	"os"

	"github.com/orcaman/mlbrot/ml"
)

type testResult struct {
	C     complex128
	Color color.RGBA
}

const (
	bailoutIteration = 30
	radius           = 2
	radiusSquared    = radius * radius

	imHeight = 800
	imWidth  = 800

	minRe        = -2.0
	maxRe        = 1.0
	minIm        = -1.2
	maxIm        = minIm + (maxRe-minRe)*imHeight/imWidth
	widthFactor  = (maxRe - minRe) / (imWidth - 1)
	heightFactor = (maxIm - minIm) / (imHeight - 1)
)

var (
	img              = image.NewRGBA(image.Rect(0, 0, imWidth, imHeight))
	writer           *bufio.Writer
	coloringMethod   = flag.String("method", "classic", "the coloring method. accepts classic and ml")
	concurrencyLevel = flag.Int("c", 20, "concurrency level")
	maxLinesToWrite  = flag.Int("n", 0, "max lines to write")
	linesWritten     = 0

	colorBlack = color.RGBA{255, 255, 255, 255}
	colorWhite = color.RGBA{0, 0, 0, 255}
	mlService  = ml.NewSvc()

	totalPixels  = imHeight * imWidth
	currentPixel = 0.0

	sem         chan bool
	imageMutex  = &sync.Mutex{}
	writerMutex = &sync.Mutex{}
)

func main() {
	flag.Parse()

	log.Printf("coloring method: %s", *coloringMethod)
	log.Printf("concurrency level: %d", *concurrencyLevel)

	sem = make(chan bool, *concurrencyLevel)

	f, err := openFile()
	writer = bufio.NewWriter(f)

	if *maxLinesToWrite > 0 || *maxLinesToWrite == -1 {
		if err != nil {
			log.Fatal(err.Error())
		}
		writer.WriteString(fmt.Sprintf("%s,%s,%s\n", "real", "imag", "result"))
	} else {
		log.Printf("no data file will be written")
		defer os.Remove(f.Name())
	}

	renderImage()

	writer.Flush()
	f.Close()

	saveImage(fmt.Sprintf("out_%s_%v.png", *coloringMethod, time.Now().UnixNano()))
}

func renderImage() {
	trackProgress()

	for i := 0; i < imHeight; i++ {
		for j := 0; j < imWidth; j++ {
			plotPixel(complex(float64(i), float64(j)))
			currentPixel++
		}
	}

	for i := 0; i < cap(sem); i++ {
		sem <- true
	}
}

func trackProgress() {
	ticker := time.NewTicker(time.Second * 1)
	fTotalPixels := float64(totalPixels)
	go func() {
		for range ticker.C {
			fmt.Printf("%v%% complete (%d/%d)\n", 100*currentPixel/fTotalPixels, int(currentPixel), totalPixels)
		}
	}()
}

// check if c belongs to set, assign color accordingly and plot
func plotPixel(c complex128) {
	sem <- true

	switch *coloringMethod {
	case "ml":
		go func(c complex128) {
			res := getColorFromMLMandelbrot(c)
			<-sem
			plot(res.C, res.Color)
		}(c)
	case "classic":
		go func(c complex128) {
			res := getColorFromClassicMandelbrot(c)
			<-sem
			plot(res.C, res.Color)
		}(c)
	}
}

func getColorFromClassicMandelbrot(c complex128) *testResult {
	b := belongsToSet(c)

	if *maxLinesToWrite == -1 || linesWritten < *maxLinesToWrite {
		writerMutex.Lock()
		writer.WriteString(fmt.Sprintf("%f,%f,%t\n", real(c), imag(c), b))
		writerMutex.Unlock()
		linesWritten++
	}

	if b {
		return &testResult{C: c, Color: colorBlack}
	}

	return &testResult{C: c, Color: colorWhite}
}

func belongsToSet(c complex128) bool {
	current := complex128(0)
	iterations := 0

	for {
		if iterations == bailoutIteration {
			break
		}

		if math.Pow(real(current), 2)+math.Pow(imag(current), 2) > radiusSquared {
			break
		}

		imC := maxIm - imag(c)*heightFactor
		reC := minRe + real(c)*widthFactor

		reNext := math.Pow(real(current), 2) - math.Pow(imag(current), 2) + reC
		imNext := 2*real(current)*imag(current) + imC
		current = complex(reNext, imNext)

		iterations++
	}

	if iterations == bailoutIteration {
		return false
	}

	return true
}

func plot(c complex128, col color.RGBA) {
	imageMutex.Lock()
	defer imageMutex.Unlock()
	img.Set(int(real(c)), int(imag(c)), col)
}

func saveImage(filename string) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()

	png.Encode(f, img)
}

func openFile() (*os.File, error) {
	return os.Create(fmt.Sprintf("data_%v.csv", time.Now().UnixNano()))
}

func getColorFromMLMandelbrot(c complex128) *testResult {
	b := ml.Predict(mlService, c)

	if b {
		return &testResult{C: c, Color: colorBlack}
	}

	return &testResult{C: c, Color: colorWhite}
}
