package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

type ModelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

var modelSes ModelSession

var (
	modelPath           string
	imagePath           string
	useCoreML           bool
	classesPath         string
	yolover             string
	confidenceThreshold float64
	nmsThreshold        float64
)

var (
	ClassesOffset = 4
	NPreds        = 8400
	PredSize      = 84
	NClasses      = 80
)

var local bool = false

var yolo_classes = []string{}

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	var err error
	// 设置默认值
	flag.StringVar(&modelPath, "m", "./best.onnx", "ONNX 模型文件路径")
	flag.StringVar(&imagePath, "p", "./test.jpg", "图像文件路径")
	flag.StringVar(&classesPath, "c", "./classes.txt", "YOLO 类别文件路径")
	flag.Float64Var(&confidenceThreshold, "con", 0.6, "confidenceThreshold（置信度阈值）")
	flag.Float64Var(&nmsThreshold, "nms", 0.5, "nmsThreshold（非极大值抑制阈值）")
	flag.StringVar(&yolover, "ver", "v8", "YOLO version")
	flag.BoolVar(&local, "l", false, "run in local")
	flag.Parse()
	yolo_classes, err = loadClasses(classesPath)
	if err != nil {
		panic(err)
	}
	if yolover == "v5" {
		NClasses = len(yolo_classes)
		ClassesOffset = 5
		NPreds = 25200
		PredSize = NClasses + ClassesOffset
	}
	modelSes, err = initSession()
	if err != nil {
		panic(err)
	}
}

// 加载类别文件到 yolo_classes
func loadClasses(classesPath string) ([]string, error) {
	file, err := os.Open(classesPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewScanner(file)
	for reader.Scan() {
		// 移除每行末尾的换行符并添加到 classes 中
		line := strings.TrimSpace(reader.Text())
		yolo_classes = append(yolo_classes, line)
	}
	if err := reader.Err(); err != nil {
		return nil, err
	}
	return yolo_classes, nil
}

func main() {
	server := http.Server{Addr: "0.0.0.0:8080"}
	http.HandleFunc("/", index)
	http.HandleFunc("/detect", detect)
	if local {
		test()
	} else {
		server.ListenAndServe()
	}
}

func test() {
	file, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	img, imgtype, err := image.Decode(file)
	if err != nil {
		log.Printf("imgtype:%v err:%v\n", imgtype, err)
	}
	startTime := time.Now()
	boxes, _ := detect_objects_on_image(img)
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("total elapsedTime: %v ms\n", elapsedTime.Milliseconds())
	buf, _ := json.Marshal(&boxes)
	log.Printf("result: %s\n", buf)
}

func index(w http.ResponseWriter, _ *http.Request) {
	file, _ := os.Open("index.html")
	buf, _ := io.ReadAll(file)
	w.Write(buf)
}

func detect(w http.ResponseWriter, r *http.Request) {
	r.ParseMultipartForm(0)
	file, _, _ := r.FormFile("image_file")
	img, imgtype, err := image.Decode(file)
	if err != nil {
		log.Printf("imgtype:%v err:%v\n", imgtype, err)
	}
	startTime := time.Now()
	boxes, err := detect_objects_on_image(img)
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("total elapsedTime: %v ms\n", elapsedTime.Milliseconds())
	if err != nil {
		log.Println(err.Error())
	}
	buf, _ := json.Marshal(&boxes)
	w.Write(buf)
}

func detect_objects_on_image(buf image.Image) ([][]interface{}, error) {
	startTime := time.Now()
	input, img_width, img_height := prepare_input(buf)
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("prepare_input took: %v ms\n", elapsedTime.Milliseconds())
	output, err := run_model(input)
	if err != nil {
		return nil, err
	}
	data := process_output(output, img_width, img_height)
	return data, nil
}

func initSession() (ModelSession, error) {
	ort.SetSharedLibraryPath("./third_party/libonnxruntime.so")
	err := ort.InitializeEnvironment()
	if err != nil {
		return ModelSession{}, err
	}

	inputShape := ort.NewShape(1, 3, 640, 640)
	//inputTensor, err := ort.NewTensor(inputShape, blank)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return ModelSession{}, err
	}
	outputShape := ort.NewShape(1, int64(len(yolo_classes)+4), 8400)
	if yolover == "v5" {
		outputShape = ort.NewShape(1, 25200, int64(len(yolo_classes)+5))
	}
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return ModelSession{}, err
	}

	options, e := ort.NewSessionOptions()
	if e != nil {
		return ModelSession{}, err
	}

	if useCoreML { // If CoreML is enabled, append the CoreML execution provider
		e = options.AppendExecutionProviderCoreML(0)
		if e != nil {
			options.Destroy()
			return ModelSession{}, err
		}
		defer options.Destroy()
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, options)

	modelSess := ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}

	return modelSess, err
}

/*
要准备 YOLOv8 模型的输入，首先加载图像，调整其大小并转换为 (3,640,640) 的张量，
其中第一项是图像像素的红色分量数组，第二项是绿色分量数组，最后一个是蓝色数组。
此外，Go 的 ONNX 库要求输入这个张量作为一维数组，例如将这三个数组一个接一个地连接起来
*/
func prepare_input(img image.Image) ([]float32, int64, int64) {
	/*
		这段代码完成了加载图像，并将其大小调整为 640x640 像素。
		然后将像素的颜色分到不同的数组中：
	*/
	size := img.Bounds().Size()
	img_width, img_height := int64(size.X), int64(size.Y)
	img = resize.Resize(640, 640, img, resize.Lanczos3)
	red := []float32{}
	green := []float32{}
	blue := []float32{}
	//接着需要从图像中提取像素及其颜色，并把他们归一化，代码如下
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			red = append(red, float32(r/257)/255.0)
			green = append(green, float32(g/257)/255.0)
			blue = append(blue, float32(b/257)/255.0)
		}
	}
	// 最后，以正确的顺序将这些数组连接成一个数组：
	input := append(red, green...)
	input = append(input, blue...)
	return input, img_width, img_height
}

func run_model(input []float32) ([]float32, error) {
	inTensor := modelSes.Input.GetData()
	copy(inTensor, input)
	startTime := time.Now()
	err := modelSes.Session.Run()
	if err != nil {
		return nil, err
	}
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("Run onnx took: %v ms\n", elapsedTime.Milliseconds())
	//该outputTensor.GetData()方法以浮点数字的一维数组形式返回输出数据。
	//该函数返回形状为 (1,84,8400) 的数组，可以将其视为大约 84x8400 矩阵。它以一维数组的形式返回。
	return modelSes.Output.GetData(), nil
}

/*
NClasses=len(yolo_classes)
ClassesOffset=5
NPreds=25200
PredSize=NClasses+ClassesOffset

	startTime := time.Now()
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("run onnx took: %v ms\n", elapsedTime.Milliseconds())
*/
func process_output(output []float32, img_width, img_height int64) [][]interface{} {
	startTime := time.Now()
	boxes := [][]interface{}{}
	if yolover == "v5" {
		for index := 0; index < 25200; index++ {
			if float64(output[PredSize*index+4]) < confidenceThreshold {
				continue
			}
			class_id, prob := 0, float32(0.0)
			for col := 0; col < NClasses; col++ {
				if output[PredSize*index+5+col] > prob {
					prob = output[PredSize*index+5+col]
					class_id = col
				}
			}
			if prob > 0.5 {
				label := yolo_classes[class_id]
				xc := output[PredSize*index]
				yc := output[PredSize*index+1]
				w := output[PredSize*index+2]
				h := output[PredSize*index+3]
				x1 := (xc - w/2) / 640 * float32(img_width)
				y1 := (yc - h/2) / 640 * float32(img_height)
				x2 := (xc + w/2) / 640 * float32(img_width)
				y2 := (yc + h/2) / 640 * float32(img_height)
				boxes = append(boxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, prob})
			}
		}
	} else {
		for index := 0; index < 8400; index++ {
			class_id, prob := 0, float32(0.0)
			for col := 0; col < NClasses; col++ {
				if output[8400*(col+4)+index] > prob {
					prob = output[8400*(col+4)+index]
					class_id = col
				}
			}
			if float64(prob) < confidenceThreshold {
				continue
			}
			label := yolo_classes[class_id]
			xc := output[index]
			yc := output[8400+index]
			w := output[2*8400+index]
			h := output[3*8400+index]
			x1 := (xc - w/2) / 640 * float32(img_width)
			y1 := (yc - h/2) / 640 * float32(img_height)
			x2 := (xc + w/2) / 640 * float32(img_width)
			y2 := (yc + h/2) / 640 * float32(img_height)
			boxes = append(boxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, prob})
		}
	}
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i][5].(float32) < boxes[j][5].(float32)
	})
	result := [][]interface{}{}
	for len(boxes) > 0 {
		result = append(result, boxes[0])
		tmp := [][]interface{}{}
		for _, box := range boxes {
			if iou(boxes[0], box) < nmsThreshold {
				tmp = append(tmp, box)
			}
		}
		boxes = tmp
	}
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("geting boxes took: %v ms\n", elapsedTime.Milliseconds())
	return result
}

func iou(box1, box2 []interface{}) float64 {
	return intersection(box1, box2) / union(box1, box2)
}

func union(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	box1_area := (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
	box2_area := (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

	return box1_area + box2_area - intersection(box1, box2)
}

func intersection(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	x1 := math.Max(box1_x1, box2_x1)
	y1 := math.Max(box1_y1, box2_y1)
	x2 := math.Min(box1_x2, box2_x2)
	y2 := math.Min(box1_y2, box2_y2)
	return (x2 - x1) * (y2 - y1)
}
