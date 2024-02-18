package main
/*
we use package onnxruntime at /usr/local/go/src
https://github.com/ivansuteja96/go-onnxruntime
推理的时候慢一点，难道是因为input和output都没有写死的原因吗？
*/
import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"github.com/nfnt/resize"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"onnxruntime"
	"os"
	"sort"
	"strings"
	"time"
)

var (
	session *onnxruntime.ORTSession
)

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
	err = initSession()
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

func test() {
	file, err := os.Open(imagePath)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	imageData, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}
	imageBuffer := bytes.NewBuffer(imageData)
	reader := bytes.NewReader(imageBuffer.Bytes())
	startTime := time.Now()
	boxes, _ := detect_objects_on_image(reader)
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("total elapsedTime: %v ms\n", elapsedTime.Milliseconds())
	buf, _ := json.Marshal(&boxes)
	log.Printf("result: %s\n", buf)
}

func detect_objects_on_image(buf io.Reader) ([][]interface{}, error) {
	startTime := time.Now()
	input, _ := prepareInput(buf, 640)
	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("prepare_input took: %v ms\n", elapsedTime.Milliseconds())
	output, err := run_model(input)
	if err != nil {
		return nil, err
	}
	data := process_output(output[0].Value.([]float32), 640, 640)
	return data, nil
}

func run_model(input []float32) ([]onnxruntime.TensorValue, error) {
	shape := []int64{1, 3, 640, 640}
	startTime := time.Now()
	res, err := session.Predict([]onnxruntime.TensorValue{
		{
			Value: input,
			Shape: shape,
		},
	})
	if err != nil {
		// 处理错误
		log.Printf("predict error:", err)
		return nil, err
	}

	endTime := time.Now()
	elapsedTime := endTime.Sub(startTime)
	log.Printf("Run onnx took: %v ms\n", elapsedTime.Milliseconds())

	return res, err
}

func prepareInput(buf io.Reader, size int) ([]float32, float32) {
	// 加载图像并调整大小为指定的大小
	img, _, _ := image.Decode(buf)
	img = resize.Resize(uint(size), uint(size), img, resize.Lanczos3)

	// 提取像素及其颜色，并归一化
	var red, green, blue []float32
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			red = append(red, float32(r/257)/255.0)
			green = append(green, float32(g/257)/255.0)
			blue = append(blue, float32(b/257)/255.0)
		}
	}

	// 将三个颜色数组连接成一个数组
	input := append(red, green...)
	input = append(input, blue...)

	// 计算图像缩放比例
	scale := float32(size) / float32(img.Bounds().Size().Y)

	return input, scale
}

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

func initSession() error {
	var err error
	/*
		ORT_LOGGING_LEVEL_VERBOSE
		Verbose informational messages (least severe).

		ORT_LOGGING_LEVEL_INFO
		Informational messages.

		ORT_LOGGING_LEVEL_WARNING
		Warning messages.

		ORT_LOGGING_LEVEL_ERROR
		Error messages.

		ORT_LOGGING_LEVEL_FATAL
		Fatal error messages (most severe).
	*/
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_ERROR, "deployment")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	session, err = onnxruntime.NewORTSession(ortEnvDet, modelPath, ortDetSO)
	if err != nil {
		log.Println(err)
		return err
	}
	ortEnvDet.Close()
	ortDetSO.Close()
	return nil
}

func main() {
	test()
	session.Close()
	//fmt.Printf("Success do predict, shape : %+v, result : %+v\n", res[0].Shape, res[0].Value)
}
