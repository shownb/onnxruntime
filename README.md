example1 来自github.com/ivansuteja96/go-onnxruntime。推断的时候不需要写input和output的name，推断时候可以只写input，不写output，output自动生成。

example2 来自https://github.com/yalue/onnxruntime_go 推断之前，要写input和output的name，而且最好把input和output的slice都提前生成，使用起来比较不方便。

example1的运行时间比example2是要慢个10来ms
