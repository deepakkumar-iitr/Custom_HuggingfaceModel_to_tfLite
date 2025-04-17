# Custom_HuggingfaceModel_to_tfLite
convert_to_tflite/
│
├── model/                    # Your model folder
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt / merges.txt
│   ├── tokenizer.json
│   └── model.safetensors     # Your trained weights
│
├── convert_to_onnx.py        # Step 1: Load & Export to ONNX
├── convert_onnx_to_tf.py     # Step 2: Convert ONNX → SavedModel
├── convert_tf_to_tflite.py   # Step 3: Convert SavedModel → TFLite
└── requirements.txt
