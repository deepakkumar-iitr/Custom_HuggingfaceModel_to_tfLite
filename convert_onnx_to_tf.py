import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("custom_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("custom_tf_model")

print("✅ TensorFlow SavedModel exported to ./custom_tf_model")
