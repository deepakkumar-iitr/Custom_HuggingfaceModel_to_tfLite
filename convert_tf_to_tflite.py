import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("custom_tf_model")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Use if model requires complex ops
]
converter._experimental_lower_tensor_list_ops = False  # Compatibility flag
tflite_model = converter.convert()

with open("custom_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as custom_model.tflite")
