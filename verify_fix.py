import sys, os
sys.path.insert(0, 'src')
import numpy as np, cv2

img = np.random.randint(80, 200, (400, 800, 3), dtype=np.uint8)
cv2.imwrite('test_xray.jpg', img)

from osteo.inference.predict_patches import predict_image_with_probs
r = predict_image_with_probs('test_xray.jpg', 'osteoporosis_mobilenetv2.h5', ['Normal','Osteopenia','Osteoporosis'])
print('PREDICT OK:', r)

from gradcam import generate_gradcam
out = generate_gradcam('osteoporosis_mobilenetv2.h5', 'test_xray.jpg', 'test_gradcam.jpg')
print('GRADCAM OK:', out, 'exists:', os.path.exists(out))

print('ALL TESTS PASSED')
