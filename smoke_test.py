import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'src'))

import numpy as np
from PIL import Image
from pathlib import Path

Path('static/uploads').mkdir(parents=True, exist_ok=True)
test_img = 'static/uploads/smoke_test.jpg'
img_array = np.random.randint(100, 200, (400, 800, 3), dtype=np.uint8)
Image.fromarray(img_array).save(test_img)
print('Test image created:', test_img)

import app as flask_app
client = flask_app.app.test_client()
with open(test_img, 'rb') as f:
    response = client.post('/predict', data={'xray': (f, 'smoke_test.jpg'), 'patient_name': 'Test'}, content_type='multipart/form-data')

print('Status code:', response.status_code)
body = response.data.decode()
found = any(cls in body for cls in ['Normal', 'Osteopenia', 'Osteoporosis'])
print('Contains prediction:', found)
if response.status_code == 200 and found:
    print('SMOKE TEST PASSED')
else:
    print('SMOKE TEST FAILED')
    print(body[:500])
