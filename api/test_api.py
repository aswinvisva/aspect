import requests

url = 'http://127.0.0.1:5000/api/v1/predictor/diagnosis'
files = {'image': open('example_image_proliferate_dr.png', 'rb')}
values = {'image': 'example_image_proliferate_dr.png'}
r = requests.post(url, files=files, data=values)
print(r.json())
