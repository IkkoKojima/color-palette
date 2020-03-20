import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def ping():
    return 'ok', 200

@app.route('/', methods=['POST'])
def gen_color_palette_from_image():
    try:
        img_base64 = request.get_json()["image64"]
        img_data = base64.b64decode(img_base64)
        img_np = np.fromstring(img_data, np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters = 3)
        clt.fit(image)
        hist = centroid_histogram(clt)
        centroids = clt.cluster_centers_
        zipped = zip(hist, centroids)

        res = []
        for p, c in zipped:
            color = {"rgb":list(c),"percent":p}
            res.append(color)
        return jsonify(res),200
    except:
        raise

def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))