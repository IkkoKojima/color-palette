import os
import cv2
from sklearn.cluster import KMeans
import numpy as np
import json

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def ping():
    return 'ok', 200

@app.route('/', methods=['POST'])
def gen_color_palette_from_image():
    try:
        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters = 3)
        clt.fit(image)
        hist = centroid_histogram(clt)
        centroids = clt.cluster_centers_
        zipped = zip(hist, centroids)

        res = "{"
        for num,(p, c) in enumerate(zipped):
            tmp ='"color{color_num}":{{"rgb":{color},"percent":{percent}}},'
            res = res + tmp.format(color_num=str(num),color=json.dumps(list(c)),percent=p)
        res = res[:-1] + "}"
        return res,200
    except:
        raise

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))