# %%
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import cgi
print("Beginning...")
from emotion_recognition import EmotionRecognizer
from utils import get_best_estimators# initialize instance

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

# predict angry audio sample
# Create a server that saves the file passed in to data/ and returns the prediction

print("Beginning...")
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)

estimators = get_best_estimators(True)
estimators_str, estimator_dict = get_estimators_name(estimators)
deeprec = EmotionRecognizer(estimator_dict["GradientBoostingClassifier"], emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], features = ["mfcc", "chroma", "mel"], verbose=0)

class RequestHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    form = cgi.FieldStorage(
      fp=self.rfile,
      headers=self.headers,
      environ={'REQUEST_METHOD': 'POST'}
    )
    file_item = form['file']
    file_name = f"{int(time.time())}.wav"
    with open(f"data/{file_name}", 'wb') as f:
      f.write(file_item.file.read())
    # self.wfile.write(b"File saved successfully.")
    prediction = deeprec.predict(f"data/{file_name}")
    print(f"Prediction: {prediction}")
    response = bytes(prediction, 'utf8')
    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.send_header('Content-length', len(response))
    self.end_headers()
    self.wfile.write(response)

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8042):
    # %%

    # train the model
    deeprec.train()
    # get the accuracy
    print(deeprec.test_score())

    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
