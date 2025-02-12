import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import cgi
from emotion_recognition import EmotionRecognizer
from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

estimators = get_best_estimators(True)
estimators_str, estimator_dict = get_estimators_name(estimators)
deeprec = EmotionRecognizer(estimator_dict["GradientBoostingClassifier"], emotions=['angry', 'sad', 'neutral', 'disgust', 'ps', 'happy'], features=["mfcc", "chroma", "mel"], verbose=0)

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        if ctype == 'multipart/form-data':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            if 'file' in form:
                file_item = form['file']
                if file_item.filename:
                    file_name = f"data/{int(time.time())}.wav"
                    with open(file_name, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    try:
                        prediction = deeprec.predict(file_name)
                        print(f"Prediction: {prediction}")
                        response = bytes(prediction, 'utf8')
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.send_header('Content-length', len(response))
                        self.end_headers()
                        self.wfile.write(response)
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(bytes(f"Error during prediction: {str(e)}", 'utf8'))
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"No file uploaded.")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Field 'file' not found in form.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Content-type must be multipart/form-data.")
        
def run(server_class=HTTPServer, handler_class=RequestHandler, port=8042):
    # Train the model
    deeprec.train()
    # Get the accuracy
    print(deeprec.test_score())

    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
