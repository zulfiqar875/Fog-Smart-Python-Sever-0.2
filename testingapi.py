from flask import Flask, request, jsonify
import sys

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route('/test', methods=['POST'])   
def test():
    req = request.json
    name = req['name']
    return jsonify({"Your name is: ": name})

@app.route('/', methods=['POST'])
def index():
    content = request.json
    age = content['age']
    systolic = content['systolic']
    diastolic = content['diastolic']
    weight = content['weight']
    height = content['height']
    
    return jsonify({ "hibp":systolic, "lobp":diastolic})
    
if __name__ == '__main__':
    app.run()