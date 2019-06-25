from flask import Flask, request, abort,make_response, jsonify, Response
from flask_restful import Resource, Api, reqparse
# from lti import *
# from mean import *
from imputasi import *

app = Flask(__name__)
# app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')
parser.add_argument('separator')


# Todo
# shows a single todo item and lets you delete a todo item
class Imputasi(Resource):
	def post(self,method):
		args = parser.parse_args()
		separator = args['separator']
		data = args['data']
		data = convert_data(data,separator)
		if method == "lti":
			result = lti(data)
			response={'hasil': result.tolist(), 'status':'berhasil'}
			return response, 200, {'Access-Control-Allow-Origin': '*',
                                   'Access-Control-Allow-Methods': 'POST'}
		elif method == "mean":
			result = mymean(data)
			response = {'hasil': result.tolist(), 'status': 'berhasil'}
			return response, 200, {'Access-Control-Allow-Origin': '*',
                                   'Access-Control-Allow-Methods': 'POST'}
		elif method == "psf":
			result = heal_data(data,missing_patch(data))
			response = {'hasil': result.tolist(), 'status': 'berhasil'}
			return response, 200, {'Access-Control-Allow-Origin': '*',
								   'Access-Control-Allow-Methods': 'POST'}
		elif method == "hotdeck":
			result = hotdeck(data)
			response = {'hasil': result.tolist(), 'status': 'berhasil'}
			return response, 200, {'Access-Control-Allow-Origin': '*',
								   'Access-Control-Allow-Methods': 'POST'}
		else:
			abort(404, message="Metode tidak terdaftar")


##
## Actually setup the Api resource routing here
##
api.add_resource(Imputasi, '/imputasi/<method>')




def convert_data(data,separator):
    data = data.strip()
    if separator == '1':
    	data = data.split("\n")
    elif separator == '2' :
    	data = data.split(";")
    else :
    	abort(404, message="Separator tidak didukung")
    
    for i in range(0, len(data)):
        if(data[i]=="NA"):
            data[i] = 0
        else:
            data[i] = float(data[i])
    data = array(data)
    return data
if __name__ == '__main__':
    app.run(debug=True)

