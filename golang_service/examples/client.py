import urllib.request
import json

def rpc_call(url, method, args):
	data = json.dumps({
	    'id': 1,
	    'method': method,
	    'params': [args]
	}).encode()
	req = urllib.request.Request(url,
		data,
		{'Content-Type': 'application/json'})
	f = urllib.request.urlopen(req)
	response = f.read().decode('utf-8')
	return json.loads(response)

# start go rpc server
# ./main -port 6007 -index ../index
url = 'http://localhost:6007/rpc'
query = 'changing mac os menu bar'
args = {'Q': query, 'C': 100, 'TFIDF': 1}
print(rpc_call(url, "QPickRPCService.Get", args).get('result', []))
