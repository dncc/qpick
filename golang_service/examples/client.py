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
queries = ['berserk episode']
args = {'Q': json.dumps(queries), 'C': 100}
print(rpc_call(url, "QPickRPCService.NGet", args).get('result', []))
