import urllib2
import json

def rpc_call(url, method, args):
	data = json.dumps({
	    'id': 1,
	    'method': method,
	    'params': [args]
	}).encode()
	req = urllib2.Request(url,
		data,
		{'Content-Type': 'application/json'})
	f = urllib2.urlopen(req)
	response = f.read()
	return json.loads(response)

url = 'http://localhost:1234/rpc'
args = {'Q': 'berserk episode'}
print rpc_call(url, "QPickRPCService.Get", args)
