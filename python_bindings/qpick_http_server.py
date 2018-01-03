#!/usr/bin/env python


from collections import defaultdict
from flask import Flask, jsonify, request
app = Flask(__name__)

from rust_qpick import Qpick
import time
import os
import pykeyvi

os.environ['RUST_BACKTRACE']='1'
qpick = Qpick('/root/dragan/index/')
i2q = pykeyvi.Dictionary('/run/shm/i2q/test_merge.kv')


@app.route('/get', methods=['GET'])
def get():
    result = {}
    try:
        queries = request.values.get('q').encode('utf-8', 'ignore')
        queries = queries.split(',')
        s = time.time(); res = list(qpick.nget(queries, 500)); s=time.time() -s
        qs = [ (d, i2q.get(str(id)).GetValue()) for (id, d) in res[:15] ]

        result = {'qs' : qs, 't': s}

    except Exception, e:
        result['code'] = 500
        result['error'] = '%s' % e

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
