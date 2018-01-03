import kenlm
model = kenlm.Model("/run/shm/lm_word.bin")
model.perplexity("formatter python files")

from rust_qpick import Qpick
import time
import os
import pykeyvi
os.environ['RUST_BACKTRACE']='full'
qpick = Qpick('/root/dragan/index/')
i2q = pykeyvi.Dictionary('/run/shm/i2q/test_merge.kv')
s = time.time(); res = list(qpick.get('clear tcp connections linux command line', 10)); time.time() -s

qp = Qpick(dir_path="/home/dnc/workspace/cliqz/qpick/index/", start_shard=10, end_shard=15)

qs = [(d, i2q.get(str(id)).GetValue()) for (id, d) in res]

---
supervisorctl start nmslib-10000
...
supervisorctl start nmslib-10009
---
from rust_qpick import Qpick
import time
import os
import pykeyvi
os.environ['RUST_BACKTRACE']='1'
qpick = Qpick('/root/dragan/index/')
i2q = pykeyvi.Dictionary('/run/shm/i2q/test_merge.kv')
s = time.time(); res = list(qpick.nget(
    [
        'how to monitor tcp connections on linux',
        'clear tcp connections linux',
    ]
    , 10)); time.time() -s
---

from cache.resources.simq_rpc_server import Recall
r = Recall('/root/dragan/index/', None)

from cache.resources.simq_rpc_server import SimQueryServer
s = SimQueryServer()
s.get_similar_queries_v2(0, ['how to monitor tcp connections on linux', 'clear tcp connections linux'], 10, 10)

# cluster version
s.get_similar_queries_v2(6, 'how to monitor tcp connections on linux', 10000, 50)
---
from cache.db.queryembeddings.simq_service import SimqService
simq = SimqService().get_client()
simq.get('how to monitor tcp connections on linux')
