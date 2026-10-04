#!/usr/bin/env python3
"""Bounded stdlib/source falsifiers; no Torch, images, corpus or native work."""
import ast
import base64
import copy
import csv
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import sys
import tarfile
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().with_name('evaluate_siglip2_compact_ranking.py')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

e=module('_compact_evaluation_tests',PATH)
math_helper=module('_compact_math_tests',PATH.with_name('evaluate_siglip2_genuine_views.py'))


def quality(r1,ap):
    return {'recall_at_1':sum(r1)/len(r1),'map_at_r':sum(ap)/len(ap),'per_query_r1':r1,'per_query_ap':ap}


def descriptor(path,digest='a'*64):
    return {'path':str(path),'sha256':digest}


def unit(number):
    return {'receipt':descriptor('/tmp/u'+str(number)+'/receipt.json'),'log':descriptor('/tmp/u'+str(number)+'.log'),
        'unit':'u'+str(number),'invocation_id':format(number,'032x'),'service_seconds':10,
        'native_peak_rss_kib':100,'both_locks_held':True}


def launch(stage='first',phase='cpu',panel='selection'):
    endpoints=[]
    for i,(seed,arm) in enumerate(e.endpoint_order(stage),1):
        endpoints.append({'seed':seed,'arm':arm,'launch':descriptor('/tmp/l'+str(i)),'terminal':unit(i),
            'checkpoint':descriptor('/tmp/u'+str(i)+'/resume.pt',format(i,'064x')),
            'terminal_state_sha256':'b'*64,'bundle':descriptor('/tmp/u'+str(i)+'/bundle/bundle.json'),
            'inference_state_sha256':'c'*64})
    args=SimpleNamespace(execution_sha256='d'*64,phase=phase,arm=None,seed=None)
    if phase == 'export':args.seed,args.arm=e.endpoint_order(stage)[-1]
    result={'schema':e.AUTHORITY_SCHEMA,'execution_sha256':args.execution_sha256,
        'training':copy.deepcopy(e.TRAINING),
        'nearest_evaluator':e.NEAREST_EVALUATOR,'genuine_evaluator':{'root':'/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4',
            'execution_sha256':e.GENUINE_EXECUTION_SHA,'code':e.GENUINE_PINS},'reference':e.REFERENCE,
        'phase':phase,'arm':args.arm,'seed':args.seed,'stage':stage,'panel':panel,'endpoints':endpoints,
        'selected_cpu':None if phase=='cpu' else unit(10),'exports':{},
        'first_selection':None if stage=='first' else unit(11),
        'selection_go':None if panel=='selection' else unit(12),'both_locks_held':True,
        'selection_previously_exposed':True,'cost_policy':e.COST_POLICY,
        'resource_policies':{p:e.policy(p) for p in ('cpu','export','score')}}
    if phase=='score':result['exports']={e.label(v):unit(i+20) for i,v in enumerate(endpoints)}
    return result,args


def controls(stage):
    return {(s,a):{'service_seconds':100,'total_training_core_seconds':60} for s,a in e.endpoint_order(stage)}


def panels(stage):
    n=e.PANELS['selection'][1]
    left=[0 if i<12 else 1 for i in range(n)]; right=left.copy();right[:6]=[1]*6
    source=quality(left,[.8]*n);concat=quality(left,[.817]*n)
    q={str(s):{'control':quality(left,[.818]*n),'candidate':quality(right,[.822]*n)} for s in e.seeds(stage)}
    return q,source,concat


def intervals(q):
    _,avg=math_helper.averaged_deltas(q,'full','selection')
    return {m:{'mean_delta':sum(avg[m])/len(avg[m]),'product_lower95':.001,'product_upper95':.1,
               'query_lower95':-.001,'query_upper95':.1} for m in e.METRICS}


# Independent definition-time graphs from the supplied installed sources.
FILELOCK_IMPORTS = {
    '__init__':'from ._api import *\nfrom ._error import *\nfrom ._async_read_write import *\nfrom ._read_write import *\nfrom ._soft import *\nfrom ._soft_rw import *\nfrom ._unix import *\nfrom ._windows import *\nfrom .asyncio import *\nfrom .version import *\n',
    '_api':'from ._error import *\nfrom ._util import *\n',
    '_async_read_write':'from ._read_write import *\n',
    '_error':'', '_read_write':'from ._api import *\nfrom ._error import *\n',
    '_soft':'from ._api import *\nfrom ._util import *\n',
    '_soft_rw/__init__':'from ._async import *\nfrom ._sync import *\n',
    '_soft_rw/_async':'from ._sync import *\n',
    '_soft_rw/_sync':'from .._api import *\nfrom .._error import *\nfrom .._soft import *\nfrom .._util import *\n',
    '_unix':'from ._api import *\nfrom ._util import *\n', '_util':'',
    '_windows':'from ._api import *\nfrom ._util import *\n',
    'asyncio':'from ._api import *\nfrom ._error import *\nfrom ._soft import *\nfrom ._unix import *\nfrom ._windows import *\n',
    'version':''}
HUB_IMPORTS = {
    '_buckets':'from .utils import *\n',
    '_commit_api':'from .file_download import *\nfrom .lfs import *\nfrom tqdm.contrib.concurrent import *\n',
    '_dataset_viewer':'from .utils import *\n', '_eval_results':'',
    '_inference_endpoints':'from .errors import *\nfrom .utils import *\n',
    '_jobs_api':'from ._space_api import *\nfrom .utils._datetime import *\n',
    '_local_folder':'from .utils._fixes import *\n',
    '_snapshot_download':'from .file_download import *\nfrom .hf_api import *\nfrom tqdm.contrib.concurrent import *\n',
    '_space_api':'from .utils import *\n',
    '_upload_large_folder':'from ._commit_api import *\nfrom ._local_folder import *\nfrom .utils.sha import *\n',
    'community':'from .utils import *\n',
    'file_download':'from ._local_folder import *\nfrom .utils.sha import *\n',
    'hf_api':'from ._buckets import *\nfrom ._commit_api import *\nfrom ._dataset_viewer import *\nfrom ._eval_results import *\nfrom ._inference_endpoints import *\nfrom ._jobs_api import *\nfrom ._space_api import *\nfrom ._upload_large_folder import *\nfrom .community import *\nfrom .file_download import *\nfrom .repocard_data import *\nfrom .utils._deprecation import *\nfrom .utils._verification import *\nfrom .utils.endpoint_helpers import *\n',
    'lfs':'from .utils.sha import *\n',
    'repocard':'from .file_download import *\nfrom .hf_api import *\nfrom .repocard_data import *\n',
    'repocard_data':'from .utils import *\n', 'utils/_deprecation':'',
    'utils/_verification':'from ..file_download import *\nfrom .sha import *\n',
    'utils/endpoint_helpers':'from ..repocard_data import *\n',
    'utils/insecure_hashlib':'', 'utils/sha':'from .insecure_hashlib import *\n'}


CHARSET_IMPORTS = {
    '__init__':'from .api import value\nfrom .legacy import value\nfrom .models import value\nfrom .utils import value\nfrom .version import value\n',
    'api':'from .cd import value\nfrom .constant import value\nfrom .md import value\nfrom .models import value\nfrom .utils import value\n',
    'cd':'from .constant import value\nfrom .md import value\nfrom .models import value\nfrom .utils import value\n',
    'constant':'', 'legacy':'from .api import value\n',
    'md':'from .constant import value\nfrom .utils import value\n',
    'models':'from .constant import value\nfrom .utils import value\n',
    'utils':'from .constant import value\n', 'version':'',
}

SCIENTIFIC_DISTRIBUTIONS = {
    'scipy':('scipy','1.18.0'), 'scikit_learn':('sklearn','1.9.0'),
    'joblib':('joblib','1.5.3'), 'threadpoolctl':('threadpoolctl','3.6.0'),
    'pandas':('pandas','3.0.3'), 'python_dateutil':('dateutil','2.9.0.post0'),
    'six':('six','1.17.0'), 'narwhals':('narwhals','2.22.1'),
    'psutil':('psutil','7.2.2'), 'pyarrow':('pyarrow','24.0.0'), 'rich':('rich','15.0.0'),
}
# Representative eager edges from the installed ASTs; bodies are synthetic.
SCIENTIFIC_IMPORTS = {
    'sklearn/__init__.py':'from . import __check_build\nfrom . import base\n',
    'sklearn/__check_build/__init__.py':'from ._check_build import value\n',
    'sklearn/base.py':'from .utils import fixes, validation\nfrom .utils._repr_html import estimator\n',
    'sklearn/utils/__init__.py':'',
    'sklearn/utils/fixes.py':'import pandas\nimport scipy.sparse.linalg\nimport scipy.sparse.csgraph\n',
    'sklearn/utils/validation.py':'import joblib\nimport narwhals.stable.v2\nimport scipy.sparse\n',
    'sklearn/utils/_repr_html/__init__.py':'',
    'sklearn/utils/_repr_html/estimator.py':
        'from pathlib import Path\ncss = "".join((Path(__file__).parent / name).read_text() for name in ("estimator.css", "params.css", "features.css"))\n',
    'sklearn/metrics/__init__.py':'from . import _ranking\n',
    'sklearn/metrics/_ranking.py':'import scipy.integrate\nimport scipy.stats\nimport sklearn.preprocessing\n',
    'sklearn/preprocessing/__init__.py':'import sklearn.callback\n',
    'sklearn/callback/__init__.py':'from . import _progressbar\n',
    'sklearn/callback/_progressbar.py':'import rich.progress\n',
    'scipy/__init__.py':'from ._lib import _ccallback\n',
    'scipy/_lib/__init__.py':'',
    'scipy/_lib/_ccallback.py':'from ._ccallback_c import value\n',
    'scipy/sparse/__init__.py':'from . import _base\n',
    'scipy/sparse/_base.py':'import scipy.sparse.linalg\n',
    'scipy/sparse/_sputils.py':'',
    # The observed native initializer edge is represented by a synthetic import.
    'scipy/sparse/csgraph/__init__.py':'from . import _laplacian, _validation\n',
    'scipy/sparse/csgraph/_laplacian.py':'import scipy.sparse.linalg\nimport scipy.sparse._sputils\n',
    'scipy/sparse/csgraph/_validation.py':
        'from scipy.sparse._sputils import value\nfrom ._tools import value\n',
    'scipy/sparse/linalg/__init__.py':'from . import _interface\n',
    'scipy/sparse/linalg/_interface.py':'import scipy.linalg\n',
    'scipy/linalg/__init__.py':'from . import _misc\n',
    'scipy/linalg/_misc.py':'from . import blas\n',
    'scipy/linalg/blas.py':'from ._fblas import value\n',
    'scipy/integrate/__init__.py':'from . import _quadrature\n',
    'scipy/integrate/_quadrature.py':'import scipy.stats\n',
    'scipy/stats/__init__.py':'from . import _stats_py\n',
    'scipy/stats/_stats_py.py':'import scipy.optimize\nimport scipy.special\n',
    'scipy/optimize/__init__.py':'from . import _optimize\n',
    'scipy/optimize/_optimize.py':'',
    'scipy/special/__init__.py':'from . import _basic\n',
    'scipy/special/_basic.py':'',
    'joblib/__init__.py':'from . import _parallel_backends, parallel\n',
    'joblib/_parallel_backends.py':'from .externals import loky\n',
    'joblib/parallel.py':'from .externals.loky import process_executor\nimport threadpoolctl\n',
    'joblib/externals/__init__.py':'',
    'joblib/externals/loky/__init__.py':'from . import process_executor\n',
    'joblib/externals/loky/process_executor.py':'from .backend import utils\nimport psutil\n',
    'joblib/externals/loky/backend/__init__.py':'',
    'joblib/externals/loky/backend/utils.py':'import psutil\n',
    'threadpoolctl.py':'',
    'pandas/__init__.py':'from . import compat\nfrom ._libs import tslibs\nimport dateutil\n',
    'pandas/compat/__init__.py':'from . import pyarrow\n',
    'pandas/compat/pyarrow.py':'import pyarrow\nimport pyarrow.compute\n',
    'pandas/_libs/__init__.py':'from . import tslibs\n',
    # Simulate the three original native initializer dependencies without loading a native.
    'pandas/_libs/tslibs/__init__.py':'from .parsing import value\nimport dateutil.parser\nimport dateutil.tz\nimport dateutil.relativedelta\nimport dateutil.easter\nzone = dateutil.tz.tz.gettz("Etc/UTC")\n',
    'dateutil/__init__.py':'from . import _version\n',
    'dateutil/_version.py':'',
    'dateutil/easter.py':'import datetime\n',
    'dateutil/parser/__init__.py':'from . import _parser, isoparser\n',
    'dateutil/parser/_parser.py':'import six\nfrom .. import relativedelta, tz\n',
    'dateutil/parser/isoparser.py':'import six\nfrom .. import tz\n',
    'dateutil/relativedelta.py':'import six\nfrom . import _common\n',
    'dateutil/_common.py':'',
    'dateutil/tz/__init__.py':'from . import tz\n',
    'dateutil/tz/tz.py':
        'import six\nfrom . import _common, _factories\n'
        'def gettz(name):\n    from dateutil.zoneinfo import get_zonefile_instance\n    return get_zonefile_instance()\n',
    'dateutil/tz/_common.py':'import six\n',
    'dateutil/tz/_factories.py':'',
    'dateutil/zoneinfo/__init__.py':
        'from io import BytesIO\nfrom tarfile import TarFile\nfrom pkgutil import get_data\nfrom dateutil.tz import value\n'
        'def get_zonefile_instance():\n'
        '    with TarFile.open(fileobj=BytesIO(get_data(__name__, "dateutil-zoneinfo.tar.gz"))) as archive:\n'
        '        return archive.extractfile("Etc/UTC").read()\n',
    'six.py':'',
    'narwhals/__init__.py':'from . import dataframe, series\n',
    'narwhals/dataframe.py':'from . import dtypes\n',
    'narwhals/series.py':'from . import dtypes\n',
    'narwhals/dtypes.py':'',
    'narwhals/stable/__init__.py':'',
    'narwhals/stable/v2/__init__.py':'import narwhals\nfrom . import dependencies\n',
    'narwhals/stable/v2/dependencies.py':'',
    'psutil/__init__.py':'from . import _pslinux, _common, _ntuples\n',
    'psutil/_pslinux.py':'from ._psutil_linux import value\nfrom . import _psposix\n',
    'psutil/_common.py':'',
    'psutil/_ntuples.py':'',
    'psutil/_psposix.py':'',
    'pyarrow/__init__.py':'from .lib import value\nfrom . import ipc, types\n',
    'pyarrow/ipc.py':'from ._ipc import value\nfrom . import util\n',
    'pyarrow/types.py':'from .lib import value\n',
    'pyarrow/util.py':'',
    'pyarrow/compute.py':'from ._compute import value\n',
    'rich/__init__.py':'',
    'rich/progress.py':'from . import console, live, table\n',
    'rich/console.py':'from . import style, text\n',
    'rich/live.py':'from . import console\n',
    'rich/table.py':'from . import text\n',
    'rich/style.py':'',
    'rich/text.py':'',
}
# Definition-time local edges from the installed sources; bodies are synthetic.
ACCELERATE_IMPORTS = {
    'accelerate/__init__.py':'import accelerate.accelerator\nimport accelerate.big_modeling\nimport accelerate.data_loader\nimport accelerate.inference\nimport accelerate.launchers\nimport accelerate.parallelism_config\nimport accelerate.state\nimport accelerate.utils\n',
    'accelerate/accelerator.py':'import accelerate.big_modeling\nimport accelerate.checkpointing\nimport accelerate.data_loader\nimport accelerate.logging\nimport accelerate.optimizer\nimport accelerate.parallelism_config\nimport accelerate.scheduler\nimport accelerate.state\nimport accelerate.tracking\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.modeling\nimport accelerate.utils.other\n',
    'accelerate/big_modeling.py':'import accelerate.hooks\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.other\n',
    'accelerate/checkpointing.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\n',
    'accelerate/commands/__init__.py':'',
    'accelerate/commands/config/__init__.py':'import accelerate.commands.config.config\nimport accelerate.commands.config.config_args\nimport accelerate.commands.config.default\nimport accelerate.commands.config.update\n',
    'accelerate/commands/config/cluster.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils\nimport accelerate.utils.constants\n',
    'accelerate/commands/config/config.py':'import accelerate.commands.config.cluster\nimport accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.commands.config.sagemaker\nimport accelerate.utils\n',
    'accelerate/commands/config/config_args.py':'import accelerate.utils\nimport accelerate.utils.constants\nimport yaml\n',
    'accelerate/commands/config/config_utils.py':'import accelerate.commands.menu\nimport accelerate.utils.dataclasses\n',
    'accelerate/commands/config/default.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils\n',
    'accelerate/commands/config/sagemaker.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\n',
    'accelerate/commands/config/update.py':'import accelerate.commands.config.config_args\nimport accelerate.commands.config.config_utils\n',
    'accelerate/commands/menu/__init__.py':'import accelerate.commands.menu.selection_menu\n',
    'accelerate/commands/menu/cursor.py':'',
    'accelerate/commands/menu/helpers.py':'',
    'accelerate/commands/menu/input.py':'import accelerate.commands.menu.keymap\n',
    'accelerate/commands/menu/keymap.py':'',
    'accelerate/commands/menu/selection_menu.py':'import accelerate.commands.menu\nimport accelerate.commands.menu.cursor\nimport accelerate.commands.menu.helpers\nimport accelerate.commands.menu.input\nimport accelerate.commands.menu.keymap\nimport accelerate.utils.imports\n',
    'accelerate/data_loader.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\nimport packaging.version\n',
    'accelerate/hooks.py':'import accelerate.state\nimport accelerate.utils\nimport accelerate.utils.imports\nimport accelerate.utils.memory\nimport accelerate.utils.modeling\nimport accelerate.utils.other\n',
    'accelerate/inference.py':'import accelerate.state\nimport accelerate.utils\n',
    'accelerate/launchers.py':'import accelerate.state\nimport accelerate.utils\nimport accelerate.utils.constants\n',
    'accelerate/logging.py':'import accelerate.state\n',
    'accelerate/optimizer.py':'import accelerate.state\nimport accelerate.utils\n',
    'accelerate/parallelism_config.py':'import accelerate.utils.dataclasses\nimport accelerate.utils.versions\n',
    'accelerate/scheduler.py':'import accelerate.state\n',
    'accelerate/state.py':'import accelerate.utils\nimport accelerate.utils.dataclasses\n',
    'accelerate/tracking.py':'import accelerate.logging\nimport accelerate.state\nimport accelerate.utils\nimport packaging.version\nimport yaml\n',
    'accelerate/utils/__init__.py':'import accelerate.parallelism_config\nimport accelerate.utils.ao\nimport accelerate.utils.bnb\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.environment\nimport accelerate.utils.fsdp_utils\nimport accelerate.utils.imports\nimport accelerate.utils.launch\nimport accelerate.utils.megatron_lm\nimport accelerate.utils.memory\nimport accelerate.utils.modeling\nimport accelerate.utils.offload\nimport accelerate.utils.operations\nimport accelerate.utils.other\nimport accelerate.utils.random\nimport accelerate.utils.torch_xla\nimport accelerate.utils.tqdm\nimport accelerate.utils.transformer_engine\nimport accelerate.utils.versions\n',
    'accelerate/utils/ao.py':'import accelerate.utils.imports\n',
    'accelerate/utils/bnb.py':'import accelerate.big_modeling\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.modeling\n',
    'accelerate/utils/constants.py':'',
    'accelerate/utils/dataclasses.py':'import accelerate.utils.constants\nimport accelerate.utils.environment\nimport accelerate.utils.imports\nimport accelerate.utils.versions\n',
    'accelerate/utils/environment.py':'import packaging.version\n',
    'accelerate/utils/fsdp_utils.py':'import accelerate.logging\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.modeling\nimport accelerate.utils.other\nimport accelerate.utils.versions\n',
    'accelerate/utils/imports.py':'import accelerate.utils.environment\nimport accelerate.utils.versions\nimport packaging.version\n',
    'accelerate/utils/launch.py':'import accelerate.commands.config.config_args\nimport accelerate.utils\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.other\nimport accelerate.utils.versions\n',
    'accelerate/utils/megatron_lm.py':'import accelerate.optimizer\nimport accelerate.scheduler\nimport accelerate.utils.imports\nimport accelerate.utils.operations\n',
    'accelerate/utils/memory.py':'import accelerate.utils.imports\n',
    'accelerate/utils/modeling.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.memory\nimport accelerate.utils.offload\nimport accelerate.utils.tqdm\nimport accelerate.utils.versions\n',
    'accelerate/utils/offload.py':'',
    'accelerate/utils/operations.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.versions\n',
    'accelerate/utils/other.py':'import accelerate.commands.config.default\nimport accelerate.logging\nimport accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\nimport accelerate.utils.modeling\nimport accelerate.utils.transformer_engine\nimport accelerate.utils.versions\nimport packaging.version\n',
    'accelerate/utils/random.py':'import accelerate.state\nimport accelerate.utils.constants\nimport accelerate.utils.dataclasses\nimport accelerate.utils.imports\n',
    'accelerate/utils/torch_xla.py':'',
    'accelerate/utils/tqdm.py':'import accelerate.state\nimport accelerate.utils.imports\n',
    'accelerate/utils/transformer_engine.py':'import accelerate.utils.imports\nimport accelerate.utils.operations\n',
    'accelerate/utils/versions.py':'import accelerate.utils.constants\nimport packaging.version\n',
}

SCIENTIFIC_NATIVE_IMPORTS = {
    'sklearn.__check_build._check_build':'sklearn/__check_build/_check_build.cpython-313-aarch64-linux-gnu.so',
    'scipy._lib._ccallback_c':'scipy/_lib/_ccallback_c.cpython-313-aarch64-linux-gnu.so',
    'scipy.linalg._fblas':'scipy/linalg/_fblas.cpython-313-aarch64-linux-gnu.so',
    'scipy.sparse.csgraph._tools':'scipy/sparse/csgraph/_tools.cpython-313-aarch64-linux-gnu.so',
    'pandas._libs.tslibs.parsing':'pandas/_libs/tslibs/parsing.cpython-313-aarch64-linux-gnu.so',
    'psutil._psutil_linux':'psutil/_psutil_linux.abi3.so',
    'pyarrow.lib':'pyarrow/lib.cpython-313-aarch64-linux-gnu.so',
    'pyarrow._ipc':'pyarrow/_ipc.cpython-313-aarch64-linux-gnu.so',
    'pyarrow._compute':'pyarrow/_compute.cpython-313-aarch64-linux-gnu.so',
}


ID2LABEL_CONSTRUCTOR_SOURCE='self.id2label = {int(key): value for key, value in self.id2label.items()}'


class EndpointFactsFixture:
    """Exact finite constructor statement and original typed serializer; stdlib only."""
    def __init__(self):
        # Installed configuration_utils.py:270, original CPU guard be45f9ec...
        class Config:
            def __init__(self,value):
                self.values=copy.deepcopy(value);self.id2label=self.values['id2label']
                exec(compile(ID2LABEL_CONSTRUCTOR_SOURCE,'<original id2label conversion>','exec'),{}, {'self':self})
                self.values['id2label']=self.id2label
            def to_dict(self):return copy.deepcopy(self.values)
        self.frozen={'id2label':{'0':'LABEL_0','1':'LABEL_1'},'label2id':{'LABEL_0':0,'LABEL_1':1},
            'hidden_size':1152,'torch_dtype':'float32','do_sample':False,'architectures':['SiglipVisionModel'],
            'nested':{'tuple':(1,2),'list':[3,4]}}
        config=Config(self.frozen)
        self.vision={'vision':b'frozen vision'};self.buffers={'position_ids':(0,1)}
        self.head={'weight':b'frozen head'};self.processor={'backend':'torchvision','size':[256,256]}
        self.state={'model':SimpleNamespace(config=config,state_dict=lambda:self.vision,
            named_buffers=lambda:self.buffers.items()),'head_object':SimpleNamespace(state_dict=lambda:self.head),
            'processor_object':SimpleNamespace(to_json_string=lambda:json.dumps(self.processor)),
            'A':b'updated A','means':{'source':1.25}}
        path=PATH.with_name('train_siglip2_substrate_adaptation.py');raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'
        fn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='fingerprint')
        def synthetic_import(name,*args):
            assert name=='torch',name
            return SimpleNamespace(Tensor=()) # No real tensors or installed third-party code.
        namespace={'hashlib':hashlib,'__builtins__':dict(vars(sys.modules['builtins']),__import__=synthetic_import)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original typed serializer only>','exec'),namespace)
        self.fingerprint=namespace['fingerprint']
        self.context={'trainer':SimpleNamespace(fingerprint=lambda _,value:self.fingerprint(value)),
            'training_context':{'initial':{'config':self.frozen}}}
    def facts(self):return e.endpoint_facts(self.context,self.state)


class PortableRuntimeFixture:
    """Installed sources and a previously admitted RECORD; no native imports."""
    def __init__(self,root,*,yaml_native=False,scientific=False,accelerate=False):
        # Legacy falsifiers keep their established installed distributions.
        # The scientific regression below uses the entire production inventory.
        self.runtime_sources={d:paths for d,paths in e.RUNTIME_SOURCES.items()
            if scientific or d not in SCIENTIFIC_DISTRIBUTIONS}
        if not accelerate:self.runtime_sources['accelerate']={'accelerate-1.14.0.dist-info/METADATA'}
        self.site=root/'site-packages';self.site.mkdir()
        packages={}
        for name in ('torch','numpy','PIL','transformers','safetensors','torchvision'):
            path=self.site/name;path.mkdir();packages[name]={'root':str(path)}
        self.package=self.site/'packaging';self.package.mkdir()
        self.sources={}
        names=('__init__.py','_elffile.py','_manylinux.py','_musllinux.py','_parser.py','_structures.py',
            '_tokenizer.py','dependency_groups.py','direct_url.py','errors.py','licenses/__init__.py',
            'licenses/_spdx.py','markers.py','metadata.py','pylock.py','requirements.py','specifiers.py',
            'tags.py','utils.py','version.py')
        for name in names:
            path=self.package/name;path.parent.mkdir(exist_ok=True)
            raw={'__init__.py':b"marker = 'source'\n",'version.py':b'from ._structures import value\n',
                 '_structures.py':b'value = 42\n'}.get(name,b'')
            path.write_bytes(raw);self.sources[path]=raw
        self.rows=[['packaging/'+str(path.relative_to(self.package)),
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
            for path,raw in self.sources.items()]
        self.record=self.site/'packaging-26.2.dist-info'/'RECORD';self.record.parent.mkdir()
        self.record.write_text(''.join(','.join(row)+'\n' for row in self.rows))
        self.bundle=root/'bundle';self.bundle.mkdir()
        self.manifest={'environment':{'packages':packages,'files':{}}}
        digest=hashlib.sha256(self.record.read_bytes()).hexdigest()
        self.context={'trainer':SimpleNamespace(admit_bundle=lambda *_:(self.manifest,{})),
            'guards':{str(self.record):digest},'required_guards':{str(self.record):digest}}
        self.endpoint={'bundle':descriptor(self.bundle/'bundle.json')}
        self.regex=self.site/'regex';self.regex.mkdir()
        self.native=self.regex/'_regex.cpython-313-aarch64-linux-gnu.so'
        self.native.write_bytes(b'original exact native file; never executed')
        self.regex_sources={self.regex/'__init__.py':b"from ._main import value\nmarker = 'source'\n",
            self.regex/'_main.py':b'from ._regex_core import value\n',self.regex/'_regex_core.py':b'value = 73\n'}
        for path,raw in self.regex_sources.items():path.write_bytes(raw)
        self.regex_rows=[['regex/'+path.name,
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
            for path,raw in {**self.regex_sources,self.native:self.native.read_bytes()}.items()]
        self.regex_record=self.site/'regex-2026.6.28.dist-info'/'RECORD';self.regex_record.parent.mkdir()
        self.regex_record.write_text(''.join(','.join(row)+'\n' for row in self.regex_rows))
        for path in (self.regex_record,self.native):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
        self.context['training_context']={'legacy':{'selected':{'source_cpu':{'origins':{
            'files':{str(self.native):self.context['required_guards'][str(self.native)]},
            'native_files':[str(self.native)]}}}}}
        self.metadata_versions=dict(v.split('-',1) for v in (
            'accelerate-1.14.0 aiohttp-3.14.1 filelock-3.29.4 hf_xet-1.5.1 httpx-0.28.1 '
            'huggingface_hub-1.16.1 jinja2-3.1.6 numpy-2.5.0 packaging-26.2 pillow-12.2.0 '
            'pydantic-2.13.4 pyyaml-6.0.3 regex-2026.6.28 safetensors-0.8.0 tokenizers-0.22.2 '
            'torch-2.12.1 tqdm-4.68.3').split())
        examples={'typing_extensions':{'typing_extensions.py':b'value = 21\n'},
            'tqdm':{'tqdm/__init__.py':b'from . import _monitor, _tqdm_pandas, cli, gui\nfrom .std import value\nfrom .version import __version__\n',
                'tqdm/_monitor.py':b'value = 1\n','tqdm/_tqdm_pandas.py':b'value = 2\n',
                'tqdm/cli.py':b'from .std import value\nfrom .version import __version__\n',
                'tqdm/gui.py':b'from .std import value\n',
                'tqdm/std.py':b'from ._monitor import value as monitor\nfrom .utils import value as utility\nvalue = monitor + utility\n',
                'tqdm/utils.py':b'value = 3\n',
                'tqdm/version.py':b"from importlib.metadata import version\n__version__ = version('tqdm')\nmarker = 'source'\n",
                'tqdm/auto.py':b"from .autonotebook import value as notebook\nfrom .asyncio import value as asynchronous\nvalue = notebook + asynchronous\nmarker = 'source'\n",
                'tqdm/autonotebook.py':b'from .std import value\n',
                'tqdm/asyncio.py':b'from .std import value\n'},
            'tokenizers':{'tokenizers/__init__.py':b'from .implementations import value\n',
                'tokenizers/implementations/__init__.py':b'value = 84\n'},
            'httpx':{'httpx/__init__.py':b'from ._api import value\n','httpx/_api.py':b'value = 101\n'},
            'httpcore':{'httpcore/__init__.py':b'value = 102\n'},'anyio':{'anyio/__init__.py':b'value = 103\n'},
            'h11':{'h11/__init__.py':b'value = 104\n'},'certifi':{'certifi/__init__.py':b'value = 105\n'},
            'idna':{'idna/__init__.py':b'value = 106\n'},'jinja2':{'jinja2/__init__.py':b'value = 107\n'},
            'markupsafe':{'markupsafe/__init__.py':b'value = 108\n'},
            'huggingface_hub':{'huggingface_hub/__init__.py':b'from .dataclasses import value\n',
                'huggingface_hub/dataclasses.py':b'from .errors import value\n','huggingface_hub/errors.py':b'value = 109\n',
                'huggingface_hub/utils/__init__.py':b"from ._http import value\nmarker = 'source'\n",
                'huggingface_hub/utils/_http.py':b"from ..errors import value\nmarker = 'source'\n",
                'huggingface_hub/utils/_runtime.py':b"import importlib.metadata\n_package_versions = {}\nfor name in ('aiohttp', 'hf_xet', 'Jinja2', 'httpx', 'numpy', 'Pillow', 'pydantic', 'safetensors', 'torch', 'fastai'):\n    try:\n        _package_versions[name] = importlib.metadata.version(name)\n    except importlib.metadata.PackageNotFoundError:\n        _package_versions[name] = 'N/A'\n"}}
        # Synthetic definition-time PyYAML graph, independent of the runtime list.
        yaml_imports={'__init__':'from .error import *\nfrom .tokens import *\nfrom .events import *\nfrom .nodes import *\nfrom .loader import *\nfrom .dumper import *\ntry:\n    from .cyaml import *\n    __with_libyaml__ = True\nexcept ImportError:\n    __with_libyaml__ = False\n',
            'loader':'from .reader import *\nfrom .scanner import *\nfrom .parser import *\nfrom .composer import *\nfrom .constructor import *\nfrom .resolver import *\n',
            'dumper':'from .emitter import *\nfrom .serializer import *\nfrom .representer import *\nfrom .resolver import *\n',
            'cyaml':'from yaml._yaml import CParser, CEmitter\nfrom .constructor import *\nfrom .serializer import *\nfrom .representer import *\nfrom .resolver import *\n',
            'composer':'from .error import *\nfrom .events import *\nfrom .nodes import *\n',
            'constructor':'from .error import *\nfrom .nodes import *\n',
            'emitter':'from .error import *\nfrom .events import *\n',
            'parser':'from .error import *\nfrom .tokens import *\nfrom .events import *\nfrom .scanner import *\n',
            'reader':'from .error import *\n','representer':'from .error import *\nfrom .nodes import *\n',
            'resolver':'from .error import *\nfrom .nodes import *\n',
            'scanner':'from .error import *\nfrom .tokens import *\n',
            'serializer':'from .error import *\nfrom .events import *\nfrom .nodes import *\n',
            'error':'','events':'','nodes':'value = 113\n','tokens':''}
        examples['pyyaml']={'yaml/'+n+'.py':(imports+"marker = 'source'\n").encode()
            for n,imports in yaml_imports.items()}
        examples['filelock']={'filelock/'+n+'.py':(imports+"value = 127\nmarker = 'source'\n").encode()
            for n,imports in FILELOCK_IMPORTS.items()}
        examples['huggingface_hub'].update({'huggingface_hub/'+n+'.py':
            (imports+"value = 131\nmarker = 'source'\n").encode() for n,imports in HUB_IMPORTS.items()})
        examples['huggingface_hub']['huggingface_hub/utils/_fixes.py']=b'from filelock import value\n'
        examples['tqdm'].update({'tqdm/contrib/__init__.py':b"from ..auto import value\nmarker = 'source'\n",
            'tqdm/contrib/concurrent.py':b"from ..auto import value\nmarker = 'source'\n",
            'tqdm/contrib/logging.py':b"import logging\nfrom ..std import value\nmarker = 'source'\n"})
        self.metadata_versions['charset_normalizer']='3.4.7'
        examples['charset_normalizer']={'charset_normalizer/'+n+'.py':
            (imports+"value = 151\nmarker = 'source'\n").encode() for n,imports in CHARSET_IMPORTS.items()}
        if scientific:
            for distribution,(package,version) in SCIENTIFIC_DISTRIBUTIONS.items():
                self.metadata_versions[distribution]=version
                examples[distribution]={n:(imports+"value = 137\nmarker = 'source'\n").encode()
                    for n,imports in SCIENTIFIC_IMPORTS.items() if n.split('/')[0].removesuffix('.py')==package}
            examples['scikit_learn'].update({'sklearn/utils/_repr_html/'+n+'.css':b'/* pinned definition-time style */'
                for n in ('estimator','params','features')})
            stream=io.BytesIO();raw=b'pinned UTC archive member'
            with tarfile.open(fileobj=stream,mode='w:gz') as archive:
                entry=tarfile.TarInfo('Etc/UTC');entry.size=len(raw)
                archive.addfile(entry,io.BytesIO(raw))
            examples['python_dateutil']['dateutil/zoneinfo/dateutil-zoneinfo.tar.gz']=stream.getvalue()
        if accelerate:
            examples['accelerate']={name:(body+"value = 157\nmarker = 'source'\n").encode()
                for name,body in ACCELERATE_IMPORTS.items()}
        for distribution in examples:self.metadata_versions.setdefault(distribution,'1.0')
        self.extra_sources={};self.extra_records={};self.natives=[self.native]
        for distribution in examples:
            sources={self.site/n:b'' for n in self.runtime_sources.get(distribution,())}
            sources.update({self.site/n:raw for n,raw in examples[distribution].items()})
            for path,raw in sources.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            self.extra_sources.update(sources)
            if distribution in ('tokenizers','markupsafe','charset_normalizer') or (distribution=='pyyaml' and yaml_native):
                name={'tokenizers':'tokenizers/tokenizers.abi3.so',
                    'markupsafe':'markupsafe/_speedups.cpython-313-aarch64-linux-gnu.so',
                    'charset_normalizer':'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so',
                    'pyyaml':'yaml/_yaml.cpython-313-aarch64-linux-gnu.so'}[distribution]
                path=self.site/name;path.write_bytes(b'original exact native file; never executed')
                sources[path]=path.read_bytes();self.natives.append(path)
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
                original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original['files'][str(path)]=digest;original['native_files'].append(str(path))
            if scientific:
                package=SCIENTIFIC_DISTRIBUTIONS.get(distribution,('',None))[0]
                for name in SCIENTIFIC_NATIVE_IMPORTS.values():
                    if name.split('/')[0]!=package:continue
                    path=self.site/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'original scientific native bytes; never executed')
                    sources[path]=path.read_bytes();self.natives.append(path)
                    digest=hashlib.sha256(sources[path]).hexdigest()
                    self.context['guards'][str(path)]=self.context['required_guards'][str(path)]=digest
                    original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
                    original['files'][str(path)]=digest;original['native_files'].append(str(path))
            record=self.site/(distribution+'-'+self.metadata_versions.get(distribution,'1.0')+'.dist-info')/'RECORD'
            record.parent.mkdir(exist_ok=True)
            rows=[[str(path.relative_to(self.site)),
                'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))]
                for path,raw in sources.items()]
            record.write_text(''.join(','.join(row)+'\n' for row in rows));self.extra_records[distribution]=record
            digest=hashlib.sha256(record.read_bytes()).hexdigest()
            self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest
        self.metadata={}
        for distribution,version in self.metadata_versions.items():
            path=self.site/(distribution+'-'+version+'.dist-info')/'METADATA'
            path.parent.mkdir(exist_ok=True)
            raw=('Metadata-Version: 2.1\nName: '+distribution+'\nVersion: '+version+'\n').encode()
            path.write_bytes(raw);self.metadata[distribution]=path
            record={'packaging':self.record,'regex':self.regex_record}.get(distribution,
                self.extra_records.get(distribution,path.with_name('RECORD')))
            name=str(path.relative_to(self.site))
            rows=[r for r in csv.reader(record.read_text().splitlines()) if r[0]!=name] if record.exists() else []
            rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='),str(len(raw))])
            top=path.with_name('top_level.txt')
            top_raw=({'pillow':'PIL'}.get(distribution,distribution)+'\n').encode();top.write_bytes(top_raw)
            rows.append([str(top.relative_to(self.site)),
                'sha256='+base64.urlsafe_b64encode(hashlib.sha256(top_raw).digest()).decode().rstrip('='),str(len(top_raw))])
            record.write_text(''.join(','.join(row)+'\n' for row in rows))
            if distribution=='packaging':self.rows=rows
            elif distribution=='regex':self.regex_rows=rows
            if distribution not in ('packaging','regex'):self.extra_records[distribution]=record
            digest=hashlib.sha256(record.read_bytes()).hexdigest()
            self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest

    @contextmanager
    def boundary(self):
        with patch.dict(e.RUNTIME_SOURCES,self.runtime_sources,clear=True),e.bundle_reads_only(self.context,self.endpoint):
            yield

    def add_identity_distribution(self,name,version,declared):
        package=self.site/name.replace('-','_');package.mkdir()
        source=package/'__init__.py';source.write_bytes(b"raise AssertionError('optional module executed')\n")
        metadata=self.site/(name.replace('-','_')+'-'+version+'.dist-info')/'METADATA'
        metadata.parent.mkdir();metadata.write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        paths=[source,metadata];top=metadata.with_name('top_level.txt')
        if declared is not None:top.write_text(declared);paths.append(top)
        record=metadata.with_name('RECORD')
        rows=[[str(p.relative_to(self.site)),
            'sha256='+base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('='),
            str(p.stat().st_size)] for p in paths]
        record.write_text(''.join(','.join(row)+'\n' for row in rows))
        digest=hashlib.sha256(record.read_bytes()).hexdigest()
        self.context['guards'][str(record)]=self.context['required_guards'][str(record)]=digest
        return metadata,top,record,source


class SourceAdmissionFixture:
    """Run the real context-composition block and pinned adapter, without native work.

    The old authority's actual dictionary expression supplies the pre-native
    context. Only external receipts/logs/wires are disconnected stand-ins.
    """
    def __init__(self, root):
        def pinned(name, filename, digest):
            if name not in sys.modules:
                e.load_authenticated(name, PATH.with_name(filename), digest, {})
            value=sys.modules[name]
            e.bound_file({},value.__file__,digest)
            return value
        reference=pinned('_compact_context_reference_tests','evaluate_siglip2_prototype_residual.py',
            e.REFERENCE['code']['evaluate_siglip2_prototype_residual.py'])
        self.baseline=pinned('_compact_context_baseline_tests','evaluate_siglip2_quadratic_readout.py',
            reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'])
        b=self.baseline
        self.partition={'original_cache':descriptor(root/'cache.npy',reference.FIT_SHA),
            'panels':{'selection':{'original_rows':[8,3,5]},'validation':{'original_rows':[9,4]}}}
        self.path=root/'partition.json';self.path.write_text(json.dumps(self.partition))
        part=descriptor(self.path,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.spec={'partition':part,'source_selection':{'inventory':descriptor(root/'inventory.json',b.SOURCE_INVENTORY_SHA)}}
        fit={'class_names':['one','two'],'targets':[0,1]}
        selected={'partition':copy.deepcopy(self.partition),'source':{'standin':'authenticated source'},
            'genuine':{'prior':{'fit':fit}},'source_driver':None,'original':None,'extract':None,
            'source_cpu':{'invocation':{'invocation_id':'0'*32}}}
        self.calls=[]
        def wire_file(guards,path,digest):
            fact=next(v for v in b.SOURCE_INVENTORY['files'].values() if v['path']==str(path))
            e.require(digest==fact['sha256'],'wire descriptor differs')
            guards[str(path)]=digest;self.calls.append(('wire',str(path)))
            return SimpleNamespace(stat=lambda:SimpleNamespace(st_size=fact['bytes']))
        admission=SimpleNamespace(bound_file=wire_file)
        # Evaluate the authenticated authority's REAL context expression: no
        # synthetic top-level partition, initial state or ownership checks.
        original=PATH.with_name('train_siglip2_quadratic_readout.py')
        e.bound_file({},original,reference.ORIGINAL_PINS[original.name])
        node=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        expression=next(n.value for n in node.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='context' for t in n.targets))
        self.legacy=eval(compile(ast.Expression(expression),str(original),'eval'),
            {'args':None,'root':root,'code':{},'launch':{'partition':part},'guards':{},
             'genuine':None,'selected':selected,'admission':admission,'record':{}})
        self.training={'legacy':self.legacy,'fit_context':{'launch':{'partition':part},'guards':{}},'guards':{}}
        self.archived={'cgroup_before':{},'cgroup_after':{},'input_guards':{},'files':{},'output':str(root)}
        inventory=b.SOURCE_INVENTORY
        self.record={'schema':'siglip2-genuine-view-evaluation-v1','phase':'score','pass':True,
            'engineering_admission_pass':True,'source':selected['source'],
            'spec':{'panel':'selection','stage':'first','endpoints':[inventory['baseline_endpoint']],
                'partition':inventory['partition'],'evaluation_reference':{'root':b.REFERENCE_ROOT,
                    'execution_sha256':b.REFERENCE_EXECUTION_SHA}},
            'resource_policy':b.policy('score'),'query_images':1734,'gallery_images':1715,'panel_products':498,
            'peak_cuda_allocated_bytes':0,'files':{n:v['sha256'] for n,v in inventory['files'].items()},
            'output':str(Path(next(iter(inventory['files'].values()))['path']).parent),
            'quality':{'179061':{'control':{'recall_at_1':inventory['control_quality']['recall_at_1'],
                'map_at_r':inventory['control_quality']['map_at_r'],'per_query_r1':[1]*1670+[0]*64,
                'per_query_ap':[inventory['control_quality']['map_at_r']]*1734}}},
            'input_guards':{},'cgroup_before':{},'cgroup_after':{}}
        for k in ('strict_independent_head_reload_exact','train_raw_unit_cpu_packed_exact','rng_flags_preserved',
            'exit_rehash_pass','full_panel_raw_unit_packed_replay_exact','per_query_replay_exact'):self.record[k]=True
        for k in ('official_read','public_latency_measured','global_production_goal_met','cuda_initialized'):self.record[k]=False
        def reader(admitted,record,terminal,seconds,guards):
            e.require(admitted is admission,'original admission object differs')
            self.calls.append(('terminal',seconds));return {}
        # Copy the adapter's namespace; ORIGINAL module globals stay untouched.
        adapter_factory=FunctionType(reference.source_selection_adapter.__code__,
            {**vars(reference),'original_terminal_reader':lambda _:reader})
        def source_adapter(baseline,context):
            adapted=adapter_factory(baseline,context)  # Full-byte/live-code/AST authentication stays real.
            def receipt(value,guards):
                self.calls.append(('source_json',value['path']))
                if value==self.spec['source_selection']['inventory']:return inventory
                e.require(value==inventory['receipt'],'source receipt descriptor differs')
                return self.record
            adapted.__globals__['read_json']=receipt
            return adapted
        self.reference=SimpleNamespace(FIT_SHA=reference.FIT_SHA,original_terminal_reader=lambda _:reader,
            source_selection_adapter=source_adapter)
        self.originals=[(m,dict(vars(m))) for m in (reference,b)]

    def compose(self):
        node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        def assigns(n,name):
            return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
        start=next(i for i,n in enumerate(node.body) if assigns(n,'legacy'))
        stop=next(i for i,n in enumerate(node.body) if assigns(n,'source_record'))+1
        namespace={**vars(e),'t':self.training,'reference':self.reference,'spec':self.spec,'args':None,
            'guards':{},'archived':self.archived,'baseline':self.baseline,
            'helper':SimpleNamespace(zero_events=lambda _:None),'native':SimpleNamespace(CONCAT_TERMINAL=unit(99))}
        exec(compile(ast.Module(body=node.body[start:stop],type_ignores=[]),str(PATH),'exec'),namespace)
        return namespace['s']


class GroupedMdFixture(PortableRuntimeFixture):
    """Recorded metadata, synthetic native bytes/maps; no extension execution."""
    def __init__(self,root):
        super().__init__(root)
        self.md=self.site/'charset_normalizer/md.cpython-313-aarch64-linux-gnu.so'
        self.md.write_bytes(b'UNAUTHORIZED physical wrapper; never read')
        self.cd=self.site/'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so'
        self.shared=self.site/'81d243bd2c585b0f4821__mypyc.cpython-313-aarch64-linux-gnu.so'
        self.shared.write_bytes(b'original grouped initializer stand-in; never executed')
        original=self.context['training_context']['legacy']['selected']['source_cpu']['origins']
        original['packages']=self.manifest['environment']['packages']
        self.context['training_context']['legacy']['warm_record']={'origins':{'native_files':[]}}
        self.context['training_context']['nearest']=SimpleNamespace(NATIVE_MEMBERS=set())
        self.group_digests={str(p.relative_to(self.site)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.cd,self.shared)}
        digest=self.group_digests[self.shared.name]
        original['native_files'].append(str(self.shared));original['files'][str(self.shared)]=digest
        self.context['guards'][str(self.shared)]=self.context['required_guards'][str(self.shared)]=digest
        self.record=self.extra_records['charset_normalizer']
        self.rows=list(csv.reader(self.record.read_text().splitlines()))
        self.rows.extend([[self.shared.name,'sha256='+base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('='),
            str(self.shared.stat().st_size)],['charset_normalizer/'+self.md.name,
            'sha256=EYzysjLHjG1gl9fj8mFGIRrs0HPeSgwZw5AWcSxih7A','201304']])
        self.write_record()
        source=PATH.with_name('qualify_siglip2_substrate_cpu.py')
        self.context['training_context']['legacy']['source_driver']=SimpleNamespace(
            __file__=str(source),__spec__=SimpleNamespace(origin=str(source)))
        self.context['guards'][str(source)]=self.context['required_guards'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        self.maps=[self.cd,self.shared];self.map_reads=0;self.md_reads=0
    def write_record(self):
        self.record.write_text(''.join(','.join(row)+'\n' for row in self.rows))
        digest=hashlib.sha256(self.record.read_bytes()).hexdigest()
        self.context['guards'][str(self.record)]=self.context['required_guards'][str(self.record)]=digest
    def module(self,path):return SimpleNamespace(__file__=str(path),__spec__=SimpleNamespace(origin=str(path)))
    @contextmanager
    def boundary(self,*,compiled=True,modules=None):
        recorded={'charset_normalizer.md':self.module(self.md if compiled else self.md.with_name('md.py'))}
        if modules is not None:recorded=modules
        original_text,original_bytes=Path.read_text,Path.read_bytes
        def text(path,*args,**kwargs):
            if str(path)=='/proc/self/maps':
                self.map_reads+=1;sys.audit('open',str(path),'r',os.O_RDONLY)
                return ''.join('1000-2000 r-xp 00000000 00:00 0 '+str(p)+'\n' for p in self.maps)
            return original_text(path,*args,**kwargs)
        def raw(path,*args,**kwargs):
            if path==self.md:
                self.md_reads+=1;raise AssertionError('unauthorized md wrapper was hashed/read')
            return original_bytes(path,*args,**kwargs)
        with patch.object(e,'GROUPED_MD_NATIVE_SHA256',self.group_digests,create=True),\
                patch.object(Path,'read_text',text),patch.object(Path,'read_bytes',raw),\
                patch.dict(sys.modules,recorded),super().boundary():
            yield


class EvaluationTests(unittest.TestCase):
    def test_exact_source_stage_seed_and_roles(self):
        for stage in ('first','full'):
            for phase in ('cpu','export','score'):
                value,args=launch(stage,phase);e.check_launch(value,args)
                for k,v in (('selection_previously_exposed',False),('both_locks_held',False),
                    ('stage','other'),('reference',{}),('endpoints',value['endpoints'][::-1])):
                    with self.subTest(stage=stage,phase=phase,k=k),self.assertRaises((ValueError,KeyError)):
                        e.check_launch({**value,k:v},args)
        value,args=launch()
        for k,v in (('first_selection',unit(11)),('selected_cpu',unit(10)),('panel','validation'),('seed',179069),
                    ('arm','control'),('exports',{'control-179061':unit(20)})):
            with self.subTest(k=k),self.assertRaises(ValueError):e.check_launch({**value,k:v},args)
        value,args=launch('full','score','validation');e.check_launch(value,args)
        for k in ('selection_go','first_selection','selected_cpu'):
            with self.assertRaises(ValueError):e.check_launch({**value,k:None},args)
        value,args=launch('full');value['training']['code']['foreign.py']='0'*64
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch();value['endpoints'][0]['seed']=True
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch('full');value['first_selection']=value['endpoints'][0]['terminal']
        with self.assertRaises(ValueError):e.check_launch(value,args)

    def test_pre_native_partition_and_pinned_source_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            f=SourceAdmissionFixture(Path(directory))
            self.assertTrue({'partition','partition_check','initial'}.isdisjoint(f.legacy))
            s=f.compose()
            self.assertEqual(s['partition']['panels']['selection']['original_rows'],[8,3,5])
            self.assertIs(s['fit'],f.legacy['prior']['fit']);self.assertIs(s['selected'],f.legacy)
            self.assertIs(s['source_record'],f.record);self.assertIs(s['origin_records'][-1],f.record)
            self.assertEqual(s['terminals'][-1],f.baseline.SOURCE_SCORE_TERMINAL)
            self.assertEqual(s['source_guards'][str(f.spec['source_selection']['inventory']['path'])],f.baseline.SOURCE_INVENTORY_SHA)
            self.assertEqual(s['guards'][str(f.path)],f.spec['partition']['sha256'])
            self.assertEqual([n for kind,n in f.calls if kind=='terminal'],[500,300])
            self.assertEqual(sum(kind=='wire' for kind,_ in f.calls),3)
            for module,values in f.originals:
                self.assertEqual(vars(module).keys(),values.keys())
                self.assertTrue(all(vars(module)[k] is v for k,v in values.items()))
            self.assertFalse(any(n.split('.')[0] in e.NATIVE for n in sys.modules))

    def test_partition_order_descriptor_cache_and_hash_rejected_before_source(self):
        for mutation in ('order','descriptor','cache','hash'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                f=SourceAdmissionFixture(Path(directory))
                if mutation=='order':f.legacy['selected']['partition']['panels']['selection']['original_rows'].reverse()
                elif mutation=='descriptor':f.training['fit_context']['launch']['partition']=descriptor(f.path,'0'*64)
                elif mutation=='cache':
                    f.partition['original_cache']['sha256']='0'*64
                    f.path.write_text(json.dumps(f.partition))
                    f.spec['partition']['sha256']=hashlib.sha256(f.path.read_bytes()).hexdigest()
                    f.legacy['selected']['partition']=copy.deepcopy(f.partition)
                else:f.path.write_text(json.dumps(f.partition)+' ')
                with self.assertRaisesRegex(ValueError,'ordered panel partition|SHA256'):f.compose()
                self.assertFalse(any(kind=='source_json' for kind,_ in f.calls))

    def test_first_screen_never_bootstraps(self):
        q,source,concat=panels('first');costs=e.paired_cost(controls('first'),'first')
        result=e.decide(math_helper,q,source,concat,'first','selection',{},costs)
        self.assertEqual(result['decision'],'CONTINUE');self.assertFalse(result['selection_go_admits_validation_only'])
        for replacement in (q['179061']['control'],quality(q['179061']['candidate']['per_query_r1'],[.7]*1734)):
            bad=copy.deepcopy(q);bad['179061']['candidate']=replacement
            self.assertEqual(e.decide(math_helper,bad,source,concat,'first','selection',{},costs)['decision'],'KILL')
            with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'first','selection',{'ci':{}},costs)
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'first','selection',{'ci':{}},costs)
        records=controls('first');records[179061,'candidate']['service_seconds']=151
        self.assertEqual(e.decide(math_helper,q,source,concat,'first','selection',{},e.paired_cost(records,'first'))['decision'],'KILL')

    def test_full_equal_seed_floor_sign_and_intervals(self):
        q,source,concat=panels('full');costs=e.paired_cost(controls('full'),'full');bounds=intervals(q)
        result=e.decide(math_helper,q,source,concat,'full','selection',bounds,costs)
        self.assertEqual(result['decision'],'GO');self.assertTrue(result['selection_go_admits_validation_only'])
        self.assertFalse(result['global_production_goal_met']);self.assertFalse(result['product_go'])
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['product_lower95']=0
        self.assertEqual(e.decide(math_helper,q,source,concat,'full','selection',bad,costs)['decision'],'KILL')
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['mean_delta']=.7
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'full','selection',bad,costs)
        bad=copy.deepcopy(q);bad['179069']['candidate']=copy.deepcopy(bad['179069']['control'])
        self.assertEqual(e.decide(math_helper,bad,source,concat,'full','selection',{},costs)['decision'],'KILL')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)
        high=quality([1]*1734,[.9]*1734)
        self.assertEqual(e.decide(math_helper,q,high,concat,'full','selection',{},costs)['decision'],'KILL')
        self.assertEqual(e.decide(math_helper,q,source,high,'full','selection',{},costs)['decision'],'KILL')
        bad=copy.deepcopy(q);bad['179069']['candidate']['per_query_ap'][0]=float('nan')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)

    def test_each_seed_core_and_whole_cost_boundary(self):
        records=controls('full');records[179069,'candidate'].update(service_seconds=150,total_training_core_seconds=90)
        self.assertTrue(all(v['pass'] for v in e.paired_cost(records,'full').values()))
        for key,value in (('total_training_core_seconds',90.01),('service_seconds',150.01)):
            bad=copy.deepcopy(records);bad[179069,'candidate'][key]=value
            self.assertFalse(e.paired_cost(bad,'full')['179069']['pass'])
        for bad_value in (0,True,float('nan'),float('inf'),101):
            bad=copy.deepcopy(records);bad[179061,'control']['total_training_core_seconds']=bad_value
            with self.assertRaises(ValueError):e.paired_cost(bad,'full')
        with self.assertRaises(ValueError):e.paired_cost(controls('first'),'full')

    def test_no_quality_before_readiness(self):
        calls=[]
        for key in e.READINESS:
            ready=dict.fromkeys(e.READINESS,True);ready[key]=False
            with self.assertRaises(ValueError):e.quality_after_readiness(ready,lambda:calls.append('quality'))
        self.assertEqual(calls,[])
        e.quality_after_readiness(dict.fromkeys(e.READINESS,True),lambda:calls.append('quality'))
        self.assertEqual(calls,['quality'])

    def test_diagnostic_and_native_roles(self):
        value=e.diagnostic_result([.1,-.1],[.09,-.2])
        self.assertFalse(value['mechanism_demonstrated']);self.assertFalse(value['utility_veto']);self.assertTrue(value['diagnostic_only'])
        self.assertEqual(e.batch_sizes(1734),[32]*54+[6]);self.assertEqual(e.batch_sizes(1715),[32]*53+[19])
        self.assertEqual(e.batch_sizes(1749)[-1],21);self.assertEqual(e.batch_sizes(1730)[-1],2)
        for invalid in (0,True,-1):
            with self.assertRaises(ValueError):e.batch_sizes(invalid)
        with self.assertRaises(ValueError):e.diagnostic_result([float('nan')],[0])

    def test_strict_files_closure_and_restored_mtime(self):
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):e.strict_json(raw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'worker.py';path.write_bytes(b'first')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();code={'worker.py':digest}
            manifest=root/'execution.json';manifest.write_text(json.dumps(code));manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(e.closure(root,manifest_sha,code,{}),code)
            prior=path.stat();path.write_bytes(b'other');os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            with self.assertRaises(ValueError):e.bound_file({},path,digest)
            path.write_bytes(b'first');link=root/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):e.bound_file({},link,digest)
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):e.bound_file({},fifo,digest)
            with self.assertRaises(ValueError):e.closure(root,manifest_sha,{'worker.py','extra'}, {})

    def test_endpoint_facts_original_constructor_config_conversion_and_fresh_parity(self):
        f=EndpointFactsFixture();live=f.state['model'].config.values
        self.assertEqual(live['id2label'],{0:'LABEL_0',1:'LABEL_1'})
        self.assertTrue(all(type(k)is int for k in live['id2label']))
        original_live=copy.deepcopy(live);original_frozen=copy.deepcopy(f.frozen)
        expected={'vision_sha256':f.fingerprint(f.vision),'members':{
            'config':f.fingerprint(f.frozen),'buffers':f.fingerprint(f.buffers),
            'processor_config':f.fingerprint(f.processor),'head':f.fingerprint(f.head),
            'A':f.fingerprint(f.state['A']),'means':f.fingerprint(f.state['means'])}}
        for _ in range(3):self.assertEqual(f.facts(),expected)
        self.assertEqual(EndpointFactsFixture().facts(),expected)
        self.assertEqual(live,original_live);self.assertEqual(f.frozen,original_frozen)
        # Even an alias-returning stand-in must remain unchanged by comparison.
        f.state['model'].config.to_dict=lambda:live
        self.assertEqual(f.facts(),expected);self.assertEqual(f.facts(),expected)
        self.assertEqual(live,original_live);self.assertEqual(f.frozen,original_frozen)
        f.state['A']=b'different A'
        changed=f.facts();self.assertNotEqual(changed['members']['A'],expected['members']['A'])
        self.assertEqual(changed['members']['config'],expected['members']['config'])

    def test_endpoint_facts_rejects_noncanonical_frozen_and_live_label_maps(self):
        class ForeignInt(int):pass
        class ForeignString(str):pass
        class ForeignDict(dict):pass
        f=EndpointFactsFixture();live=f.state['model'].config.values
        original_live=copy.deepcopy(live);original_frozen=copy.deepcopy(f.frozen)
        cases=(
            ('live_bool',{False:'LABEL_0',1:'LABEL_1'},False),
            ('live_float',{0.0:'LABEL_0',1:'LABEL_1'},False),
            ('live_foreign_int',{ForeignInt(0):'LABEL_0',1:'LABEL_1'},False),
            ('live_strings',{'0':'LABEL_0','1':'LABEL_1'},False),
            ('live_noncanonical',{'00':'LABEL_0',1:'LABEL_1'},False),
            ('live_collision',{0:'LABEL_0','0':'LABEL_0',1:'LABEL_1'},False),
            ('live_missing',{0:'LABEL_0'},False),('live_empty',{},False),
            ('live_changed_key',{0:'LABEL_0',2:'LABEL_1'},False),
            ('live_foreign_key',{object():'LABEL_0',1:'LABEL_1'},False),
            ('live_changed_label',{0:'CHANGED',1:'LABEL_1'},False),
            ('live_non_string_label',{0:None,1:'LABEL_1'},False),
            ('live_foreign_label',{0:ForeignString('LABEL_0'),1:'LABEL_1'},False),
            ('live_map_type',[(0,'LABEL_0'),(1,'LABEL_1')],False),
            ('live_foreign_map',ForeignDict({0:'LABEL_0',1:'LABEL_1'}),False),
            ('frozen_bool',{False:'LABEL_0','1':'LABEL_1'},True),
            ('frozen_int',{0:'LABEL_0','1':'LABEL_1'},True),
            ('frozen_foreign_string',{ForeignString('0'):'LABEL_0','1':'LABEL_1'},True),
            ('frozen_noncanonical',{'00':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_negative',{'-0':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_signed',{'+0':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_collision',{'0':'LABEL_0','00':'LABEL_0','1':'LABEL_1'},True),
            ('frozen_missing',{'0':'LABEL_0'},True),('frozen_empty',{},True),
            ('frozen_changed_key',{'0':'LABEL_0','2':'LABEL_1'},True),
            ('frozen_changed_label',{'0':'CHANGED','1':'LABEL_1'},True),
            ('frozen_foreign_label',{'0':ForeignString('LABEL_0'),'1':'LABEL_1'},True),
            ('frozen_map_type',[('0','LABEL_0'),('1','LABEL_1')],True),
            ('frozen_foreign_map',ForeignDict({'0':'LABEL_0','1':'LABEL_1'}),True))
        for case,labels,frozen in cases:
            with self.subTest(case=case):
                live.clear();live.update(copy.deepcopy(original_live))
                f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
                (f.frozen if frozen else live)['id2label']=labels
                with self.assertRaisesRegex(ValueError,'id2label'):f.facts()
        for frozen in (False,True):
            live.clear();live.update(copy.deepcopy(original_live));f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
            del (f.frozen if frozen else live)['id2label']
            with self.subTest(missing_field_frozen=frozen),self.assertRaisesRegex(ValueError,'id2label'):f.facts()
        live.clear();live.update(copy.deepcopy(original_live));f.frozen.clear();f.frozen.update(copy.deepcopy(original_frozen))
        f.state['model'].config.to_dict=lambda:ForeignDict(live)
        with self.assertRaisesRegex(ValueError,'id2label'):f.facts()

    def test_endpoint_facts_retains_all_other_complete_typed_config_changes(self):
        f=EndpointFactsFixture();live=f.state['model'].config.values
        saved=copy.deepcopy(live);expected=f.fingerprint(f.frozen)
        changes=(('hidden_size',1153),('hidden_size',1152.0),('torch_dtype',None),('do_sample',0),
            ('architectures',('SiglipVisionModel',)),('nested',{'tuple':[1,2],'list':[3,4]}),
            ('label2id',{'LABEL_0':False,'LABEL_1':1}),('foreign_field','extra'),('nested',None))
        for key,value in changes:
            with self.subTest(key=key,value=value):
                live.clear();live.update(copy.deepcopy(saved));live[key]=value
                self.assertNotEqual(f.facts()['members']['config'],expected)
        live.clear();live.update(copy.deepcopy(saved));del live['nested']
        self.assertNotEqual(f.facts()['members']['config'],expected)
        live.clear();live.update(copy.deepcopy(saved));self.assertEqual(f.facts()['members']['config'],expected)

    def test_bundle_boundary_grouped_md_metadata_fresh_cached_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory))
            for _ in range(2):
                with f.boundary():
                    for path in (f.md,f.site/'charset_normalizer/foreign.so',f.site/'charset_normalizer/resume.pt'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.open(path,os.O_RDONLY)
                    with self.assertRaisesRegex(ValueError,'attempted write'):os.open(f.md,os.O_WRONLY)
            self.assertEqual(f.map_reads,2);self.assertEqual(f.md_reads,0)
            self.assertNotIn(str(f.md),f.context['guards']);self.assertNotIn(str(f.md),f.context['required_guards'])
            runtime=next(iter(f.context['portable_audits'].values()))[1]
            self.assertNotIn(f.md,runtime)
            with f.boundary(compiled=False):self.assertTrue(f.md.with_name('md.py').read_bytes())
            self.assertEqual(f.map_reads,2)

    def test_bundle_boundary_grouped_md_map_denials_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory));foreign=f.site/'foreign.so';foreign.write_bytes(b'foreign')
            for cached in (False,True):
                if cached:
                    with f.boundary():pass
                for maps in ([],[f.cd],[f.shared],[f.cd,f.shared,f.md],[f.cd,f.shared,foreign],
                        [f.cd,f.shared,f.site/'missing.so']):
                    with self.subTest(cached=cached,maps=maps):
                        f.maps=maps
                        with patch.object(e,'bound_file',wraps=e.bound_file) as hashes:
                            with self.assertRaises(ValueError):
                                with f.boundary():pass
                            self.assertEqual(hashes.call_count,0,'map denial must precede file hashing')
                f.maps=[f.cd,f.shared]
            self.assertEqual(f.md_reads,0)

    def test_bundle_boundary_grouped_md_identity_record_and_origin_denials(self):
        with tempfile.TemporaryDirectory() as directory:
            f=GroupedMdFixture(Path(directory));original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
            guards=copy.deepcopy(f.context['guards']);required=copy.deepcopy(f.context['required_guards'])
            proof=copy.deepcopy(original);rows=copy.deepcopy(f.rows);digests=dict(f.group_digests)
            for cached in (False,True):
                if cached:
                    with f.boundary():pass
                for case in ('group_sha','group_guard','group_native','group_file','record_guard','md_row_hash',
                        'md_row_size','md_row_missing','md_row_duplicate','md_row_path','md_byte_guard','source_guard',
                        'mixed_file','mixed_spec','foreign_file','foreign_site','relative_origin','missing_spec',
                        'foreign_module','source_spec_identity','group_mutation','source_mutation'):
                    f.context['guards']=copy.deepcopy(guards);f.context['required_guards']=copy.deepcopy(required)
                    original.clear();original.update(copy.deepcopy(proof));f.rows=copy.deepcopy(rows)
                    f.group_digests.clear();f.group_digests.update(digests);f.write_record()
                    modules={'charset_normalizer.md':f.module(f.md)};restore=None
                    if case=='group_sha':f.group_digests[f.shared.name]='a'*64
                    elif case=='group_guard':f.context['required_guards'][str(f.shared)]='a'*64
                    elif case=='group_native':original['native_files'].remove(str(f.shared))
                    elif case=='group_file':original['files'][str(f.cd)]='a'*64
                    elif case=='record_guard':f.context['required_guards'][str(f.record)]='a'*64
                    elif case.startswith('md_row_'):
                        index=next(i for i,r in enumerate(f.rows) if r[0]=='charset_normalizer/'+f.md.name)
                        if case=='md_row_hash':f.rows[index][1]='sha256='+'A'*43
                        elif case=='md_row_size':f.rows[index][2]='1'
                        elif case=='md_row_missing':f.rows.pop(index)
                        elif case=='md_row_duplicate':f.rows.append(f.rows[index])
                        else:f.rows[index][0]='charset_normalizer/foreign.so'
                        f.write_record()
                    elif case=='md_byte_guard':f.context['required_guards'][str(f.md)]='a'*64
                    elif case=='source_guard':
                        source=f.context['training_context']['legacy']['source_driver'].__file__
                        f.context['required_guards'][source]='a'*64
                    elif case=='mixed_file':modules['charset_normalizer.md'].__file__=str(f.md.with_name('md.py'))
                    elif case=='mixed_spec':modules['charset_normalizer.md'].__spec__.origin=str(f.md.with_name('md.py'))
                    elif case=='foreign_file':modules['charset_normalizer.md']=f.module(f.md.with_name('other.so'))
                    elif case=='foreign_site':modules['charset_normalizer.md']=f.module(Path(directory)/f.md.name)
                    elif case=='relative_origin':modules['charset_normalizer.md']=f.module(Path('charset_normalizer')/f.md.name)
                    elif case=='missing_spec':modules['charset_normalizer.md'].__spec__=None
                    elif case=='foreign_module':modules={'charset_normalizer.api':f.module(f.md)}
                    elif case=='source_spec_identity':
                        source=f.context['training_context']['legacy']['source_driver']
                        restore=(source.__spec__,'origin',source.__spec__.origin);source.__spec__.origin='foreign.py'
                    elif case in ('group_mutation','source_mutation'):
                        path=f.shared if case=='group_mutation' else f.md.with_name('md.py')
                        restore=(path,path.read_bytes());path.write_bytes(b'changed')
                    with self.subTest(cached=cached,case=case),self.assertRaises((ValueError,OSError)):
                        with f.boundary(modules=modules):pass
                    if restore:
                        if len(restore)==3:setattr(restore[0],restore[1],restore[2])
                        else:restore[0].write_bytes(restore[1])
                f.context['guards']=copy.deepcopy(guards);f.context['required_guards']=copy.deepcopy(required)
                original.clear();original.update(copy.deepcopy(proof));f.rows=copy.deepcopy(rows);f.write_record()
                f.group_digests.clear();f.group_digests.update(digests)
            self.assertEqual(f.md_reads,0)

    def test_bundle_boundary_denies_train_warm_optimizer_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture=PortableRuntimeFixture(root)
            owned=fixture.bundle/'vision.pt';owned.write_bytes(b'owned')
            outside=root/'resume.pt';outside.write_bytes(b'forbidden')
            with fixture.boundary():
                self.assertEqual(owned.read_bytes(),b'owned')
                for path in (outside,root/'warm.pt',root/'optimizer.pt',root/'teachers.npy'):
                    with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                with self.assertRaisesRegex(ValueError,'attempted write'):owned.write_bytes(b'changed')
            self.assertEqual(outside.read_bytes(),b'forbidden');self.assertEqual(owned.read_bytes(),b'owned')

    def test_bundle_boundary_self_maps_reads_and_proc_denials_fresh_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=PortableRuntimeFixture(root)
            alias=Path('/proc/self/maps');current=Path('/proc')/str(os.getpid())/'maps'
            self.assertEqual(alias.resolve(),current)
            other=Path('/proc')/str(1 if os.getpid()!=1 else 2)/'maps'
            forbidden=[other,Path('/proc/maps'),*(Path('/proc/self')/name for name in
                ('status','smaps','smaps_rollup','mem','fd','fd/999999'))]
            foreign=f.regex/'foreign.so';foreign.write_bytes(b'foreign native; never executed')
            history=[root/name for name in ('resume.pt','warm.pt','optimizer.pt','teachers.npy','checkpoint.pt')]
            for path in history:path.write_bytes(b'historical dependency')
            self.assertNotIn('portable_audits',f.context)
            for cached in (False,True):
                with self.subTest(cached=cached),f.boundary():
                    for path in (alias,current):
                        self.assertIn(b'\n',path.read_bytes())
                        fd=os.open(path,os.O_RDONLY)
                        try:self.assertTrue(os.read(fd,64))
                        finally:os.close(fd)
                        with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'forbidden')
                        for mode in ('w','a','x','r+'):
                            with self.assertRaisesRegex(ValueError,'attempted write'):
                                sys.audit('open',str(path),mode,os.O_RDONLY)
                        for flags in (os.O_WRONLY,os.O_RDWR,os.O_CREAT,os.O_TRUNC,os.O_APPEND):
                            with self.assertRaisesRegex(ValueError,'attempted write'):
                                sys.audit('open',str(path),None,flags)
                    for path in (*forbidden,foreign,*history):
                        with self.subTest(path=str(path)),self.assertRaisesRegex(ValueError,'external dependency'):
                            path.read_bytes()
                    for path in (Path('/proc'),alias.parent,current.parent,alias,current,alias.parent/'fd'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.listdir(path)
                        with self.assertRaisesRegex(ValueError,'external dependency'):os.scandir(path)
                if not cached:audits=f.context['portable_audits'].copy()
                else:self.assertEqual(f.context['portable_audits'],audits)
            self.assertEqual(foreign.read_bytes(),b'foreign native; never executed')
            for path in history:self.assertEqual(path.read_bytes(),b'historical dependency')

    def test_bundle_boundary_imports_record_pinned_source_without_cached_code(self):
        saved={n:m for n,m in sys.modules.items() if n=='packaging' or n.startswith('packaging.')}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));source=f.package/'__init__.py';prior=source.stat()
                source.write_bytes(b"marker = 'cached'\n");py_compile.compile(str(source),doraise=True)
                source.write_bytes(f.sources[source]);os.utime(source,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                extra=f.package/'unqualified.py';extra.write_bytes(b'value = 99\n')
                history=[f.package/name for name in ('resume.pt','warm.pt','optimizer.pt','teachers.npy','checkpoint.pt')]
                for path in history:path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name=='packaging' or name.startswith('packaging.'):sys.modules.pop(name)
                        with f.boundary():
                            if direct_loader:packaging=module('packaging',source)
                            else:packaging=importlib.import_module('packaging')
                            from packaging import version
                            self.assertEqual(packaging.marker,'source','unpinned cached code executed')
                            self.assertEqual(version.value,42)
                            for path in (extra,f.record,f.site/'other.dist-info'/'METADATA',*history):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            with self.assertRaisesRegex(ValueError,'attempted write'):source.write_bytes(b'changed')
                            with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(source))).read_bytes()
                # A stdlib directory containing site-packages grants no extra files.
                del f.context['portable_audits']
                with patch.object(e.sysconfig,'get_path',return_value=str(Path(directory))),f.boundary():
                    with self.assertRaisesRegex(ValueError,'external dependency'):extra.read_bytes()
                self.assertEqual({p:f.context['guards'][str(p)] for p in f.sources},
                    {p:hashlib.sha256(raw).hexdigest() for p,raw in f.sources.items()})
                prior=source.stat();source.write_bytes(b"marker = 'mutate'\n")
                os.utime(source,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name=='packaging' or name.startswith('packaging.'):sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_rejects_missing_mutated_and_foreign_runtime(self):
        for case in ('missing_pin','mutated_record','missing_row','duplicate_row','missing_source','mutated_source',
                     'wrong_size','wrong_hash','foreign_record','source_symlink','foreign_module','wrong_module_role'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root);source=f.package/'version.py'
                if case=='missing_pin':f.context['required_guards'].clear()
                elif case=='mutated_record':f.record.write_bytes(f.record.read_bytes()+b'\n')
                elif case in ('missing_row','duplicate_row','wrong_size','wrong_hash'):
                    rows=copy.deepcopy(f.rows)
                    if case=='missing_row':rows.pop()
                    elif case=='duplicate_row':rows.append(rows[-1])
                    elif case=='wrong_hash':rows[-1][1]='sha256='+'A'*43
                    else:rows[-1][-1]='999'
                    with f.record.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
                    digest=hashlib.sha256(f.record.read_bytes()).hexdigest()
                    f.context['guards'][str(f.record)]=f.context['required_guards'][str(f.record)]=digest
                elif case=='missing_source':source.unlink()
                elif case=='mutated_source':source.write_bytes(b'value = 99\n')
                elif case=='foreign_record':
                    foreign=root/'foreign'/'packaging-26.2.dist-info'/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(f.record.read_bytes());digest=hashlib.sha256(foreign.read_bytes()).hexdigest()
                    f.context['required_guards']={str(foreign):digest};f.context['guards']={str(foreign):digest}
                elif case=='source_symlink':
                    foreign=root/'foreign.py';foreign.write_bytes(f.sources[source]);source.unlink();source.symlink_to(foreign)
                fake=SimpleNamespace(__file__=str(root/'foreign.py'),__spec__=SimpleNamespace(origin=str(root/'foreign.py')))
                if case=='wrong_module_role':
                    fake=SimpleNamespace(__file__=str(f.package/'__init__.py'),
                        __spec__=SimpleNamespace(origin=str(f.package/'__init__.py')))
                with patch.dict(sys.modules,{'packaging.version':fake} if case in ('foreign_module','wrong_module_role') else {}):
                    with self.assertRaises((ValueError,OSError)):
                        with f.boundary():pass

    def test_bundle_boundary_sequential_lazy_sources_and_exact_original_native(self):
        prefixes=('packaging','regex')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                sources={f.package/'__init__.py':f.sources[f.package/'__init__.py'],**f.regex_sources}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached').replace(b'73',b'99'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                foreign=f.regex/'foreign.so';foreign.write_bytes(b'foreign')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            packaging=module('packaging',f.package/'__init__.py') if direct_loader else importlib.import_module('packaging')
                            from packaging import version
                            regex=module('regex',f.regex/'__init__.py') if direct_loader else importlib.import_module('regex')
                            self.assertEqual((packaging.marker,version.value,regex.marker,regex.value),('source',42,'source',73))
                            self.assertEqual(f.native.read_bytes(),b'original exact native file; never executed')
                            for path in (foreign,f.regex_record,f.regex/'unqualified.py',f.regex/'resume.pt',f.site/'foreign.py'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for path in sources:
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                            with self.assertRaisesRegex(ValueError,'attempted write'):f.native.write_bytes(b'changed')
                native=SimpleNamespace(__file__=str(f.native),__spec__=SimpleNamespace(origin=str(f.native)))
                with patch.dict(sys.modules,{'regex._regex':native}),f.boundary():pass
                for path in (*f.regex_sources,f.native):
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(b'x'*len(raw))
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_processor_dependency_sources_have_exact_origins(self):
        expected={'typing_extensions':21,'tokenizers':84,'httpx':101,'httpcore':102,'anyio':103,
            'h11':104,'certifi':105,'idna':106,'jinja2':107,'markupsafe':108,'huggingface_hub':109}
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in expected}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),f.boundary():
                    actual={name:importlib.import_module(name).value for name in expected}
                    self.assertEqual(actual,expected)
                    for name in expected:
                        path=f.site/(name+'.py') if name=='typing_extensions' else f.site/name/'__init__.py'
                        self.assertEqual(sys.modules[name].__file__,str(path))
                        with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                    for name in ('tokenizers','httpx','huggingface_hub'):
                        with self.assertRaisesRegex(ValueError,'external dependency'):(f.site/name/'resume.pt').read_bytes()
                foreign=SimpleNamespace(__file__=str(f.site/'foreign.py'),__spec__=SimpleNamespace(origin=str(f.site/'foreign.py')))
                for name in expected:
                    with self.subTest(name=name),patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
                for name in expected:
                    path=f.site/(name+'.py') if name=='typing_extensions' else f.site/name/'__init__.py'
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(b'x'*len(raw))
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.subTest(name=name),self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in expected:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_hub_runtime_source_contract_is_finite(self):
        names=('__init__.py constants.py dataclasses.py errors.py serialization/__init__.py serialization/_base.py '
            'serialization/_torch.py utils/__init__.py utils/_auth.py utils/_cache_assets.py utils/_cache_manager.py '
            'utils/_chunk_utils.py utils/_datetime.py utils/_detect_agent.py utils/_experimental.py utils/_fixes.py '
            'utils/_git_credential.py utils/_headers.py utils/_hf_uris.py utils/_http.py utils/_lfs.py '
            'utils/_pagination.py utils/_parsing.py utils/_paths.py utils/_runtime.py utils/_safetensors.py '
            'utils/_subprocess.py utils/_telemetry.py utils/_terminal.py utils/_typing.py utils/_validators.py '
            'utils/_xet.py utils/logging.py utils/tqdm.py _buckets.py _commit_api.py _dataset_viewer.py '
            '_eval_results.py _inference_endpoints.py _jobs_api.py _local_folder.py _snapshot_download.py '
            '_space_api.py _upload_large_folder.py community.py file_download.py hf_api.py lfs.py '
            'repocard.py repocard_data.py utils/_deprecation.py utils/_verification.py '
            'utils/endpoint_helpers.py utils/insecure_hashlib.py utils/sha.py').split()
        self.assertEqual(e.RUNTIME_SOURCES['huggingface_hub'],{'huggingface_hub/'+n for n in names}|
            {'huggingface_hub-1.16.1.dist-info/METADATA'})

    def test_hub_runtime_sources_require_original_record_hash_and_size(self):
        for case in ('missing_guard','foreign_guard','mutated_record','missing_row','wrong_hash','wrong_size'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));record=f.extra_records['huggingface_hub']
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                else:
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]=='huggingface_hub/utils/__init__.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaises(ValueError):
                    with f.boundary():pass

    def test_bundle_boundary_hub_utils_cached_source_fallback_and_denials(self):
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='huggingface_hub'}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                sources={p:raw for p,raw in f.extra_sources.items() if p.relative_to(f.site).as_posix() in
                    ('huggingface_hub/utils/__init__.py','huggingface_hub/utils/_http.py')}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                history=[f.site/'huggingface_hub'/name for name in
                    ('utils/resume.pt','utils/optimizer.pt','utils/teachers.npy','utils/unqualified.py','_login.py')]
                foreign=root/'foreign.py';foreign.write_bytes(b'forbidden')
                for path in history:path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0]=='huggingface_hub':sys.modules.pop(name)
                        with f.boundary():
                            path=f.site/'huggingface_hub/utils/__init__.py'
                            utils=module('huggingface_hub.utils',path) if direct_loader else importlib.import_module('huggingface_hub.utils')
                            sibling=sys.modules['huggingface_hub.utils._http']
                            self.assertEqual((utils.value,utils.marker,sibling.marker),(109,'source','source'))
                            for name,value in (('huggingface_hub.utils',utils),('huggingface_hub.utils._http',sibling)):
                                origin=f.site/(name.replace('.','/')+('/__init__.py' if value is utils else '.py'))
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual(value.__file__,str(origin));self.assertEqual(value.__spec__.origin,str(origin))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(origin))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):origin.write_bytes(b'changed')
                            for path in (*history,foreign,f.extra_records['huggingface_hub']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                fake=SimpleNamespace(__file__=str(foreign),__spec__=SimpleNamespace(origin=str(foreign)))
                for name in ('huggingface_hub.utils','huggingface_hub.utils._http'):
                    with patch.dict(sys.modules,{name:fake}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
                for path in (f.site/'huggingface_hub/constants.py',*sources,f.site/'huggingface_hub/serialization/_torch.py'):
                    raw=path.read_bytes();prior=path.stat();path.write_bytes(raw+b'# mutation\n')
                    os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='huggingface_hub':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_bundle_boundary_filelock_hub_transitive_source_fallback_and_denials(self):
        prefixes=('filelock','huggingface_hub','tqdm')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                names=({'filelock/'+n+'.py' for n in FILELOCK_IMPORTS}|
                    {'huggingface_hub/'+n+'.py' for n in HUB_IMPORTS}|
                    {'tqdm/contrib/__init__.py','tqdm/contrib/concurrent.py','tqdm/contrib/logging.py'})
                sources={f.site/n:f.extra_sources[f.site/n] for n in names}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('filelock/resume.pt','filelock/optimizer.pt','filelock/teachers.npy',
                    'filelock/unqualified.py','filelock/foreign.so','huggingface_hub/_login.py',
                    'huggingface_hub/inference/__init__.py','huggingface_hub/utils/_xet_progress_reporting.py',
                    'tqdm/contrib/slack.py','tqdm/notebook.py')]+[root/'foreign.py']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original_before=copy.deepcopy(original);required_before=dict(f.context['required_guards'])
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            lock=module('filelock',f.site/'filelock/__init__.py') if direct_loader else importlib.import_module('filelock')
                            self.assertEqual((lock.value,lock.marker),(127,'source'))
                            for name in ('huggingface_hub._snapshot_download','huggingface_hub.repocard'):
                                self.assertEqual(importlib.import_module(name).value,131)
                            self.assertEqual(sys.modules['tqdm.contrib.concurrent'].value,8)
                            self.assertEqual(importlib.import_module('tqdm.contrib.logging').value,4)
                            for path,raw in sources.items():
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                value=sys.modules[name]
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((value.__file__,value.__spec__.origin,value.marker),(str(path),str(path),'source'))
                                self.assertEqual(path.read_bytes(),raw)
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,f.extra_records['filelock'],f.extra_records['huggingface_hub'],f.extra_records['tqdm']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            with self.assertRaisesRegex(ValueError,'external dependency'):
                                importlib.import_module('filelock.unqualified')
                self.assertEqual(original,original_before);self.assertEqual(f.context['required_guards'],required_before)
                foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                for name in ('filelock','filelock._api','filelock._soft_rw._sync','huggingface_hub.hf_api',
                        'tqdm.contrib.concurrent','tqdm.contrib.logging'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_filelock_hub_closure_requires_original_record_hash_and_size(self):
        for distribution,name in (('filelock','filelock/_soft_rw/_sync.py'),
                ('huggingface_hub','huggingface_hub/hf_api.py'),('tqdm','tqdm/contrib/concurrent.py'),
                ('tqdm','tqdm/contrib/logging.py')):
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                with self.subTest(distribution=distribution,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root);record=f.extra_records[distribution]
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()));row=next(r for r in rows if r[0]==name)
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with self.assertRaises(ValueError):
                        with f.boundary():pass

    def test_filelock_hub_source_mutations_rejected_before_and_after_cached_boundary(self):
        names=({'filelock/'+n+'.py' for n in FILELOCK_IMPORTS}|
            {'huggingface_hub/'+n+'.py' for n in HUB_IMPORTS}|
            {'tqdm/contrib/__init__.py','tqdm/contrib/concurrent.py','tqdm/contrib/logging.py',
                'filelock-3.29.4.dist-info/METADATA'})
        for cached in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                if cached:
                    with f.boundary():pass
                for name in sorted(names):
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(name=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_accelerate_runtime_inventory_is_finite(self):
        self.assertEqual(len(ACCELERATE_IMPORTS),49)
        self.assertEqual(e.RUNTIME_SOURCES['accelerate'],set(ACCELERATE_IMPORTS)|
            {'accelerate-1.14.0.dist-info/METADATA'})
        self.assertFalse(set(ACCELERATE_IMPORTS)&{'accelerate/utils/rich.py',
            'accelerate/utils/deepspeed.py','accelerate/commands/launch.py','accelerate/local_sgd.py',
            'accelerate/memory_utils.py','accelerate/test_utils/__init__.py'})

    def test_accelerate_sources_and_original_records_reject_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory),accelerate=True)
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for name in sorted(ACCELERATE_IMPORTS):
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(source=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            record=f.extra_records['accelerate'];record_raw=record.read_bytes()
            source='accelerate/commands/config/config_args.py'
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record_raw+b'\n')
                elif case=='foreign_record':
                    foreign=f.site.parent/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(record_raw);f.context['required_guards'].pop(str(record))
                    f.context['guards'][str(foreign)]=required[str(record)]
                else:
                    rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==source)
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    with self.subTest(record=case),self.assertRaises(ValueError):
                        with f.boundary():pass
                finally:
                    record.write_bytes(record_raw);f.context['guards']=guards;f.context['required_guards']=required

    def test_scientific_runtime_inventory_is_finite(self):
        counts={'scipy':409,'scikit_learn':109,'joblib':38,'threadpoolctl':1,'pandas':250,
            'python_dateutil':14,'six':1,'narwhals':56,'psutil':5,'pyarrow':9,'rich':54}
        assets={'sklearn/utils/_repr_html/'+n+'.css' for n in ('estimator','params','features')}
        assets.add('dateutil/zoneinfo/dateutil-zoneinfo.tar.gz')
        for distribution,count in counts.items():
            paths=e.RUNTIME_SOURCES[distribution]
            self.assertEqual(sum(n.endswith('.py') for n in paths),count)
            self.assertTrue(all('..' not in Path(n).parts and not Path(n).is_absolute() for n in paths))
            self.assertTrue(all(n.endswith('.py') or n in assets for n in paths))
        selected=set().union(*(e.RUNTIME_SOURCES[d] for d in counts))
        self.assertEqual({n for n in selected if not n.endswith('.py')},assets)
        self.assertTrue(set(SCIENTIFIC_IMPORTS)<=selected)
        self.assertTrue({'scipy/_external/array_api_compat/numpy/fft.py',
            'sklearn/externals/array_api_compat/numpy/fft.py'}<=selected)
        forbidden={'sklearn/cluster/__init__.py','sklearn/ensemble/__init__.py',
            'sklearn/datasets/__init__.py','scipy/datasets/__init__.py','scipy/io/__init__.py',
            'pandas/plotting/_matplotlib/__init__.py','pandas/_version.py','dateutil/rrule.py',
            'dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz',
            'narwhals/_arrow/dataframe.py','narwhals/_pandas_like/dataframe.py',
            'rich/markdown.py','rich/syntax.py','pyarrow/parquet/__init__.py'}
        self.assertFalse(selected & forbidden)
        for distribution in ('markdown_it_py','mdurl','pygments','tzdata'):
            self.assertNotIn(distribution,e.RUNTIME_SOURCES)

    def test_bundle_boundary_scientific_transitive_source_fallback_and_denials(self):
        prefixes=tuple(package for package,_ in SCIENTIFIC_DISTRIBUTIONS.values())+('accelerate','yaml','packaging','tqdm')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root,scientific=True,accelerate=True)
                sources={f.site/n:f.extra_sources[f.site/n] for n in set(SCIENTIFIC_IMPORTS)|set(ACCELERATE_IMPORTS)}
                for path,raw in sources.items():
                    prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('sklearn/resume.pt','scipy/optimizer.pt','pandas/teachers.npy',
                    'sklearn/cluster/__init__.py','scipy/io/__init__.py','scipy/sparse/csgraph/_optional.py',
                    'pandas/plotting/_matplotlib/__init__.py',
                    'dateutil/rrule.py','dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz',
                    'accelerate/resume.pt','accelerate/optimizer.pt','accelerate/teachers.npy',
                    'accelerate/utils/rich.py','accelerate/utils/deepspeed.py','accelerate/commands/launch.py',
                    'accelerate/test_utils/__init__.py','accelerate/foreign.so',
                    'narwhals/_arrow/dataframe.py','rich/markdown.py','pyarrow/parquet/__init__.py',
                    'sklearn/utils/_repr_html/estimator.js','scipy/foreign.so')]+[root/'foreign.py',root/'proc/stat']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                # Extra rows in an original RECORD still confer no optional code/native access.
                for distribution,names in (('scipy',('scipy/foreign.so',)),
                        ('python_dateutil',('dateutil/rrule.py','dateutil/zoneinfo/rebuild.py','dateutil/zoneinfo/foreign.tar.gz')),
                        ('accelerate',('accelerate/utils/rich.py','accelerate/foreign.so'))):
                    record=f.extra_records[distribution]
                    for name in names:
                        raw=(f.site/name).read_bytes()
                        record.write_text(record.read_text()+name+',sha256='+
                            base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                originals=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                original_before=copy.deepcopy(originals);required_before=dict(f.context['required_guards'])
                natives={name:SimpleNamespace(value=137,__file__=str(f.site/path),
                    __spec__=SimpleNamespace(origin=str(f.site/path))) for name,path in SCIENTIFIC_NATIVE_IMPORTS.items()}
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with patch.dict(sys.modules,natives),f.boundary():
                            loaded=module('sklearn',f.site/'sklearn/__init__.py') if direct_loader else importlib.import_module('sklearn')
                            self.assertEqual((loaded.value,loaded.marker),(137,'source'))
                            self.assertEqual(sys.modules['pandas._libs.tslibs'].zone,b'pinned UTC archive member')
                            archive=f.site/'dateutil/zoneinfo/dateutil-zoneinfo.tar.gz'
                            self.assertEqual(archive.read_bytes(),f.extra_sources[archive])
                            with self.assertRaisesRegex(ValueError,'attempted write'):archive.write_bytes(b'changed')
                            importlib.import_module('sklearn.metrics')
                            self.assertEqual(importlib.import_module('accelerate').value,157)
                            for path,raw in sources.items():
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                loaded=sys.modules[name]
                                self.assertIsInstance(loaded.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((loaded.__file__,loaded.__spec__.origin,loaded.marker),(str(path),str(path),'source'))
                                self.assertEqual(path.read_bytes(),raw)
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,*(f.extra_records[d] for d in (*SCIENTIFIC_DISTRIBUTIONS,'accelerate'))):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for name in ('sklearn.cluster','scipy.io','scipy.sparse.csgraph._optional',
                                    'dateutil.rrule','dateutil.zoneinfo.rebuild','accelerate.utils.rich',
                                    'accelerate.commands.launch','rich.markdown'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):importlib.import_module(name)
                # The complete inventory, including modules absent from the small graph,
                # has been authenticated to the fixture's original RECORD bytes.
                for distribution in (*SCIENTIFIC_DISTRIBUTIONS,'accelerate'):
                    for name in e.RUNTIME_SOURCES[distribution]:
                        path=f.site/name
                        self.assertEqual(f.context['guards'][str(path)],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(originals,original_before);self.assertEqual(f.context['required_guards'],required_before)
                foreign=SimpleNamespace(__file__=str(forbidden[-2]),__spec__=SimpleNamespace(origin=str(forbidden[-2])))
                for name in ('sklearn.utils.validation','scipy.stats._stats_py','scipy.sparse.csgraph._validation','pandas.compat.pyarrow',
                        'dateutil.parser._parser','dateutil.easter','dateutil.zoneinfo','narwhals.stable.v2','psutil._pslinux',
                        'rich.console','accelerate.state','accelerate.utils.launch','accelerate.commands.config.config_args'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_scientific_runtime_mutations_and_original_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory),scientific=True)
            paths=[f.site/n for n in ('sklearn/utils/validation.py','scipy/stats/_stats_py.py','scipy/sparse/csgraph/_validation.py',
                'pandas/compat/pyarrow.py','dateutil/parser/_parser.py','narwhals/stable/v2/__init__.py',
                'dateutil/easter.py','dateutil/zoneinfo/__init__.py','dateutil/zoneinfo/dateutil-zoneinfo.tar.gz',
                'joblib/externals/loky/process_executor.py','psutil/_pslinux.py','pyarrow/compute.py',
                'rich/console.py','threadpoolctl.py','six.py','sklearn/utils/_repr_html/estimator.css')]
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for path in paths:
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(path=path.name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            for distribution,source in (('scipy','scipy/sparse/csgraph/_validation.py'),
                    ('python_dateutil','dateutil/easter.py'),
                    ('python_dateutil','dateutil/zoneinfo/__init__.py'),
                    ('python_dateutil','dateutil/zoneinfo/dateutil-zoneinfo.tar.gz')):
                record=f.extra_records[distribution];record_raw=record.read_bytes();record_sha=f.context['required_guards'][str(record)]
                for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
                    guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record_raw+b'\n')
                    elif case=='foreign_record':
                        foreign=f.site.parent/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True,exist_ok=True)
                        foreign.write_bytes(record_raw);f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=record_sha
                    else:
                        rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==source)
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    try:
                        with self.subTest(source=source,record=case),self.assertRaises(ValueError):
                            with f.boundary():pass
                    finally:
                        record.write_bytes(record_raw);f.context['guards']=guards;f.context['required_guards']=required
            record=f.extra_records['scipy'];record_raw=record.read_bytes()
            native=f.site/SCIENTIFIC_NATIVE_IMPORTS['scipy.sparse.csgraph._tools']
            original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
            for case in ('unobserved','original_hash','required_guard','record_hash','record_size'):
                saved=copy.deepcopy(original);guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='unobserved':original['native_files'].remove(str(native))
                elif case=='original_hash':original['files'][str(native)]='a'*64
                elif case=='required_guard':f.context['required_guards'][str(native)]='a'*64
                else:
                    rows=list(csv.reader(record_raw.decode().splitlines()));row=next(r for r in rows if r[0]==str(native.relative_to(f.site)))
                    if case=='record_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    if case=='unobserved':
                        with f.boundary(),self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    else:
                        with self.subTest(native=case),self.assertRaises(ValueError):
                            with f.boundary():pass
                finally:
                    original.clear();original.update(saved);record.write_bytes(record_raw)
                    f.context['guards']=guards;f.context['required_guards']=required;f.context.pop('portable_audits',None)

    def test_charset_source_loader_fallback_preserves_native_wrapper_denial(self):
        selected={'charset_normalizer/'+n+'.py' for n in CHARSET_IMPORTS}
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='charset_normalizer'}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                native=f.site/'charset_normalizer/cd.cpython-313-aarch64-linux-gnu.so'
                mapped=SimpleNamespace(value=151,__file__=str(native),__spec__=SimpleNamespace(origin=str(native)))
                for name in selected:
                    path=f.site/name;raw=f.extra_sources[path];prior=path.stat()
                    path.write_bytes(raw.replace(b'source',b'cached'));py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('charset_normalizer/md.cpython-313-aarch64-linux-gnu.so',
                    'charset_normalizer/__main__.py','charset_normalizer/cli/__init__.py',
                    'charset_normalizer/cli/__main__.py','charset_normalizer/unqualified.py',
                    'charset_normalizer/resume.pt','charset_normalizer/optimizer.pt','charset_normalizer/teachers.npy',
                    'chardet/__init__.py')]+[root/'foreign.py']
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                record=f.extra_records['charset_normalizer'];raw=forbidden[0].read_bytes()
                record.write_text(record.read_text()+str(forbidden[0].relative_to(f.site))+',sha256='+
                    base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                h=hashlib.sha256(record.read_bytes()).hexdigest()
                f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                before=copy.deepcopy(original);required=dict(f.context['required_guards'])
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0]=='charset_normalizer':sys.modules.pop(name)
                        with patch.dict(sys.modules,{'charset_normalizer.cd':mapped}),f.boundary():
                            loaded=module('charset_normalizer',f.site/'charset_normalizer/__init__.py') if direct_loader else importlib.import_module('charset_normalizer')
                            self.assertEqual((loaded.value,loaded.marker),(151,'source'))
                            for path in (f.site/n for n in selected if n!='charset_normalizer/cd.py'):
                                name=str(path.relative_to(f.site)).removesuffix('.py').replace('/','.').removesuffix('.__init__')
                                loaded=sys.modules[name]
                                self.assertIsInstance(loaded.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((loaded.__file__,loaded.__spec__.origin,loaded.marker),(str(path),str(path),'source'))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            self.assertIs(sys.modules['charset_normalizer.cd'],mapped)
                            for path in (*forbidden,record):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                            for name in ('charset_normalizer.cli','charset_normalizer.unqualified','chardet'):
                                with self.assertRaisesRegex(ValueError,'external dependency'):importlib.import_module(name)
                self.assertEqual(e.RUNTIME_SOURCES['charset_normalizer'],selected)
                self.assertEqual(original,before);self.assertEqual(f.context['required_guards'],required)
                self.assertNotIn(str(forbidden[0]),f.context['guards'])
                foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                for name in ('charset_normalizer','charset_normalizer.api','charset_normalizer.md','charset_normalizer.cd'):
                    with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='charset_normalizer':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_charset_original_records_and_current_source_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory))
            for cached in (False,True):
                f.context.pop('portable_audits',None)
                if cached:
                    with f.boundary():pass
                for name in CHARSET_IMPORTS:
                    path=f.site/('charset_normalizer/'+name+'.py');raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(source=name,cached=cached),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            f.context.pop('portable_audits',None)
            record=f.extra_records['charset_normalizer'];raw=record.read_bytes()
            for case in ('missing_guard','mutated_record','missing_row','wrong_hash','wrong_size'):
                guards=dict(f.context['guards']);required=dict(f.context['required_guards'])
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='mutated_record':record.write_bytes(raw+b'\n')
                else:
                    rows=list(csv.reader(raw.decode().splitlines()));row=next(r for r in rows if r[0]=='charset_normalizer/api.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows));h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                try:
                    with self.subTest(record=case),self.assertRaises(ValueError):
                        with f.boundary():pass
                finally:record.write_bytes(raw);f.context['guards']=guards;f.context['required_guards']=required

    def test_numpy_distribution_origin_preserves_missing_metadata(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as directory:
            f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
            required_before=dict(f.context['required_guards'])
            other=f.metadata['packaging'].with_name('direct_url.json');other.write_text('{}')
            unknown=probe.with_name('foreign.json');unknown.write_text('{}')
            for cached in (False,True):
                with self.subTest(cached=cached),patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),f.boundary():
                    distribution=importlib.metadata.distribution('numpy')
                    self.assertIsNone(distribution.read_text('direct_url.json'))
                    self.assertIsNone(distribution.origin)
                    # The exact installed NumPy >=3.13 expression handles None
                    # by its normal AttributeError path, without importing NumPy.
                    try:editable=distribution.origin.dir_info.editable
                    except AttributeError:editable=False
                    self.assertFalse(editable)
                    with self.assertRaises(FileNotFoundError):probe.read_bytes()
                    with self.assertRaisesRegex(ValueError,'attempted write'):probe.write_text('{}')
                    for path in (other,unknown):
                        with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                    with self.assertRaisesRegex(ValueError,'external dependency'):
                        importlib.metadata.distribution('packaging').origin
            self.assertFalse(probe.exists());self.assertNotIn(str(probe),f.context['guards'])
            self.assertEqual(f.context['required_guards'],required_before)

    def test_numpy_missing_origin_rejects_new_files_symlinks_and_record_rows(self):
        import importlib.metadata
        for cached in (False,True):
            for case in ('regular','empty','symlink','dangling'):
                with self.subTest(cached=cached,case=case),tempfile.TemporaryDirectory() as directory:
                    f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                    if cached:
                        with f.boundary():pass
                    target=f.bundle/'original.json';target.write_text('{}')
                    if case=='regular':probe.write_text('{"dir_info":{"editable":true}}')
                    elif case=='empty':probe.write_bytes(b'')
                    else:probe.symlink_to(target if case=='symlink' else target.with_name('missing.json'))
                    with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                        with f.boundary():pass
        for case in ('regular','symlink'):
            with self.subTest(live=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                replacement=f.bundle/'replacement.json';target=f.bundle/'original.json';target.write_text('{}')
                if case=='regular':replacement.write_text('{"dir_info":{"editable":true}}')
                else:replacement.symlink_to(target)
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():
                        # A rename simulates an external concurrent replacement;
                        # the existing write-open denial remains active.
                        replacement.replace(probe)
                        with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                            importlib.metadata.distribution('numpy').origin
        for present in (False,True):
            with self.subTest(declared=present),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));probe=f.metadata['numpy'].with_name('direct_url.json')
                raw=b'{"dir_info":{"editable":true}}'
                if present:probe.write_bytes(raw)
                record=f.extra_records['numpy'];record.write_text(record.read_text()+str(probe.relative_to(f.site))+
                    ',sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')+','+str(len(raw))+'\n')
                h=hashlib.sha256(record.read_bytes()).hexdigest()
                f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():pass

    def test_bundle_boundary_distribution_identity_scan_and_versions(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=PortableRuntimeFixture(root)
            declared=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
            inferred=f.add_identity_distribution('optional-inferred','2.3.4',None)
            empty=f.add_identity_distribution('optional-empty','3.4.5','')
            original=copy.deepcopy(f.context['training_context']['legacy']['selected']['source_cpu']['origins'])
            forbidden=[root/'foreign.py',declared[3].with_name('resume.pt'),declared[3].with_name('foreign.so'),
                declared[0].with_name('entry_points.txt'),declared[0].with_name('PKG-INFO'),
                declared[3].parent/'_vendor/nested-1.0.dist-info/METADATA']
            for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
            # Even a trusted wheel's vendored identity row grants no nested metadata root.
            vendor=forbidden[-1]
            with declared[2].open('a') as stream:
                csv.writer(stream).writerow([str(vendor.relative_to(f.site)),
                    'sha256='+base64.urlsafe_b64encode(hashlib.sha256(vendor.read_bytes()).digest()).decode().rstrip('='),
                    str(vendor.stat().st_size)])
            h=hashlib.sha256(declared[2].read_bytes()).hexdigest()
            f.context['guards'][str(declared[2])]=f.context['required_guards'][str(declared[2])]=h
            required=dict(f.context['required_guards'])
            with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                for cached in (False,True):
                    with self.subTest(cached=cached),f.boundary():
                        mapping=importlib.metadata.packages_distributions()
                        self.assertEqual(mapping['optional_declared'],['optional-declared'])
                        self.assertEqual(mapping['optional_inferred'],['optional-inferred'])
                        self.assertEqual(mapping['optional_empty'],['optional-empty'])
                        for distribution,version in (('optional-declared','1.2.3'),('optional-inferred','2.3.4'),('optional-empty','3.4.5')):
                            self.assertEqual(importlib.metadata.version(distribution),version)
                        self.assertEqual(importlib.metadata.version('torch'),'2.12.1')
                        self.assertEqual(importlib.metadata.version('Pillow'),'12.2.0')
                        for metadata,top,record,source in (declared,inferred,empty):
                            self.assertIn(str(metadata),f.context['guards'])
                            self.assertNotIn(str(source),f.context['guards'])
                            with self.assertRaisesRegex(ValueError,'external dependency'):source.read_bytes()
                            with self.assertRaisesRegex(ValueError,'external dependency'):
                                importlib.import_module(source.parent.name)
                            with self.assertRaisesRegex(ValueError,'attempted write'):metadata.write_bytes(b'changed')
                            with self.assertRaisesRegex(ValueError,'attempted write'):top.write_bytes(b'changed')
                            with self.assertRaisesRegex(ValueError,'attempted write'):record.write_bytes(b'changed')
                        with self.assertRaises(FileNotFoundError):inferred[1].read_bytes()
                        self.assertEqual(inferred[2].read_bytes(),inferred[2].read_text().encode())
                        self.assertEqual(empty[1].read_bytes(),b'')
                        with self.assertRaisesRegex(ValueError,'external dependency'):declared[2].read_bytes()
                        for path in forbidden:
                            with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                # A known absent probe is rechecked on entry, at its read, and on exit.
                inferred[1].write_text('unqualified\n')
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():pass
                inferred[1].unlink()
                with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                    with f.boundary():inferred[1].symlink_to(inferred[3])
                inferred[1].unlink()
                with f.boundary():
                    inferred[1].symlink_to(root/'missing')
                    try:
                        with self.assertRaisesRegex(ValueError,'absent distribution identity changed'):
                            inferred[1].read_bytes()
                    finally:inferred[1].unlink()
                unknown=f.site/'foreign-1.0.dist-info';unknown.mkdir();(unknown/'METADATA').write_text('Name: foreign\nVersion: 1.0\n')
                with f.boundary(),self.assertRaisesRegex(ValueError,'external dependency'):
                    importlib.metadata.packages_distributions()
            self.assertEqual(f.context['required_guards'],required)
            self.assertEqual(f.context['training_context']['legacy']['selected']['source_cpu']['origins'],original)
            for name in ('optional_declared','optional_inferred','optional_empty'):self.assertNotIn(name,sys.modules)

    def test_distribution_identity_requires_original_record_and_exact_rows(self):
        import importlib.metadata
        for identity in ('METADATA','top_level.txt'):
            for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row',
                         'wrong_hash','wrong_size','duplicate_row','foreign_row','symlink'):
                with self.subTest(identity=identity,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    metadata,top,record,_=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
                    target=metadata if identity=='METADATA' else top
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    elif case=='symlink':
                        foreign=root/'foreign.txt';foreign.write_bytes(target.read_bytes());target.unlink();target.symlink_to(foreign)
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()));row=next(r for r in rows if r[0]==str(target.relative_to(f.site)))
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        elif case=='wrong_size':row[2]='999'
                        elif case=='duplicate_row':rows.append(row.copy())
                        else:row[0]='../foreign/'+row[0]
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]),self.assertRaises(ValueError):
                        with f.boundary():importlib.metadata.packages_distributions()

    def test_distribution_identity_current_bytes_rechecked_with_restored_mtime(self):
        for cached in (False,True):
            with self.subTest(cached=cached),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                metadata,top,record,_=f.add_identity_distribution('optional-declared','1.2.3','optional_declared\n')
                inferred=f.add_identity_distribution('optional-inferred','2.3.4',None)
                if cached:
                    with f.boundary():pass
                for path in (metadata,top,record,inferred[0],inferred[2]):
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    try:
                        with self.subTest(path=path),self.assertRaisesRegex(ValueError,'SHA256'):
                            with f.boundary():pass
                    finally:
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_bundle_boundary_yaml_transitive_source_fallback_and_denials(self):
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0]=='yaml'}
        try:
            for name in saved:sys.modules.pop(name)
            for observed_native in (False,True):
                with self.subTest(observed_native=observed_native),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root,yaml_native=observed_native)
                    sources={p:raw for p,raw in f.extra_sources.items() if p.parent.name=='yaml'}
                    for path,raw in sources.items():
                        prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached').replace(b'113',b'999'))
                        py_compile.compile(str(path),doraise=True)
                        path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    forbidden=[f.site/'yaml'/n for n in ('resume.pt','optimizer.pt','teachers.npy','unqualified.py','foreign.so')]
                    forbidden.append(root/'foreign.py')
                    for path in forbidden:path.write_bytes(b'forbidden')
                    native=f.site/'yaml/_yaml.cpython-313-aarch64-linux-gnu.so'
                    original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                    original_before=copy.deepcopy(original);required_before=dict(f.context['required_guards'])
                    # Represent an already mapped original module; never load an extension.
                    mapped=SimpleNamespace(__file__=str(native),__spec__=SimpleNamespace(origin=str(native)),
                        CParser=object(),CEmitter=object())
                    with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                        for direct_loader in (True,False):
                            for name in tuple(sys.modules):
                                if name.split('.')[0]=='yaml':sys.modules.pop(name)
                            with patch.dict(sys.modules,{'yaml._yaml':mapped} if observed_native else {}),f.boundary():
                                yaml=module('yaml',f.site/'yaml/__init__.py') if direct_loader else importlib.import_module('yaml')
                                self.assertEqual((yaml.value,yaml.marker,yaml.__with_libyaml__),(113,'source',observed_native))
                                expected={'yaml'+('' if p.stem=='__init__' else '.'+p.stem) for p in sources}
                                if not observed_native:expected.remove('yaml.cyaml')
                                self.assertEqual({n for n in sys.modules if n.split('.')[0]=='yaml' and n!='yaml._yaml'},expected)
                                for name in expected:
                                    value=sys.modules[name];path=f.site/'yaml'/('__init__.py' if name=='yaml' else name.split('.')[1]+'.py')
                                    self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                    self.assertEqual((value.__file__,value.__spec__.origin,value.marker),(str(path),str(path),'source'))
                                for path,raw in sources.items():
                                    self.assertEqual(path.read_bytes(),raw)
                                    with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                    with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                                for path in (*forbidden,f.extra_records['pyyaml']):
                                    with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                                with self.assertRaisesRegex(ValueError,'external dependency'):
                                    importlib.import_module('yaml.unqualified')
                                if observed_native:
                                    self.assertEqual(native.read_bytes(),b'original exact native file; never executed')
                                    with self.assertRaisesRegex(ValueError,'attempted write'):native.write_bytes(b'changed')
                                else:
                                    with self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    self.assertEqual(original,original_before);self.assertEqual(f.context['required_guards'],required_before)
                    foreign=SimpleNamespace(__file__=str(forbidden[-1]),__spec__=SimpleNamespace(origin=str(forbidden[-1])))
                    for name in ('yaml','yaml.loader','yaml.cyaml','yaml.nodes'):
                        with patch.dict(sys.modules,{name:foreign}),self.assertRaisesRegex(ValueError,'origin differs'):
                            with f.boundary():pass
                    for name in tuple(sys.modules):
                        if name.split('.')[0]=='yaml':sys.modules.pop(name)
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0]=='yaml':sys.modules.pop(name)
            sys.modules.update(saved)

    def test_yaml_runtime_requires_original_record_hash_and_complete_rows(self):
        for case in ('missing_guard','foreign_guard','mutated_record','foreign_record','missing_row','wrong_hash','wrong_size'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root);record=f.extra_records['pyyaml']
                if case=='missing_guard':f.context['required_guards'].pop(str(record))
                elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                elif case=='foreign_record':
                    foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                    foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                    f.context['required_guards'].pop(str(record))
                    f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                else:
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]=='yaml/loader.py')
                    if case=='missing_row':rows.remove(row)
                    elif case=='wrong_hash':row[1]='sha256='+'A'*43
                    else:row[2]='999'
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                with self.assertRaises(ValueError):
                    with f.boundary():pass

    def test_yaml_current_source_and_record_bytes_rechecked_with_restored_mtime(self):
        for cached in (False,True):
            with self.subTest(cached=cached),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory))
                if cached:
                    with f.boundary():pass
                paths=[p for p in f.extra_sources if p.parent.name=='yaml']+[f.extra_records['pyyaml'],f.metadata['pyyaml']]
                for path in paths:
                    raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.subTest(path=path.name),self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))

    def test_yaml_native_authority_stays_exact_and_optional(self):
        for case in ('unobserved','missing_guard','foreign_guard','foreign_origin','not_native_files',
                     'record_hash','record_size','record_missing','module_origin','cached_guard','cached_origin','cached_bytes'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory),yaml_native=case!='unobserved')
                native=f.site/'yaml/_yaml.cpython-313-aarch64-linux-gnu.so';record=f.extra_records['pyyaml']
                original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                if case.startswith('cached_'):
                    with f.boundary():pass
                if case=='unobserved':
                    native.write_bytes(b'unobserved')
                    h=hashlib.sha256(native.read_bytes()).hexdigest()
                    f.context['guards'][str(native)]=f.context['required_guards'][str(native)]=h
                    with record.open('a') as stream:
                        csv.writer(stream).writerow([str(native.relative_to(f.site)),
                            'sha256='+base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('='),'10'])
                elif case=='missing_guard':f.context['required_guards'].pop(str(native))
                elif case in ('foreign_guard','cached_guard'):f.context['required_guards'][str(native)]='a'*64
                elif case in ('foreign_origin','cached_origin'):original['files'][str(native)]='a'*64
                elif case=='not_native_files':original['native_files'].remove(str(native))
                elif case=='cached_bytes':
                    prior=native.stat();native.write_bytes(b'x'*prior.st_size)
                    os.utime(native,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                elif case.startswith('record_'):
                    rows=list(csv.reader(record.read_text().splitlines()))
                    row=next(r for r in rows if r[0]==str(native.relative_to(f.site)))
                    if case=='record_hash':row[1]='sha256='+'A'*43
                    elif case=='record_size':row[2]='1'
                    else:rows.remove(row)
                    record.write_text(''.join(','.join(r)+'\n' for r in rows))
                if case.startswith('record_') or case=='unobserved':
                    h=hashlib.sha256(record.read_bytes()).hexdigest()
                    f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                foreign=SimpleNamespace(__file__=str(native),__spec__=SimpleNamespace(origin=str(f.site/'yaml/foreign.so')))
                with patch.dict(sys.modules,{'yaml._yaml':foreign} if case=='module_origin' else {}):
                    if case in ('unobserved','not_native_files'):
                        with f.boundary():
                            with self.assertRaisesRegex(ValueError,'external dependency'):native.read_bytes()
                    else:
                        with self.assertRaises(ValueError):
                            with f.boundary():pass

    def test_bundle_boundary_tqdm_auto_source_fallback_and_metadata_reads(self):
        import importlib.metadata
        prefixes=('tqdm','huggingface_hub')
        saved={n:m for n,m in sys.modules.items() if n.split('.')[0] in prefixes}
        try:
            for name in saved:sys.modules.pop(name)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);f=PortableRuntimeFixture(root)
                sources={p:raw for p,raw in f.extra_sources.items() if p.parent.name=='tqdm'}
                for path in (f.site/'tqdm/auto.py',f.site/'tqdm/version.py'):
                    raw=sources[path];prior=path.stat();path.write_bytes(raw.replace(b'source',b'cached'))
                    py_compile.compile(str(path),doraise=True)
                    path.write_bytes(raw);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                forbidden=[f.site/n for n in ('tqdm/resume.pt','tqdm/teachers.npy','tqdm/notebook.py',
                    'tqdm/contrib/slack.py','accelerate/__init__.py','aiohttp/__init__.py',
                    'pydantic/__init__.py','hf_xet/unqualified.so','gradio-1.0.dist-info/METADATA',
                    'tqdm-4.68.3.dist-info/entry_points.txt')]
                foreign=root/'foreign.py';forbidden.append(foreign)
                for path in forbidden:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'forbidden')
                with patch.object(sys,'path',[str(f.site),e.sysconfig.get_path('stdlib')]):
                    for direct_loader in (True,False):
                        for name in tuple(sys.modules):
                            if name.split('.')[0] in prefixes:sys.modules.pop(name)
                        with f.boundary():
                            auto=module('tqdm.auto',f.site/'tqdm/auto.py') if direct_loader else importlib.import_module('tqdm.auto')
                            self.assertEqual((auto.value,auto.marker,sys.modules['tqdm.version'].__version__,
                                sys.modules['tqdm.version'].marker),(8,'source','4.68.3','source'))
                            runtime=importlib.import_module('huggingface_hub.utils._runtime')
                            self.assertEqual(runtime._package_versions['fastai'],'N/A')
                            self.assertEqual(runtime._package_versions['Pillow'],'12.2.0')
                            for distribution,version in f.metadata_versions.items():
                                self.assertEqual(importlib.metadata.version(distribution.replace('_','-')),version)
                                self.assertIn('Version: '+version,f.metadata[distribution].read_text())
                                with self.assertRaisesRegex(ValueError,'attempted write'):
                                    f.metadata[distribution].write_bytes(b'changed')
                            for path in sources:
                                name='tqdm'+('' if path.name=='__init__.py' else '.'+path.stem)
                                value=sys.modules[name]
                                self.assertIsInstance(value.__loader__,importlib.machinery.SourceFileLoader)
                                self.assertEqual((value.__file__,value.__spec__.origin),(str(path),str(path)))
                                with self.assertRaises(OSError):Path(importlib.util.cache_from_source(str(path))).read_bytes()
                                with self.assertRaisesRegex(ValueError,'attempted write'):path.write_bytes(b'changed')
                            for path in (*forbidden,f.extra_records['tqdm'],f.extra_records['accelerate']):
                                with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                fake=SimpleNamespace(__file__=str(foreign),__spec__=SimpleNamespace(origin=str(foreign)))
                for name in ('tqdm.auto','tqdm.version','tqdm.std'):
                    with patch.dict(sys.modules,{name:fake}),self.assertRaisesRegex(ValueError,'origin differs'):
                        with f.boundary():pass
        finally:
            for name in tuple(sys.modules):
                if name.split('.')[0] in prefixes:sys.modules.pop(name)
            sys.modules.update(saved)

    def test_tqdm_metadata_rows_require_original_record_hash_and_size(self):
        for distribution in ('tqdm','accelerate','huggingface_hub','packaging'):
            for case in ('missing_guard','foreign_guard','mutated_record','missing_row','wrong_hash','wrong_size','foreign_record'):
                with self.subTest(distribution=distribution,case=case),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    record=f.metadata[distribution].with_name('RECORD')
                    if case=='missing_guard':f.context['required_guards'].pop(str(record))
                    elif case=='foreign_guard':f.context['required_guards'][str(record)]='a'*64
                    elif case=='mutated_record':record.write_bytes(record.read_bytes()+b'\n')
                    elif case=='foreign_record':
                        foreign=root/'foreign'/record.parent.name/'RECORD';foreign.parent.mkdir(parents=True)
                        foreign.write_bytes(record.read_bytes());h=hashlib.sha256(foreign.read_bytes()).hexdigest()
                        f.context['required_guards'].pop(str(record))
                        f.context['guards'][str(foreign)]=f.context['required_guards'][str(foreign)]=h
                    else:
                        rows=list(csv.reader(record.read_text().splitlines()))
                        row=next(r for r in rows if r[0]==str(f.metadata[distribution].relative_to(f.site)))
                        if case=='missing_row':rows.remove(row)
                        elif case=='wrong_hash':row[1]='sha256='+'A'*43
                        else:row[2]='999'
                        record.write_text(''.join(','.join(r)+'\n' for r in rows))
                        h=hashlib.sha256(record.read_bytes()).hexdigest()
                        f.context['guards'][str(record)]=f.context['required_guards'][str(record)]=h
                    with self.assertRaises(ValueError):
                        with f.boundary():pass

    def test_tqdm_metadata_and_source_mutations_are_rejected(self):
        for name in ('tqdm/auto.py','tqdm/version.py','tqdm-4.68.3.dist-info/METADATA',
                     'accelerate-1.14.0.dist-info/METADATA','huggingface_hub-1.16.1.dist-info/METADATA'):
            for cached in (False,True):
                with self.subTest(name=name,cached=cached),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);f=PortableRuntimeFixture(root)
                    if cached:
                        with f.boundary():pass
                    path=f.site/name;raw=path.read_bytes();prior=path.stat()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'SHA256'):
                        with f.boundary():pass

    def test_bundle_boundary_native_grants_require_record_guard_and_original_cpu_origin(self):
        for case in ('unobserved','missing_guard','foreign_guard','foreign_origin','record_hash','record_size',
                     'record_missing','module_origin','cached_guard','cached_origin'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                f=PortableRuntimeFixture(Path(directory));original=f.context['training_context']['legacy']['selected']['source_cpu']['origins']
                if case.startswith('cached_'):
                    with f.boundary():pass
                if case=='unobserved':
                    extra=f.regex/'unobserved.so';extra.write_bytes(b'foreign')
                    h=hashlib.sha256(extra.read_bytes()).hexdigest();f.context['guards'][str(extra)]=h
                    f.context['required_guards'][str(extra)]=h
                    f.regex_rows.append(['regex/unobserved.so','sha256='+base64.urlsafe_b64encode(bytes.fromhex(h)).decode().rstrip('='),'7'])
                elif case=='missing_guard':f.context['required_guards'].pop(str(f.native))
                elif case in ('foreign_guard','cached_guard'):f.context['required_guards'][str(f.native)]='a'*64
                elif case in ('foreign_origin','cached_origin'):original['files'][str(f.native)]='a'*64
                elif case in ('record_hash','record_size','record_missing'):
                    if case=='record_hash':f.regex_rows[-1][1]='sha256='+'A'*43
                    elif case=='record_size':f.regex_rows[-1][2]='1'
                    else:f.regex_rows.pop()
                if case.startswith('record_') or case=='unobserved':
                    f.regex_record.write_text(''.join(','.join(row)+'\n' for row in f.regex_rows))
                    h=hashlib.sha256(f.regex_record.read_bytes()).hexdigest()
                    f.context['guards'][str(f.regex_record)]=f.context['required_guards'][str(f.regex_record)]=h
                foreign=SimpleNamespace(__file__=str(f.native),__spec__=SimpleNamespace(origin=str(f.regex/'foreign.so')))
                with patch.dict(sys.modules,{'regex._regex':foreign} if case=='module_origin' else {}):
                    if case=='unobserved':
                        with f.boundary():
                            for path in f.natives:self.assertEqual(path.read_bytes(),b'original exact native file; never executed')
                            with self.assertRaisesRegex(ValueError,'external dependency'):extra.read_bytes()
                    else:
                        with self.assertRaises(ValueError):
                            with f.boundary():pass

    def test_partial_metadata_receipt_and_foreign_bindings_rejected(self):
        value,args=launch();flags={'threads':1};source={'actual':'source'}
        args.authority=Path('/tmp/launch');args.authority_sha256='a'*64;args.output=Path('/tmp/output')
        context={'args':args,'launch':value,'code':dict.fromkeys(e.FILES,'b'*64),
            'training_context':{'source':source,'legacy':{'selected':{'source_cpu':{'numerical_flags':flags}}}},
            'costs':e.paired_cost(controls('first'),'first'),'reference':SimpleNamespace(check_synthetic_bootstrap=lambda _:None),
            'guards':{}}
        record={'schema':e.SCHEMA,'phase':'cpu','arm':None,'seed':None,'stage':'first','panel':'selection',
            'execution_sha256':args.execution_sha256,'source_code':context['code'],'source':source,
            'cost_policy':e.COST_POLICY,'launch':value,'authority':descriptor(args.authority),
            'authority_sha256':args.authority_sha256,'binding':e.binding(context),'numerical_flags':flags,
            'output':str(args.output),'cost':context['costs'],'resource_policy':e.policy('cpu'),
            'wall_seconds':1,'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':0,'cuda_initialized':False,
            'invocation':{'argv':e.cli(args),'optimize':0,'cuda_visible_devices':'','cublas_workspace_config':':4096:8'},
            'synthetic_bootstrap':{},'files':{},'quality_read':False,'metadata_only':True,
            'updated_payloads_authenticated':True,'malformed_inference_rejected':True,
            'payload_facts':{e.label(v):{} for v in value['endpoints']},
            'calibration':{'same_role_forward_exact':True,'raw_unit_packed_exact':True}}
        for key in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
            'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority'):record[key]=True
        for key in ('official_read','global_production_goal_met','public_latency_measured','product_go'):record[key]=False
        with patch.object(e,'read_json',return_value=value):
            e.check_receipt(context,record,'cpu')
            for key,bad in (('metadata_only',False),('updated_payloads_authenticated',False),('payload_facts',{}),
                ('quality_read',True),('source_code',{}),('stage','full'),('panel','validation'),('numerical_flags',{})):
                with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                    e.check_receipt(context,{**record,key:bad},'cpu')

    def test_whole_unit_resource_caps(self):
        for phase in ('cpu','export','score'):
            record={'resource_policy':e.policy(phase),'wall_seconds':1,'process_peak_rss_kib':100,
                'peak_cuda_allocated_bytes':1 if phase=='export' else 0,'cuda_initialized':phase=='export'}
            e.check_resource_facts(record,phase)
            for key,value in (('wall_seconds',e.policy(phase)['seconds']),('process_peak_rss_kib',8*1024**2+1),
                ('peak_cuda_allocated_bytes',10_000_000_000),('cuda_initialized',phase!='export')):
                with self.assertRaises(ValueError):e.check_resource_facts({**record,key:value},phase)

    def test_owned_native_api_exit_and_exact_membership(self):
        from test_siglip2_nearest_ranking import NativeAdmissionFixture,driver
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=NativeAdmissionFixture(root)
            class ExitContext(dict):
                def __getitem__(self,key):
                    accessed.add(key);return super().__getitem__(key)
            accessed=set()
            fit=f.context['fit_context']=ExitContext(f.context['fit_context'])
            phases=fit['phase_seconds'];fit.pop('unit_started')  # Authority does not supply the run clock.
            api=f.admit()
            def code_descriptor(name,names):
                target=root/name;target.mkdir()
                for n in names:(target/n).write_bytes(PATH.with_name(n).read_bytes())
                code={n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in names}
                manifest=target/'execution.json';manifest.write_text(json.dumps(code))
                return {'root':str(target),'code':code,'execution_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            own=code_descriptor('own',e.FILES);train=code_descriptor('train',e.TRAIN_FILES) if all(PATH.with_name(n).exists() for n in e.TRAIN_FILES) else own
            refs=[code_descriptor('ref'+str(i),pins) for i,pins in enumerate((e.NEAREST_EVALUATOR['code'],e.GENUINE_PINS,e.REFERENCE['code']))]
            # No model exists; use the REAL native API, with only disconnected trainer/source snapshots stubbed.
            trainer=SimpleNamespace(require_no_training=lambda _:None,helper_guard=lambda _:None,
                admit_bundle=lambda *_:({},{}))
            f.context['nearest']=driver
            context={'trainer':trainer,'training_context':f.context,'args':SimpleNamespace(phase='export',execution_sha256=own['execution_sha256']),
                'root':Path(own['root']),'code':own['code'],'guards':{},'launch':{'training':train,
                    'nearest_evaluator':refs[0],'genuine_evaluator':refs[1],'reference':refs[2],'endpoints':[]}}
            runtime_root=root/'runtime';runtime_root.mkdir();portable=PortableRuntimeFixture(runtime_root)
            with portable.boundary():pass
            e.merge_guards(context['guards'],portable.context['guards'])
            def action():
                phases.clear();return e.exit_rehash(context)
            with patch.object(e,'guard_helpers'),patch.object(e,'NEAREST_EVALUATOR',refs[0]), \
                patch.object(e,'GENUINE_PINS',refs[1]['code']),patch.object(e,'TRAIN_FILES',train['code']),redirect_stdout(io.StringIO()):
                # Execute the actual run prefix through native_start, stopping
                # before the Torch import. The fixture used to mask this bug
                # by supplying its own unrelated fitter start timestamp.
                run_node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
                stop=next(i for i,n in enumerate(run_node.body) if isinstance(n,ast.Import) and
                    any(a.name=='torch' for a in n.names))
                prefix=copy.deepcopy(run_node);prefix.body=prefix.body[:stop]
                def start(value):
                    self.assertIs(value,context)
                    self.assertIs(action(),f.legacy['origins'])
                    self.assertIs(fit['unit_started'],e.UNIT_STARTED,'inherited exit must use the original evaluator start')
                    self.assertIs(fit['phase_seconds'],phases)
                def run_prefix(node):
                    namespace={**vars(e),'cli':lambda _:sys.argv,'authority':lambda _:context,'native_start':start}
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(PATH),'exec'),namespace)
                    namespace['run'](context['args'])
                for phase in ('cpu','export','score'):
                    context['args'].phase=phase;fit.pop('unit_started',None)
                    before=e.time.perf_counter()-e.UNIT_STARTED
                    run_prefix(prefix)
                    after=e.time.perf_counter()-e.UNIT_STARTED
                    self.assertEqual(list(phases),['old_exit_begin','old_exit_end','own_union_begin',
                        'own_union_end','closure_begin','closure_end'])
                    self.assertTrue(all(before<=v<=after for v in phases.values()))
                    self.assertEqual(list(phases.values()),sorted(phases.values()))
                required={'legacy','old','root','args','code','guards','readout','phase_seconds','unit_started'}
                self.assertTrue(required<=accessed,'real private exit context fields were not exercised')
                # Each inherited required field is consumed by the authentic
                # private adapter; neither a gate nor a source predicate is stubbed.
                for key in sorted(required):
                    saved=fit.pop(key)
                    try:
                        phases.clear()
                        with self.subTest(missing=key),self.assertRaises(KeyError) as error:action()
                        self.assertEqual(error.exception.args,(key,))
                    finally:fit[key]=saved
                target=ast.parse("context['training_context']['fit_context']['unit_started']=None").body[0].targets[0]
                for replacement in (None,'time.perf_counter()',"context['training_context']['nearest'].UNIT_STARTED"):
                    class ChangeStart(ast.NodeTransformer):
                        count=0
                        def visit_Assign(self,node):
                            if any(ast.dump(t,include_attributes=False)==ast.dump(target,include_attributes=False) for t in node.targets):
                                self.count+=1
                                if replacement is None:return None
                                node.value=ast.parse(replacement,mode='eval').body
                            return self.generic_visit(node)
                    change=ChangeStart();mutant=change.visit(copy.deepcopy(prefix))
                    self.assertEqual(change.count,1)
                    fit.pop('unit_started',None)
                    expected=KeyError if replacement is None else AssertionError
                    with self.subTest(start=replacement),self.assertRaises(expected):run_prefix(mutant)
                    if replacement is None:
                        with self.assertRaisesRegex(KeyError,'unit_started'):action()
                context['args'].phase='export';run_prefix(prefix)
                self.assertIs(action(),f.legacy['origins'])
                f.set_origins({p:h for i,(p,h) in enumerate(f.files.items()) if i})
                with self.assertRaisesRegex(ValueError,'exact four'):action()
                f.set_origins(f.files)
                with self.assertRaisesRegex(ValueError,'owned legacy'):api.audit_origins(dict(f.legacy))
                private=next(c.cell_contents for c in api.audit_origins.__closure__ if isinstance(c.cell_contents,driver.FunctionType)
                    and c.cell_contents.__name__=='audit_origins')
                with patch.dict(private.__globals__,{'_nearest_supplement':{'files':{},'modules':{}}}),self.assertRaisesRegex(ValueError,'global binding changed'):action()
                runtime=portable.package/'version.py';raw=runtime.read_bytes();prior=runtime.stat()
                runtime.write_bytes(b'changed');os.utime(runtime,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'SHA256'):action()
                runtime.write_bytes(raw);os.utime(runtime,ns=(prior.st_atime_ns,prior.st_mtime_ns))
                manifest=Path(own['root'])/'execution.json';raw=manifest.read_bytes();manifest.write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError,'SHA256'):action()
                manifest.write_bytes(raw);f.unchanged_originals(self)

    def test_source_order_bundle_loader_and_no_global_rebinding(self):
        source=PATH.read_text();tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        text=lambda name:ast.get_source_segment(source,nodes[name])
        score=text('score_exports')
        self.assertLess(score.index('archived_replay('),score.index('quality_after_readiness('))
        self.assertLess(score.index('repeated=native.read_wires('),score.index('quality_after_readiness('))
        self.assertLess(score.index("launch['stage'] == 'full' and immediate"),score.index('paired_intervals('))
        export=text('native_export');self.assertIn('for pass_index in range(2)',export)
        self.assertIn("portable.load_inference(directory,endpoint['bundle']['sha256'],'cuda')",export)
        self.assertIn('with bundle_reads_only(context,endpoint)',export);self.assertIn('require_exact=True',export)
        self.assertNotIn('native_raw',export);self.assertNotIn('cache_rows',export)
        exit_source=text('exit_rehash');self.assertIn("api.exit_rehash(t['fit_context'])",exit_source)
        self.assertIn("require_exact=context['args'].phase == 'export'",exit_source)
        for forbidden in ('reset_peak','globals()[','if False','torch.load('):
            if forbidden != 'torch.load(':self.assertNotIn(forbidden,source)
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertTrue(all(n.name.split('.')[0] not in e.NATIVE for n in node.names))
        self.assertEqual(e.TRAIN_FILES,{'train_siglip2_compact_ranking.py','test_siglip2_compact_ranking.py'})


if __name__ == '__main__':
    unittest.main()
